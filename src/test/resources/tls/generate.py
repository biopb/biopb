"""Regenerate the TLS fixtures TlsTrustTest serves and trusts.

Throwaway test material: the keys protect nothing. Run with any Python that has
``cryptography``; the files land beside this script.

  server       self-signed leaf, SANs localhost + 127.0.0.1
  othername    self-signed leaf whose only SAN is other.example (name mismatch)
  expired      self-signed leaf for localhost, notAfter in 2020
  ca / ca-leaf a private CA and a leaf it signed for other.example only
"""

import datetime
import ipaddress
from pathlib import Path

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import NameOID

HERE = Path(__file__).parent
UTC = datetime.timezone.utc
LONG_AGO = datetime.datetime(2019, 1, 1, tzinfo=UTC)
FAR = datetime.datetime(2126, 1, 1, tzinfo=UTC)


def name(common_name):
    return x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, common_name)])


def mint(stem, common_name, sans, *, not_after=FAR, issuer=None, ca=False):
    key = ec.generate_private_key(ec.SECP256R1())
    builder = (
        x509.CertificateBuilder()
        .subject_name(name(common_name))
        .issuer_name(issuer[1].subject if issuer else name(common_name))
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(LONG_AGO)
        .not_valid_after(not_after)
        .add_extension(x509.BasicConstraints(ca=ca, path_length=None), critical=True)
    )
    if sans:
        builder = builder.add_extension(
            x509.SubjectAlternativeName(sans), critical=False
        )
    signer = issuer[0] if issuer else key
    cert = builder.sign(signer, hashes.SHA256())
    (HERE / f"{stem}.pem").write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    (HERE / f"{stem}.key").write_bytes(
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )
    return key, cert


localhost = [
    x509.DNSName("localhost"),
    x509.IPAddress(ipaddress.ip_address("127.0.0.1")),
]
mint("server", "localhost", localhost)
mint("othername", "other", [x509.DNSName("other.example")])
mint(
    "expired",
    "localhost",
    localhost,
    not_after=datetime.datetime(2020, 1, 1, tzinfo=UTC),
)
ca = mint("ca", "biopb test ca", None, ca=True)
mint("ca-leaf", "other", [x509.DNSName("other.example")], issuer=ca)
