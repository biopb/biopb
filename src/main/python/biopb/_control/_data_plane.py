"""Data-plane endpoint resolution — shared, stdlib-only.

Where the tensor (data) plane listens, what credential reaches it, and what
anchors its TLS: one answer, resolved in one place.

The order here is **ask, then guess**:

1. an explicit override (a ``--server`` flag), then ``BIOPB_TENSOR_URL``;
2. the control's ``GET /health`` -> ``data_plane.grpc_url`` — authoritative,
   carrying host, port *and* scheme;
3. the default base-port endpoint, with the scheme probed off the socket.

Step 3 covers the case of a plane launched directly, outside any control.

**The credential follows the address.** An override or ``$BIOPB_TENSOR_URL``
deliberately routed around the control, and quietly attaching this machine's
credential to that dial would send a local secret somewhere it was never issued
for. Those endpoints carry an explicit token or none.

**So does the TLS anchor.** A plane the control named is identified by the
certificate record it published; an address that bypassed the control is trusted
as ``$BIOPB_TENSOR_TLS_CA`` or ``$BIOPB_TENSOR_TLS_FINGERPRINT`` say, else on
first use. This module and :class:`biopb.tensor.Connection` are the two places a
plane is dialed, and both ask :func:`data_plane_trust`.
"""

from __future__ import annotations

import json
import logging
import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

from ._endpoints import BASE_DEFAULT_PORT, control_base_url, flight_port_for

logger = logging.getLogger(__name__)

ENV_TENSOR_URL = "BIOPB_TENSOR_URL"
ENV_TENSOR_TOKEN = "BIOPB_TENSOR_TOKEN"
ENV_TENSOR_TLS_CA = "BIOPB_TENSOR_TLS_CA"
ENV_TENSOR_TLS_FINGERPRINT = "BIOPB_TENSOR_TLS_FINGERPRINT"

_LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1"}


class LocalTrustError(RuntimeError):
    """The local data plane's TLS certificate could not be used as a trust anchor.

    A distinct type because the layers above classify connect failures by
    *substring*, and the most likely cause here — an unreadable cert file —
    stringifies as ``[Errno 13] Permission denied``. That matches an
    authentication marker, so a file-permission problem would be reported as "the
    server needs a token" and send the reader after a credential that has nothing
    to do with it. Matching on the type is exact, and no errno wording can break it.
    """


class TlsConfigError(LocalTrustError):
    """A trust anchor set in the environment cannot be used.

    A :class:`LocalTrustError` so every layer that already reports those by type
    reports this one the same way, rather than by substring.
    """


def default_data_plane_url() -> str:
    """The endpoint a default deployment puts the data plane on (``grpc://…:8815``)."""
    return f"grpc://127.0.0.1:{flight_port_for(BASE_DEFAULT_PORT)}"


def is_local_url(url: str) -> bool:
    """Whether *url* points at this machine."""
    try:
        host = urlparse(url).hostname
    except ValueError:
        return False
    return host is None or host in _LOCAL_HOSTS


def probe_data_plane_scheme(
    host: str, port: int, timeout: float = 0.5
) -> Optional[str]:
    """``"grpcs"`` / ``"grpc"`` by asking the listener, or ``None`` if nothing is there.

    The scheme is the one thing a directly-launched plane still tells you for
    free: a TLS listener completes a handshake and a plaintext one does not. So
    ask it, rather than inferring from the presence of a cert on disk — a cert
    minted once by ``cert init`` says nothing about whether the running plane was
    started with ``--tls``, and guessing wrong in *either* direction produces the
    same "server unreachable" that #615 was filed for.

    Certificate validation is deliberately off: this asks a yes/no question about
    the wire protocol, and the answer decides which scheme to dial. Trust is
    established afterwards, on the real connection, by :func:`local_data_plane_fingerprint`
    or TOFU.
    """
    import socket
    import ssl

    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    try:
        with socket.create_connection((host, port), timeout=timeout) as sock:
            try:
                sock.settimeout(timeout)
                with ctx.wrap_socket(sock, server_hostname=host):
                    return "grpcs"
            except (OSError, ValueError):
                # A plaintext HTTP/2 listener answers a ClientHello with garbage
                # (or a reset). It is listening; it just isn't TLS.
                return "grpc"
    except OSError:
        return None


def local_data_plane_fingerprint(url: str) -> Optional[str]:
    """Identity of the certificate a *local* plane serves, as a SHA-256 digest.

    A loopback ``grpcs://`` plane is this machine's own, so what it serves is
    knowable here rather than something to accept on first sight (TOFU). The
    digest is checked against the certificate the server actually presents on
    every connect, which is strictly stronger than a pin learned from the wire
    and keeps the client out of the shared pin store — where an operator's
    ``cert init --force`` would otherwise strand it.

    A fingerprint rather than the PEM itself, deliberately: handing the client a
    PEM resolves trust entirely offline, which also skips the hostname-override
    probe, and a local client dials loopback. A certificate carrying only the
    host's public name — the ordinary shape of an operator's own cert — then
    fails hostname verification on every connect (biopb/biopb#916).

    Two sources, in order:

    - what the plane **published** for this port (:mod:`biopb._tls_record`),
      which is the only thing that knows about a ``--tls-cert`` the plane was
      handed;
    - failing that, the certificate the plane would have **minted**
      (``state/biopb/tls/server-cert.pem``), which is what a plane too old to
      publish anything serves.

    ``None`` — leaving TOFU in charge — for a plaintext endpoint or a remote one,
    whose certificate is not on this disk and cannot be. Raises
    :class:`LocalTrustError` when a local plane is TLS and neither source
    answers: silently falling back to TOFU there would trade a verified identity
    for an unverified one exactly where the strong option was meant to apply.

    Known edge: a loopback ``grpcs://`` URL that is really an ``ssh -L`` tunnel to
    a *remote* plane is indistinguishable from a local one by host alone, so it is
    checked against the local plane's identity and fails. Loud and fixable (dial
    the plane directly, or tunnel to a non-loopback alias), not silent.
    """
    if not url.lower().startswith("grpcs://") or not is_local_url(url):
        return None

    from .. import _tls_material, _tls_record
    from .._locations import tls_served_certs, tls_server_cert

    port = urlparse(url).port
    if port is not None:
        published = _tls_record.lookup(port)
        if published:
            return published

    cert_path = tls_server_cert()
    try:
        pem = cert_path.read_bytes()
    except OSError as exc:
        raise LocalTrustError(
            f"The local data plane at {url} serves TLS, but nothing on this "
            f"machine says which certificate: no record for port {port} in "
            f"{tls_served_certs()}, and no certificate at {cert_path} ({exc}). A "
            "local plane is verified against what it serves, not pinned from the "
            "wire, so this is not retried as trust-on-first-use. Check the state "
            "dir is the one the server writes to (BIOPB_STATE_HOME); a plane "
            "publishes its record at startup, so restart one that predates this "
            "build, or mint the certificate with `biopb-tensor-server cert init`."
        ) from exc
    if not pem.strip():
        raise LocalTrustError(
            f"The local data plane's TLS certificate at {cert_path} is empty."
        )
    try:
        return _tls_material.fingerprint(_tls_material.leaf_pem(pem))
    except ValueError as exc:
        raise LocalTrustError(
            f"The local data plane's TLS certificate at {cert_path} is not "
            f"readable as PEM ({exc})."
        ) from exc


def control_grpc_url(timeout: float = 1.0) -> Optional[str]:
    """The data-plane URL the control publishes, or ``None`` if no control answers.

    GETs the control's bare, unauthenticated ``/health`` and reads
    ``data_plane.grpc_url`` from the supervisor snapshot — the single source of
    truth for where the plane lives (biopb/biopb#413), because the control is what
    chose the bind, the port, and the scheme. Best-effort: an absent, slow, or
    malformed control is "no answer", never an exception, so the caller falls
    through to the default rather than failing to resolve anything at all.
    """
    try:
        with urllib.request.urlopen(
            f"{control_base_url()}/health", timeout=timeout
        ) as resp:
            if resp.status != 200:
                return None
            payload = json.loads(resp.read().decode())
    except Exception:  # noqa: BLE001 - best-effort discovery; the caller falls back
        return None
    data_plane = payload.get("data_plane") if isinstance(payload, dict) else None
    if isinstance(data_plane, dict):
        url = data_plane.get("grpc_url")
        if isinstance(url, str) and url:
            return url
    return None


# How an endpoint's URL was arrived at, phrased to be read inside an error
# message ("… at grpc://… (from the control plane)"). The point is that a failed
# dial can name where the address came from: "unreachable" means something
# different for an address the control published than for a guessed default.
_ORIGINS = {
    "flag": "given on the command line",
    "env": f"from ${ENV_TENSOR_URL}",
    "control": "from the control plane",
    "default": "the default endpoint — no control plane answered",
}


@dataclass(frozen=True)
class DataPlaneEndpoint:
    """A resolved data-plane dial: where, with what credential, on what anchor."""

    url: str
    token: Optional[str] = None
    tls_fingerprint: Optional[str] = None
    tls_ca_pem: Optional[bytes] = None
    origin: str = "default"

    @property
    def origin_note(self) -> str:
        """Human phrase for :attr:`origin`, for error messages."""
        return _ORIGINS.get(self.origin, self.origin)


def resolve_data_plane(
    override: Optional[str] = None,
    token: Optional[str] = None,
    *,
    timeout: float = 1.0,
    probe: bool = True,
) -> DataPlaneEndpoint:
    """Resolve the data-plane endpoint: override -> env -> control -> default.

    *override* is an explicit ``--server``-style address and wins over everything;
    it is the escape hatch for a plane nothing records — one launched directly on
    a custom port. *token* is an explicit ``--token`` and likewise wins over the
    environment and over the control's credential file.

    The credential file is read only for an endpoint the control named (see the
    module docstring): an address that bypassed the control is dialed with an
    explicit token or with none.

    Set ``probe=False`` to skip the socket scheme probe on the default fallback
    (a caller that only wants to *name* the endpoint, not dial it).

    Raises :class:`LocalTrustError` when the resolved plane is local TLS but
    nothing on this machine says what it serves — see :func:`local_data_plane_fingerprint`.
    The TLS anchor is :func:`data_plane_trust`'s.
    """
    url, origin = _resolve_url(override, timeout=timeout, probe=probe)
    trust = data_plane_trust(url, origin)
    return DataPlaneEndpoint(
        url=url,
        # The credential file is the control's handoff for the plane IT owns, so
        # it travels only with an endpoint the control named. An address given on
        # the command line or in the environment routed around the control on
        # purpose -- possibly to somebody else's server -- and attaching this
        # machine's token to that dial would hand a local credential to a host the
        # user never authorized it for. Those endpoints authenticate explicitly or
        # not at all.
        token=resolve_data_plane_token(
            token, allow_credential_file=origin == "control"
        ),
        tls_fingerprint=trust.fingerprint,
        tls_ca_pem=trust.ca_pem,
        origin=origin,
    )


@dataclass(frozen=True)
class TlsAnchor:
    """What to trust of a TLS data plane: a CA/leaf PEM, or a leaf's SHA-256.

    At most one is set; neither leaves the client to trust on first use.
    """

    ca_pem: Optional[bytes] = None
    fingerprint: Optional[str] = None


def configured_tls_anchor() -> TlsAnchor:
    """The anchor the environment names: ``$BIOPB_TENSOR_TLS_CA`` (a PEM file) or
    ``$BIOPB_TENSOR_TLS_FINGERPRINT``, else none.

    For a plane whose certificate this machine cannot see -- a remote one --
    where trust-on-first-use is only as good as a pin store that outlives the
    process, which a container or a scheduler job does not have. When both are
    set the CA wins, as it does for the constructor and a server's upstream
    profile, and the fingerprint is reported as ignored. Raises
    :class:`TlsConfigError` when the CA file is unusable.
    """
    ca_path = os.environ.get(ENV_TENSOR_TLS_CA, "").strip()
    fingerprint = os.environ.get(ENV_TENSOR_TLS_FINGERPRINT, "").strip()
    if ca_path and fingerprint:
        logger.warning(
            "Both $%s and $%s are set; using the CA and ignoring the fingerprint.",
            ENV_TENSOR_TLS_CA,
            ENV_TENSOR_TLS_FINGERPRINT,
        )
    if ca_path:
        from .. import _tls_material

        try:
            return TlsAnchor(
                ca_pem=_tls_material.read_pem(Path(ca_path), f"${ENV_TENSOR_TLS_CA}")
            )
        except _tls_material.TlsMaterialError as exc:
            raise TlsConfigError(str(exc)) from exc
    return TlsAnchor(fingerprint=fingerprint or None)


def data_plane_trust(url: str, origin: str) -> TlsAnchor:
    """The TLS anchor to dial *url* with, given where its address came from.

    A plane the control named is this machine's own and is identified by the
    record it published (:func:`local_data_plane_fingerprint`); nothing in the
    environment overrides that, so a stale variable cannot point a local client
    at an old certificate. Any other address routed around the control, so an
    anchor the environment configures wins, and without one the local record
    (loopback ``grpcs://``) or trust on first use decides.
    """
    if origin != "control":
        configured = configured_tls_anchor()
        if configured.ca_pem or configured.fingerprint:
            return configured
    return TlsAnchor(fingerprint=local_data_plane_fingerprint(url))


def _resolve_url(
    override: Optional[str], *, timeout: float, probe: bool
) -> tuple[str, str]:
    if override:
        return override, "flag"
    env = os.environ.get(ENV_TENSOR_URL)
    if env:
        return env, "env"
    published = control_grpc_url(timeout=timeout)
    if published:
        return published, "control"
    port = flight_port_for(BASE_DEFAULT_PORT)
    scheme = probe_data_plane_scheme("127.0.0.1", port) if probe else None
    return f"{scheme or 'grpc'}://127.0.0.1:{port}", "default"


def resolve_data_plane_token(
    explicit: Optional[str] = None, *, allow_credential_file: bool = True
) -> Optional[str]:
    """The data-plane token: explicit -> ``BIOPB_TENSOR_TOKEN`` -> credential file.

    The credential file is what closes the gap for a *local plane behind a token*
    (biopb/biopb#470): the control writes the resolved token to an owner-only file
    in the user's state dir, so a client that never inherited the control's
    environment can still authenticate. The core CLI read only the env var
    until #615, which is why a token-gated local plane reported itself as
    unreachable.

    ``allow_credential_file=False`` drops that last step, for a caller dialing an
    endpoint the control did not name: the file holds *this machine's* credential
    for the control's own plane, and it has no business being sent to an address
    the user pointed elsewhere. The two explicit sources still apply — someone
    naming a server can also name its token.

    ``None`` — unauthenticated, correct for a tokenless local plane — when nothing
    yields one. A blank value is ``None``, never ``""``: an empty string would be
    sent as an empty ``Bearer`` header rather than omitted.
    """
    given = (explicit or "").strip() or os.environ.get(ENV_TENSOR_TOKEN, "").strip()
    if given:
        return given
    if not allow_credential_file:
        return None

    from .._credentials import read_credential

    return read_credential()
