"""Data-plane endpoint resolution — shared, stdlib-only.

Where the tensor (data) plane listens, what credential reaches it, and what
anchors its TLS: one answer, resolved in one place.

It used to be four places (biopb/biopb#615). ``biopb server cache-stats`` and the
four ``biopb tensor`` commands each *reconstructed* the endpoint from a default —
hardcoded ``grpc://``, hardcoded port, token from the environment only — while
biopb-mcp *asked the control*, which is the only site that can be right: the
control chose the bind, the port, and the scheme, so its supervisor snapshot is
the fact and everything else is a guess that goes stale the moment ``--base-port``
or ``--tls`` moves. Reconstruction also cannot express what it does not know: a
plane on a moved base is invisible to it, and a ``grpcs://`` plane is dialed
plaintext and reported as down.

So the order here is **ask, then guess**:

1. an explicit override (a ``--server`` flag), then ``BIOPB_TENSOR_URL``;
2. the control's ``GET /health`` -> ``data_plane.grpc_url`` — authoritative,
   carrying host, port *and* scheme;
3. the default base-port endpoint, with the scheme probed off the socket.

Step 3 covers the one case discovery cannot: a plane launched directly, outside
any control, which persists nothing for a reader to find. The port is still a
guess there (nothing records a directly-chosen one — pass ``--server`` for that),
but the *scheme* need not be, so :func:`probe_scheme` asks the listener instead
of assuming. Every :class:`Endpoint` records which of the three answered, so a
failed dial can say where the address came from rather than blaming the server.

**The credential follows the address.** Where the endpoint came from decides
whether the control's credential file is readable for it: that file is the token
the control wrote for the plane *it* owns, so it travels only with the endpoint
the control itself named. An override or ``$BIOPB_TENSOR_URL`` deliberately
routed around the control — it may name another user's server, or a lab store
across the network — and quietly attaching this machine's credential to that dial
would send a local secret somewhere it was never issued for. Those endpoints
carry an explicit token or none.

Deliberately stdlib-only, like ``_endpoints`` / ``_credentials`` / ``_locations``
beside it: it is imported by the core CLI and by biopb-mcp, and importing it must
never drag in pyarrow. Classifying a *Flight* failure therefore does not live
here — that belongs to the layer that already has the client (see
``biopb.tensor.cli._dial_error``).
"""

from __future__ import annotations

import json
import os
import urllib.request
from dataclasses import dataclass
from typing import Optional
from urllib.parse import urlparse

from ._endpoints import BASE_DEFAULT_PORT, control_base_url, flight_port_for

# The one env var naming the data plane. ``BIOPB_TENSOR_SERVER`` (the old
# ``biopb tensor`` spelling) was retired with #615: two names for one concept, on
# commands that dial the same plane, is how a user ends up setting the one the
# command they are running does not read.
ENV_URL = "BIOPB_TENSOR_URL"
ENV_TOKEN = "BIOPB_TENSOR_TOKEN"

# Hosts that mean "this machine". A plane here has its TLS cert on this disk, so
# it is trusted from the file rather than pinned from the wire (:func:`local_ca`).
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


def default_url() -> str:
    """The endpoint a default deployment puts the data plane on (``grpc://…:8815``)."""
    return f"grpc://127.0.0.1:{flight_port_for(BASE_DEFAULT_PORT)}"


def is_local_url(url: str) -> bool:
    """Whether *url* points at this machine."""
    try:
        host = urlparse(url).hostname
    except ValueError:
        return False
    return host is None or host in _LOCAL_HOSTS


def probe_scheme(host: str, port: int, timeout: float = 0.5) -> Optional[str]:
    """``"grpcs"`` / ``"grpc"`` by asking the listener, or ``None`` if nothing is there.

    The scheme is the one thing a directly-launched plane still tells you for
    free: a TLS listener completes a handshake and a plaintext one does not. So
    ask it, rather than inferring from the presence of a cert on disk — a cert
    minted once by ``cert init`` says nothing about whether the running plane was
    started with ``--tls``, and guessing wrong in *either* direction produces the
    same "server unreachable" that #615 was filed for.

    Certificate validation is deliberately off: this asks a yes/no question about
    the wire protocol, and the answer decides which scheme to dial. Trust is
    established afterwards, on the real connection, by :func:`local_ca` or TOFU.
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


def local_ca(url: str) -> Optional[bytes]:
    """Explicit TLS trust anchor for a *local* plane, read off local disk.

    A loopback ``grpcs://`` plane is this machine's own and the certificate it
    serves is already on this machine's disk — so trust it directly instead of
    pinning whatever the handshake presents (TOFU). An anchor read from local disk
    is strictly stronger than one learned from the wire, and it keeps the client
    out of the shared pin store, which would otherwise strand it the moment an
    operator rotated the cert with ``cert init --force``.

    ``None`` — leaving TOFU in charge — for a plaintext endpoint or a remote one,
    whose cert is not on this disk and cannot be. Raises :class:`LocalTrustError`
    when a local plane is TLS but its cert is unreadable: silently falling back to
    TOFU there would trade a verified anchor for an unverified one exactly where
    the strong option was meant to apply.

    Known edge: a loopback ``grpcs://`` URL that is really an ``ssh -L`` tunnel to
    a *remote* plane is indistinguishable from a local one by host alone, so it is
    anchored on the local cert and the handshake fails. Loud and fixable (dial the
    plane directly, or tunnel to a non-loopback alias), not silent.
    """
    if not url.lower().startswith("grpcs://") or not is_local_url(url):
        return None

    from ._locations import tls_server_cert

    cert_path = tls_server_cert()
    try:
        pem = cert_path.read_bytes()
    except OSError as exc:
        raise LocalTrustError(
            f"The local data plane at {url} serves TLS, but its certificate could "
            f"not be read from {cert_path} ({exc}). A local plane is trusted from "
            "its cert on disk, not by pinning it from the wire — so this is not "
            "retried as trust-on-first-use. Check the state dir is the one the "
            "server writes to (XDG_STATE_HOME), or re-mint the cert with "
            "`biopb-tensor-server cert init`."
        ) from exc
    if not pem.strip():
        raise LocalTrustError(
            f"The local data plane's TLS certificate at {cert_path} is empty."
        )
    return pem


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
    "env": f"from ${ENV_URL}",
    "control": "from the control plane",
    "default": "the default endpoint — no control plane answered",
}


@dataclass(frozen=True)
class Endpoint:
    """A resolved data-plane dial: where, with what credential, on what anchor."""

    url: str
    token: Optional[str] = None
    tls_ca_pem: Optional[bytes] = None
    origin: str = "default"

    @property
    def origin_note(self) -> str:
        """Human phrase for :attr:`origin`, for error messages."""
        return _ORIGINS.get(self.origin, self.origin)


def resolve(
    override: Optional[str] = None,
    token: Optional[str] = None,
    *,
    timeout: float = 1.0,
    probe: bool = True,
) -> Endpoint:
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

    Raises :class:`LocalTrustError` when the resolved plane is local TLS but its
    certificate cannot be read — see :func:`local_ca`.
    """
    url, origin = _resolve_url(override, timeout=timeout, probe=probe)
    return Endpoint(
        url=url,
        # The credential file is the control's handoff for the plane IT owns, so
        # it travels only with an endpoint the control named. An address given on
        # the command line or in the environment routed around the control on
        # purpose -- possibly to somebody else's server -- and attaching this
        # machine's token to that dial would hand a local credential to a host the
        # user never authorized it for. Those endpoints authenticate explicitly or
        # not at all.
        token=resolve_token(token, allow_credential_file=origin == "control"),
        tls_ca_pem=local_ca(url),
        origin=origin,
    )


def _resolve_url(
    override: Optional[str], *, timeout: float, probe: bool
) -> tuple[str, str]:
    if override:
        return override, "flag"
    env = os.environ.get(ENV_URL)
    if env:
        return env, "env"
    published = control_grpc_url(timeout=timeout)
    if published:
        return published, "control"
    port = flight_port_for(BASE_DEFAULT_PORT)
    scheme = probe_scheme("127.0.0.1", port) if probe else None
    return f"{scheme or 'grpc'}://127.0.0.1:{port}", "default"


def resolve_token(
    explicit: Optional[str] = None, *, allow_credential_file: bool = True
) -> Optional[str]:
    """The data-plane token: explicit -> ``BIOPB_TENSOR_TOKEN`` -> credential file.

    The credential file is what closes the gap for a *local plane behind a token*
    (biopb/biopb#470): the control writes the resolved token to an owner-only file
    in the user's state dir, so a client that never inherited the control's
    environment can still authenticate. The core CLI read only the env var until
    #615, which is why a token-gated local plane reported itself as unreachable.

    ``allow_credential_file=False`` drops that last step, for a caller dialing an
    endpoint the control did not name: the file holds *this machine's* credential
    for the control's own plane, and it has no business being sent to an address
    the user pointed elsewhere. The two explicit sources still apply — someone
    naming a server can also name its token.

    ``None`` — unauthenticated, correct for a tokenless local plane — when nothing
    yields one. A blank value is ``None``, never ``""``: an empty string would be
    sent as an empty ``Bearer`` header rather than omitted.
    """
    given = (explicit or "").strip() or os.environ.get(ENV_TOKEN, "").strip()
    if given:
        return given
    if not allow_credential_file:
        return None

    from ._credentials import read_credential

    return read_credential()
