"""One handle on the data plane, shared by whatever in a process reads it.

A :class:`Connection` holds the live :class:`TensorFlightClient`, the URL it
dials and, when it has none, why. It caches nothing else: the catalog, health
and sources are the plane's to answer, asked on the client.

Where the plane is comes from, in order: ``$BIOPB_TENSOR_URL``, then the
control (:func:`biopb.control.ensure_data_plane`), which also brings the plane
up. An address from anywhere but the control is dialed with an explicit token
or ``$BIOPB_TENSOR_TOKEN``, never with the control's credential file.

``connect`` blocks on the network. It is meant to run off a GUI thread, and
while it does, readers on other threads see the previous ``client`` until the
new one is ready.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Optional

from biopb.control import _data_plane, ensure_data_plane

from .client import TensorFlightClient

logger = logging.getLogger(__name__)

#: How long :meth:`Connection.connect` waits for a plane it was told is coming
#: up: the control's ensure, then the plane's own scan.
START_TIMEOUT_S = 60.0

NO_CONTROL_MESSAGE = (
    "No biopb control plane is running, so there is no data plane to connect to. "
    "Run `biopb control start`."
)

# Lowercased substrings that mark a failure as authentication, or as an
# unreachable server, across the Flight/gRPC stacks.
_AUTH_ERROR_MARKERS = (
    "unauthenticat",
    "unauthoriz",
    "permission denied",
    "invalid token",
    "missing token",
)
_UNREACHABLE_MARKERS = (
    "unavailable",
    "refused",
    "failed to connect",
    "deadline",
    "timed out",
    "timeout",
)


class _Starting(Exception):
    """The plane answered but is not ``SERVING`` yet."""


def _starting_message(health: dict) -> str:
    msg = (
        "Tensor server is starting — scanning its data folder; this can take "
        "a while for large catalogs."
    )
    bits = []
    if health.get("source_count") is not None:
        bits.append(f"{health['source_count']} sources registered so far")
    if health.get("uptime_seconds") is not None:
        bits.append(f"up {int(health['uptime_seconds'])}s")
    return msg + (" (" + ", ".join(bits) + ")" if bits else "")


def connect_error_message(
    exc: Exception, url: str, token: Optional[str], *, origin: str = "control"
) -> str:
    """Why a dial to *url* failed, phrased for the person who has to fix it.

    *origin* is where the address came from — ``"control"``, ``"env"``
    (``$BIOPB_TENSOR_URL``) or ``"manual"`` — because the fix for a missing
    token differs: only the control's address was offered its credential file.
    """
    # By type first: an unreadable cert says "Permission denied", which the auth
    # markers below would otherwise claim.
    if isinstance(exc, _data_plane.LocalTrustError):
        return str(exc)

    text = f"{type(exc).__name__}: {exc}".strip()
    low = text.lower()
    if any(m in low for m in _AUTH_ERROR_MARKERS):
        if token:
            return (
                f"Authentication failed: the tensor server at {url} rejected the token."
            )
        if origin == "env":
            return (
                f"Authentication required: the tensor server at {url} needs a token. "
                f"Set ${_data_plane.ENV_TOKEN}. This endpoint came from "
                f"${_data_plane.ENV_URL}, so it bypassed the control plane and the "
                "control's local credential file was not used for it."
            )
        if origin == "manual":
            return f"Authentication required: the tensor server at {url} needs a token."
        return (
            f"Authentication required: the tensor server at {url} needs a token, "
            "but the control plane's credential file held none. Restart the "
            f"control (`biopb control start`), or set ${_data_plane.ENV_TOKEN}."
        )
    if any(m in low for m in _UNREACHABLE_MARKERS):
        return f"Cannot reach the tensor server at {url} — is it running?"
    return f"Could not connect to the tensor server at {url}: {text}"


class Connection:
    """The data plane this process reads: ``client``, ``url``, ``last_message``.

    ``client`` is ``None`` until a :meth:`connect` succeeds, and again after
    one fails; ``last_message`` then says why.
    """

    def __init__(self) -> None:
        self.client: Optional[TensorFlightClient] = None
        self.url: Optional[str] = None
        self.last_message: str = ""

    def connect(
        self,
        url: Optional[str] = None,
        token: Optional[str] = None,
        *,
        timeout: float = START_TIMEOUT_S,
    ) -> bool:
        """Dial the data plane; ``True`` when ``client`` is ready to use.

        With no *url*, ``$BIOPB_TENSOR_URL`` or else the control names the
        plane. A *url* is dialed as given, with *token* and nothing else.

        A plane the control named is waited on through *timeout*, since the
        control may have just started it; one named anywhere else is not, except
        through a scan it reports as ``STARTING``. Never raises.
        """
        if url is not None:
            return self._dial(url, token, origin="manual", timeout=timeout)
        env = os.environ.get(_data_plane.ENV_URL, "").strip()
        if env:
            token = _data_plane.resolve_token(allow_credential_file=False)
            return self._dial(env, token, origin="env", timeout=timeout)
        plane = ensure_data_plane(timeout=timeout)
        if plane is None:
            self.client = None
            self.url = None
            self.last_message = NO_CONTROL_MESSAGE
            return False
        return self._dial(
            plane["url"], plane["token"], origin="control", timeout=timeout
        )

    def _dial(
        self, url: str, token: Optional[str], *, origin: str, timeout: float
    ) -> bool:
        self.url = url
        deadline = time.monotonic() + timeout
        interval = 0.5
        while True:
            try:
                self.client = self._open(url, token)
                self.last_message = ""
                return True
            except _Starting as exc:
                self.last_message = str(exc)
            except Exception as exc:  # noqa: BLE001 - recorded, not raised
                self.last_message = connect_error_message(
                    exc, url, token, origin=origin
                )
                # Only the control's plane may still be binding its port.
                if origin != "control":
                    break
            if time.monotonic() >= deadline:
                break
            time.sleep(interval)
            interval = min(interval * 2, 5.0)
        self.client = None
        logger.info("data plane at %s not connected: %s", url, self.last_message)
        return False

    @staticmethod
    def _open(url: str, token: Optional[str]) -> TensorFlightClient:
        """A client for *url* that the plane has answered, and accepted."""
        client = TensorFlightClient(
            url, token=token, tls_fingerprint=_data_plane.local_fingerprint(url)
        )
        try:
            health = client.health_check()
            if health.get("status", "SERVING") != "SERVING":
                raise _Starting(_starting_message(health))
            # health answers anyone; this is the call that checks the token.
            client.query_sources("SELECT 1 FROM sources LIMIT 0")
        except BaseException:
            client.close()
            raise
        return client
