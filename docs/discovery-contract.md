# Finding the data plane: the discovery contract

How a client on this machine finds the tensor (data) plane and the token for
it. It is files, environment variables and two HTTP calls, so any language can
implement it; `biopb.control` is the Python one, and `biopb.tensor.Connection`
dials what it returns.

## State directory

`$BIOPB_STATE_HOME/biopb` when set (it must be absolute), else
`~/.local/state/biopb`, on every platform. `XDG_*` is not read.

## The control

Host and port, each resolved on its own: `BIOPB_CONTROL_HOST` /
`BIOPB_CONTROL_PORT`, then `control.json` in the state directory
(`{"host", "port", …}`, written by a serving control and left behind by a
crashed one, so a hint to probe, not proof), then `127.0.0.1:8813`.

## The plane's address

First answer wins:

1. `$BIOPB_TENSOR_URL`;
2. the control's `GET /health` (unauthenticated): `data_plane.grpc_url`;
3. `127.0.0.1:8815`, scheme probed off the socket.

A client that reads data stops at 2: no control means no plane to connect to
(#628), and `Connection` says so. Step 3 is for diagnostics, the `biopb tensor`
commands.

## The token

First answer wins: `$BIOPB_TENSOR_TOKEN`, then `tensor-server.token` in the
state directory (owner-only, one line) **only when the address came from the
control**, else none. It goes to the control as the `X-Biopb-Token` header and
to the plane as a Flight bearer token.

## Starting the plane

`POST /api/data_plane/ensure?client_timeout=<seconds>`, with the token header
when there is one. The control answers before `client_timeout`; a 200 carries
`{"data_plane": {…, "grpc_url"}}`.

## A local TLS plane

For a loopback `grpcs://` address the client verifies the certificate the plane
presents against a SHA-256 fingerprint it reads locally: `tls-served.json` in
the state directory, keyed by port, else the digest of `tls/server-cert.pem`. A
local TLS plane with neither is an error, not a fallback to trust-on-first-use.

## Stability

The files, variables, header, query parameter and the `data_plane.grpc_url`
field above are the contract: fields are added, never renamed or removed, and a
change is an SDK release. The rest of `/health` (for example `auth_required`,
`chat_proxied`) is the web UI's and is not part of it.
