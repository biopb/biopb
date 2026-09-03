"""Headers an OpenAI-compatible endpoint needs that the OpenAI API never defined.

Two clients in this package speak that API — the chat loop's provider adapter
(``mcp/_model.py``, hand-rolled ``httpx``) and the agentbench harness
(``_tests/agentbench``, the ``openai`` SDK). Both may sit behind a *gateway*
rather than a vendor, and a gateway generally wants something of its own on
every request: a session id to attribute the traffic to one conversation, an
attribution header, a routing tag. Nothing in the OpenAI shape carries those.

So this is the one place that answers "what else goes on the wire, and what is
in it", and both clients read it. Two transports, one policy — the alternative
is what we had, where a header added to one client silently left the other
broken.

**Vendors are data, not code.** :data:`_KNOWN_GATEWAYS` maps a host to the
headers it requires, so supporting the next one is a row rather than a release,
and ``chat.extra_headers`` (or the harness's ``*_HEADERS`` variables) lets a
user say it before we have the row at all. Configured entries merge over the
defaults by name, so overriding one of a gateway's headers keeps the rest, and
setting a name to nothing removes it.

**The session id is not really about headers.** A conversation needs a stable
identity for its own sake — correlating a trace, reading a log, telling two bench
arms apart — and a header that wants one is a consumer of it. So it is minted
where a conversation begins (``_chat.reset``, a bench backend's construction)
and passed in here, rather than invented per request.
"""

import logging
import uuid
from urllib.parse import urlsplit

logger = logging.getLogger(__name__)

#: A field name is a ``token`` (RFC 9110 §5.1, §5.6.2); these are its characters
#: besides ALPHA and DIGIT. Checked rather than assumed: these names are typed
#: by hand into a config file or an environment variable, and what a transport
#: does with one that is not a token differs -- ``httpx`` and the SDK do not
#: agree on whether it raises, and a turn is a bad place to find out.
_TCHAR = frozenset("!#$%&'*+-.^_`|~")

#: Header names this will not carry, whatever asks for it. These are how a
#: credential travels, both clients already supply their own, and the values
#: here reach a browser: ``chat.extra_headers`` lives in the config file that
#: ``GET /api/mcp_config`` serves whole. Refusing them is what makes the
#: setting's "no secrets here" a property rather than a request -- a key put
#: here would otherwise *work*, which is the mistake that gets one rendered in
#: an admin page. Keys belong in the credential file (``_model.KEY_NAME``).
_CREDENTIAL_HEADERS = frozenset(
    {"authorization", "proxy-authorization", "x-api-key", "api-key"}
)

#: Hosts that require headers of their own, and what they require. Values are
#: ``"Name: value"``, with ``{session}`` substituted per :func:`extra_headers`.
#: Subdomains match, so one row covers a vendor's regional endpoints.
#:
#: Deliberately short: a row goes in when we have seen the requirement, not
#: when we have read a doc. Others known to want headers, unverified here:
#: OpenRouter (``HTTP-Referer`` / ``X-Title``, attribution only), Helicone and
#: Portkey (session and trace ids), Cloudflare AI Gateway (``cf-aig-metadata``).
_KNOWN_GATEWAYS = {
    # Announced 2026-09: requests without it "may error" from 09-06.
    "opencode.ai": ("x-opencode-session: {session}",),
}


def new_session_id() -> str:
    """An opaque id for one conversation, stable for as long as it lasts.

    Random rather than derived from anything about the session: it is sent to a
    third party on every request, so it must not carry a hostname, a path, a
    user or a key. Prefixed so it is recognisable as ours in someone else's log
    when we have to ask them about it.
    """
    return f"biopb-{uuid.uuid4().hex}"


def _is_token(name: str) -> bool:
    """Whether *name* is a valid HTTP field name."""
    return bool(name) and all(
        c.isascii() and (c.isalnum() or c in _TCHAR) for c in name
    )


def _is_field_value(value: str) -> bool:
    """Whether *value* can go on the wire as a field value.

    The check that matters is the absence of CR and LF: a value carrying either
    is not one header but two, which is header injection, and the value here can
    come from an environment variable. The rest of C0 and DEL go with them --
    they are not field-vchar, and no gateway asks for one.
    """
    return not any((ord(c) < 0x20 and c != "\t") or ord(c) == 0x7F for c in value)


def _defaults_for(base_url: str) -> tuple:
    host = (urlsplit(base_url).hostname or "").lower()
    for known, headers in _KNOWN_GATEWAYS.items():
        if host == known or host.endswith(f".{known}"):
            return headers
    return ()


def extra_headers(base_url: str, configured=(), *, session: str = "") -> dict:
    """The extra headers for *base_url*, as ``{name: value}``.

    *configured* is a sequence of ``"Name: value"`` strings — the user's, from
    ``chat.extra_headers`` or a harness variable — merged over whatever
    :data:`_KNOWN_GATEWAYS` supplies for the host, matched case-insensitively so
    an override lands on the header it means to replace. An entry with an empty
    value drops that header, which is how a user turns off a default that has
    started doing harm before we can ship a release removing it.

    ``{session}`` in a value becomes *session*. A value that is empty once
    substituted is dropped rather than sent blank: a gateway reading a header it
    requires as present-but-empty gives a worse error than one reading it as
    absent, and an empty one here means we had no conversation to name.

    **Every entry is validated, and a bad one is skipped rather than raised.**
    This is hand-edited text, and one malformed line should not take a turn --
    or a whole bench run -- down with it. Skipped entries are logged by name, so
    a header that is quietly not being sent is findable. No entry's text is ever
    logged -- the one thing that must not be put here is the thing most likely
    to be, and a log line is a file.
    """
    merged: dict = {}
    entries = (*_defaults_for(base_url), *(configured or ()))
    for position, entry in enumerate(entries, start=1):
        name, sep, value = str(entry).partition(":")
        name = name.strip()
        if not sep or not _is_token(name):
            # Its position, not its text: an entry malformed enough to reach
            # here is also where a pasted key ends up, and a log line is a file.
            logger.warning(
                "ignoring header entry %d: not 'Name: value' with a valid field name",
                position,
            )
            continue
        if name.lower() in _CREDENTIAL_HEADERS:
            logger.warning(
                "ignoring %s: credentials belong in the credential file, not in "
                "configuration that is served to a browser",
                name,
            )
            continue
        value = value.strip().replace("{session}", session)
        if not value:
            merged.pop(name.lower(), None)
            continue
        if not _is_field_value(value):
            logger.warning("ignoring header %s: its value is not a field value", name)
            continue
        merged[name.lower()] = (name, value)
    return dict(merged.values())


def parse_env_headers(raw: str) -> tuple:
    """``"Name: value"`` lines from one environment variable.

    Newline-separated, because a header value may contain any of the characters
    a one-line separator would have to be — including the ``:`` that separates
    the name from the value, and the ``;`` and ``,`` that appear inside dates
    and lists. In dotenv files, write newlines as the two characters ``\\n``;
    the small dotenv reader intentionally reads one physical line at a time.
    One header is the common case and needs no separator at all.
    """
    raw = raw.replace("\\n", "\n")
    return tuple(line.strip() for line in raw.splitlines() if line.strip())
