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

import uuid
from urllib.parse import urlsplit

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
    """
    merged: dict = {}
    for entry in (*_defaults_for(base_url), *(configured or ())):
        name, sep, value = str(entry).partition(":")
        name = name.strip()
        if not name or not sep:
            # Not "Name: value". Skipped rather than raised: this comes from a
            # config file a person edits by hand, and one malformed line should
            # not take the chat pane down with it.
            continue
        value = value.strip().replace("{session}", session)
        if not value:
            merged.pop(name.lower(), None)
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
