"""The chat loop's provider adapter: one OpenAI-compatible call, and its key.

`_chat.run_turn` takes the model as an injected async callable so the loop can
be tested with no key and no network. This is the real one, and it is small on
purpose — the spike that settled hand-roll-vs-vendor found the model call was
about a dozen lines of ``httpx`` and that every hard problem was biopb-specific
plumbing a framework would not have known about either
(``docs/chat-client-evaluation.md``).

**Where the key lives.** In an owner-only credential file, not in
``mcp-config.json`` and not in the environment. The config file is served whole
by the control's ``GET /api/mcp_config`` so the admin page can edit it — a key
there would be rendered in a browser and cross the very channel it protects. An
environment variable is worse still: ``biopb._credentials`` was written on the
finding that env vars leak through ``/proc/<pid>/environ``, ``ps e`` and every
inherited child, which is why the data-plane token moved out of one. The chat
key gets the same treatment for a sharper reason: it is a *foreign* credential
with billing attached, so a leak reaches past this machine in a way the
data-plane token cannot. ``chat.api_key_env`` still overrides, for CI and
development.

**Two API shapes, chosen by configuration.** ``chat.api`` picks between
chat-completions and the Responses API. The loop only ever sees the
chat-completions shape -- ``(messages, tools) -> assistant message`` -- and the
Responses translation lives here, in both directions, so ``_chat.py`` stays one
loop rather than two. It is configured rather than detected because the
discriminator is bad: a gateway serving both answers a wrong-route call with a
bare 500 (measured on opencode's zen: every ``muse-spark`` 500s on
``/chat/completions`` and answers on ``/responses``, ``big-pickle`` the reverse),
which is exactly what a real outage looks like, and ``GET /models`` returns
identical metadata for both families.
"""

import logging

import httpx
from biopb._credentials import read_credential

from .. import _endpoint
from .._message_shape import describe_messages
from . import _chat
from ._chat import VisionUnsupported

logger = logging.getLogger(__name__)

#: Words a provider reaches for when the image is what it objected to. Matched
#: only against the answer to a request that actually carried one, so the bar is
#: deliberately low: a false positive costs a session its screenshots, a false
#: negative costs it every remaining turn.
_VISION_REFUSALS = ("image", "vision", "multimodal")

#: Credential file name, a sibling of the data plane's ``tensor-server.token``.
KEY_NAME = "chat-provider.token"


class ChatNotConfigured(RuntimeError):
    """Chat is off, has no model, or has no key.

    One exception for all three because they are one situation from the user's
    side — "this is not set up yet" — and the message says which part is missing
    so the answer is actionable rather than a shrug.
    """


def _headers(config, key):
    """What goes on the wire besides the payload.

    The key, plus whatever the endpoint requires of its own (:mod:`_endpoint`).
    Built per call rather than once, for the same reason the key is read per
    call: the conversation this belongs to can change under a long-lived model
    callable, and a stale session id is worse than none -- it attributes this
    thread's traffic to the one before it.
    """
    from .._config import get_setting

    # The key last: `extra_headers` refuses to carry a credential header at
    # all, and this is the second half of that -- configuration cannot displace
    # the key even if the refusal above it is ever relaxed.
    return {
        **_endpoint.extra_headers(
            get_setting(config, "chat.base_url"),
            get_setting(config, "chat.extra_headers"),
            session=_chat.session_id(),
        ),
        "Authorization": f"Bearer {key}",
    }


def _carries_image(messages):
    """Whether this payload has an image part in it."""
    return any(
        isinstance(part, dict) and part.get("type") == "image_url"
        for msg in messages
        for part in (msg.get("content") if isinstance(msg.get("content"), list) else ())
    )


def api_key(config):
    """The provider key: the configured env var first, then the credential file.

    Env first so a developer can override without touching the file, file as the
    supported path. ``None`` when neither has one.
    """
    import os

    from .._config import get_setting

    # No explicit default: passing one to get_setting *disables* the
    # DEFAULT_CONFIG fallback, and a config file with no chat section at all --
    # the common case -- must still get the default env var name.
    name = get_setting(config, "chat.api_key_env") or ""
    return (os.environ.get(name) if name else None) or read_credential(KEY_NAME)


def check_ready(config):
    """Raise :class:`ChatNotConfigured` unless the provider can be reached.

    Called before a turn is accepted rather than at the first model call, so a
    misconfigured install says so instead of taking the user's message, running
    tools against their kernel, and only then failing at the provider.

    Whether chat is *offered* is not asked here: that is ``observe.chat_enabled``
    and it decides whether these routes exist at all, so anything that gets this
    far is on by construction.
    """
    from .._config import get_setting

    if not get_setting(config, "chat.model"):
        raise ChatNotConfigured(
            "No chat model is configured. Set chat.model in mcp-config.json — "
            "there is no default, because guessing one would bill you for a "
            "model you did not choose."
        )
    if not api_key(config):
        from biopb._credentials import credential_file

        raise ChatNotConfigured(
            "No provider key. Write it to "
            f"{credential_file(KEY_NAME)} (owner-only), or set "
            f"${get_setting(config, 'chat.api_key_env')}."
        )


#: How long the model list may take. Shorter than ``chat.request_timeout``,
#: which is sized for a turn: this one is answering a keystroke.
_LIST_TIMEOUT = 10


async def list_models(config):
    """The provider's own catalogue, or ``[]`` when it does not publish one.

    ``GET {base_url}/models`` is the OpenAI-compatible spelling and most servers
    implement it, but it is optional -- an endpoint that 404s here still serves
    completions perfectly well. So every failure is an empty list rather than an
    error: the caller's job is to offer names it is sure of, not to make the
    absence of a catalogue into the user's problem.

    The order is the provider's, not ours. It is their curation, and sorting it
    alphabetically would bury the model they put first.
    """
    from .._config import get_setting

    key = api_key(config)
    if not key:
        return []
    url = get_setting(config, "chat.base_url").rstrip("/") + "/models"
    try:
        async with httpx.AsyncClient(timeout=_LIST_TIMEOUT) as client:
            reply = await client.get(url, headers=_headers(config, key))
        if reply.status_code >= 400:
            logger.debug("%s answered %s for the model list", url, reply.status_code)
            return []
        data = reply.json().get("data") or []
    except Exception as exc:  # noqa: BLE001 - no list is a usable answer
        logger.debug("could not read the model list from %s: %s", url, exc)
        return []
    return [
        {"value": m["id"], "name": m["id"]}
        for m in data
        if isinstance(m, dict) and m.get("id")
    ]


def _route(config):
    """The URL to POST a turn to, per ``chat.api``."""
    from .._config import get_setting

    base = get_setting(config, "chat.base_url").rstrip("/")
    if get_setting(config, "chat.api") == "responses":
        return base + "/responses"
    return base + "/chat/completions"


def _error_detail(reply, url):
    """The provider's own words -- unless they are a web page.

    A body of HTML means the POST did not reach an API at all: the usual cause
    is a ``base_url`` that already carries the route, so the appended one lands
    on the gateway's website and its 404 page comes back. Quoting 500 bytes of
    markup names neither the problem nor the fix, so that one case is answered
    instead of echoed.
    """
    body = reply.text or ""
    ctype = reply.headers.get("content-type", "")
    head = body.lstrip()[:64].lower()
    if "html" in ctype.lower() or head.startswith(("<!doctype", "<html")):
        return (
            f"{url} answered with a web page, not JSON. Check chat.base_url: it "
            "is the API root, and biopb appends the route itself."
        )
    return body[:500]


def _responses_part(part):
    """One chat-completions content part in the Responses spelling."""
    if part.get("type") == "image_url":
        # A string here, where chat-completions nests it in an object.
        url = (part.get("image_url") or {}).get("url") or ""
        return {"type": "input_image", "image_url": url}
    return {"type": "input_text", "text": part.get("text") or ""}


def _responses_input(messages):
    """Chat-completions messages as Responses input items.

    Three of the four shapes are not messages at all over there: a tool result
    is a ``function_call_output`` item keyed by ``call_id``, an assistant turn
    that called tools becomes one item per call beside its text, and image parts
    change spelling. Everything else passes through as a role + content message.
    """
    items = []
    for msg in messages:
        role = msg.get("role")
        content = msg.get("content")
        if role == "tool":
            items.append(
                {
                    "type": "function_call_output",
                    "call_id": msg.get("tool_call_id") or "",
                    "output": content or "",
                }
            )
        elif role == "assistant" and msg.get("tool_calls"):
            if content:
                items.append({"role": "assistant", "content": content})
            for call in msg["tool_calls"]:
                fn = call.get("function") or {}
                items.append(
                    {
                        "type": "function_call",
                        "call_id": call.get("id") or "",
                        "name": fn.get("name") or "",
                        "arguments": fn.get("arguments") or "{}",
                    }
                )
        elif isinstance(content, list):
            parts = [_responses_part(p) for p in content]
            items.append({"role": role, "content": parts})
        else:
            items.append({"role": role, "content": content or ""})
    return items


def _responses_tools(tools):
    """Tool declarations flattened out of their ``function`` envelope."""
    out = []
    for tool in tools or []:
        fn = tool.get("function") if isinstance(tool, dict) else None
        if not fn:
            out.append(tool)
            continue
        out.append(
            {
                "type": "function",
                "name": fn.get("name") or "",
                "description": fn.get("description") or "",
                "parameters": (
                    fn.get("parameters") or {"type": "object", "properties": {}}
                ),
                # Stated rather than defaulted, because the default is the
                # gateway's to choose and strict mode rejects the schemas
                # biopb's tools publish (open objects, optional properties).
                "strict": False,
            }
        )
    return out


def _responses_message(data, model_name):
    """A Responses reply as the assistant message the loop expects.

    Reasoning items are dropped: their ``encrypted_content`` is only useful
    handed back to the same provider on the next call, and the loop's thread is
    re-sent whole every turn, so keeping them would cost tokens for state
    nothing here reads.
    """
    status = data.get("status")
    if status and status != "completed":
        detail = (data.get("error") or {}).get("message") or data.get(
            "incomplete_details"
        )
        raise RuntimeError(f"{model_name} returned status {status}: {detail}")
    output = data.get("output") or []
    if not output:
        raise RuntimeError(f"{model_name} returned no output")
    text, calls = [], []
    for item in output:
        if item.get("type") == "message":
            for part in item.get("content") or []:
                if part.get("type") == "output_text" and part.get("text"):
                    text.append(part["text"])
        elif item.get("type") == "function_call":
            calls.append(
                {
                    "id": item.get("call_id") or item.get("id") or "",
                    "type": "function",
                    "function": {
                        "name": item.get("name") or "",
                        "arguments": item.get("arguments") or "{}",
                    },
                }
            )
    message = {"role": "assistant", "content": "".join(text)}
    if calls:
        message["tool_calls"] = calls
    return message


def make_model(config):
    """Build the async ``(messages, tools) -> assistant message`` the loop takes.

    The key is read per call rather than captured, so replacing the credential
    file takes effect on the next turn instead of at the next restart — the file
    is how a user *sets* their key, and a session that has to be restarted to
    notice is a support question waiting to happen.
    """

    from .._config import get_setting

    async def model(messages, tools):
        key = api_key(config)
        if not key:
            raise ChatNotConfigured("No provider key.")
        model_name = get_setting(config, "chat.model")
        responses = get_setting(config, "chat.api") == "responses"
        payload = {
            "model": model_name,
            # The loop decides when it is finished by whether tool_calls come
            # back, so the model must stay free to answer instead of calling.
            "tool_choice": "auto",
        }
        if responses:
            payload["input"] = _responses_input(messages)
            payload["tools"] = _responses_tools(tools)
            # The thread is re-sent whole on every call, so server-side
            # retention buys the loop nothing and would leave a copy of the
            # user's conversation with the provider by default.
            payload["store"] = False
        else:
            payload["messages"] = messages
            payload["tools"] = tools
        url = _route(config)
        timeout = get_setting(config, "chat.request_timeout")
        async with httpx.AsyncClient(timeout=timeout) as client:
            reply = await client.post(url, json=payload, headers=_headers(config, key))
        if reply.status_code >= 400:
            # The provider's own words, truncated: a 400 here is usually a
            # payload the model rejected (a schema it dislikes, a context
            # overflow), and the detail is the only thing that identifies which.
            detail = _error_detail(reply, url)
            if _carries_image(messages) and any(
                word in detail.lower() for word in _VISION_REFUSALS
            ):
                # The one rejection the loop can do something about, so it is
                # told apart from the ones it cannot: a model with no vision
                # fails every turn after the screenshot, not just that one.
                raise VisionUnsupported(
                    f"{model_name} rejected an image ({reply.status_code}): {detail}"
                )
            # A 4xx is a verdict on the payload, so the payload's shape is
            # what identifies which turn drew it -- the provider names the
            # field it wanted, never the message that lacked it
            # (biopb/biopb#990). A 5xx is the provider's own fault and the
            # thread had nothing to do with it.
            shape = (
                f"\n{describe_messages(messages)}" if reply.status_code < 500 else ""
            )
            raise RuntimeError(
                f"{model_name} returned {reply.status_code}: {detail}{shape}"
            )
        if responses:
            return _responses_message(reply.json(), model_name)
        choices = reply.json().get("choices") or []
        if not choices:
            raise RuntimeError(f"{model_name} returned no choices")
        return choices[0].get("message") or {}

    return model
