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
"""

import logging

import httpx
from biopb._credentials import read_credential

logger = logging.getLogger(__name__)

#: Credential file name, a sibling of the data plane's ``tensor-server.token``.
KEY_NAME = "chat-provider.token"


class ChatNotConfigured(RuntimeError):
    """Chat is off, has no model, or has no key.

    One exception for all three because they are one situation from the user's
    side — "this is not set up yet" — and the message says which part is missing
    so the answer is actionable rather than a shrug.
    """


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
        payload = {
            "model": get_setting(config, "chat.model"),
            "messages": messages,
            "tools": tools,
            # The loop decides when it is finished by whether tool_calls come
            # back, so the model must stay free to answer instead of calling.
            "tool_choice": "auto",
        }
        url = get_setting(config, "chat.base_url").rstrip("/") + "/chat/completions"
        timeout = get_setting(config, "chat.request_timeout")
        async with httpx.AsyncClient(timeout=timeout) as client:
            reply = await client.post(
                url, json=payload, headers={"Authorization": f"Bearer {key}"}
            )
        if reply.status_code >= 400:
            # The provider's own words, truncated: a 400 here is usually a
            # payload the model rejected (a schema it dislikes, a context
            # overflow), and the detail is the only thing that identifies which.
            raise RuntimeError(
                f"{get_setting(config, 'chat.model')} returned "
                f"{reply.status_code}: {reply.text[:500]}"
            )
        choices = reply.json().get("choices") or []
        if not choices:
            raise RuntimeError(
                f"{get_setting(config, 'chat.model')} returned no choices"
            )
        return choices[0].get("message") or {}

    return model
