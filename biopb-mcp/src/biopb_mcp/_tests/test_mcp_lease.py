"""The session lease (``_lease``): who the session answers to."""

import pytest

from biopb_mcp.mcp import _lease, _writers


@pytest.fixture(autouse=True)
def fresh(monkeypatch):
    _lease._reset()
    _writers.clear_claim()
    clock = {"now": 1000.0}
    monkeypatch.setattr(_lease.time, "monotonic", lambda: clock["now"])
    yield clock
    _lease._reset()
    _writers.clear_claim()


def test_a_free_session_is_taken_and_reported():
    assert _lease.snapshot() == {"holder": None}
    got = _lease.acquire("agent", "a")
    assert got["ok"] and got["holder"] == "agent"
    assert _lease.snapshot()["holder"] == "agent"


def test_the_other_kind_is_refused_while_the_lease_lives():
    _lease.acquire("agent", "a")
    refused = _lease.acquire("chat", _lease.CHAT_TOKEN)
    assert refused["ok"] is False and refused["holder"] == "agent"
    assert _lease.snapshot()["holder"] == "agent"


def test_the_holder_asking_again_renews_without_a_change():
    changes = []
    _lease.on_change(lambda *c: changes.append(c))
    _lease.acquire("agent", "a")
    assert _lease.acquire("agent", "a")["ok"]
    assert changes == [(None, "agent")]


def test_force_replaces_the_holder_and_the_replaced_kind_hears_of_it():
    changes = []
    _lease.on_change(lambda *c: changes.append(c))
    _lease.acquire("chat", _lease.CHAT_TOKEN)
    assert _lease.acquire("agent", "a", force=True)["ok"]
    assert changes == [(None, "chat"), ("chat", "agent")]


def test_release_is_the_holders_alone():
    _lease.acquire("agent", "a")
    assert _lease.release("someone-else") is False
    assert _lease.snapshot()["holder"] == "agent"
    assert _lease.release("a") is True
    assert _lease.snapshot() == {"holder": None}


def test_a_lease_lapses_without_renewal(fresh):
    changes = []
    _lease.on_change(lambda *c: changes.append(c))
    _lease.acquire("agent", "a")
    fresh["now"] += _lease.TTL["agent"] - 1
    assert _lease.renew("a") is True
    fresh["now"] += _lease.TTL["agent"] - 1
    assert _lease.snapshot()["holder"] == "agent"  # the renewal counted
    fresh["now"] += 2
    assert _lease.snapshot() == {"holder": None}
    assert changes[-1] == ("agent", None)
    assert _lease.renew("a") is False


def test_chat_outlives_an_agent_beat_but_not_its_own_window(fresh):
    _lease.acquire("chat", _lease.CHAT_TOKEN)
    fresh["now"] += _lease.TTL["agent"] + 1
    assert _lease.snapshot()["holder"] == "chat"
    fresh["now"] += _lease.TTL["chat"]
    assert _lease.snapshot()["holder"] is None


def test_a_busy_holder_does_not_lapse(fresh):
    busy = {"on": True}
    _lease.set_busy_probe("chat", lambda: busy["on"])
    _lease.acquire("chat", _lease.CHAT_TOKEN)
    fresh["now"] += _lease.TTL["chat"] * 3
    assert _lease.snapshot()["holder"] == "chat"
    busy["on"] = False
    assert _lease.snapshot()["holder"] is None


def test_a_change_of_holder_clears_the_kernels_claim():
    # The kernel's own claim outlives whoever made it; the next holder must not
    # be measured against it.
    _writers._note_claim("biopb-chat", "chat")
    assert _writers.claim_holder() == "biopb-chat"
    _lease.acquire("agent", "a")
    assert _writers.claim_holder() is None


def test_an_unknown_kind_is_a_programming_error():
    with pytest.raises(ValueError):
        _lease.acquire("robot", "x")
