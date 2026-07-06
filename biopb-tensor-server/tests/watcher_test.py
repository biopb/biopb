"""Tests for periodic rescan watcher behavior."""

import pytest
from biopb_tensor_server.watcher import (
    PeriodicRescanWatcher,
    WatcherEventType,
    get_watcher,
)


def test_periodic_watcher_emits_rescan_after_interval(monkeypatch, tmp_path):
    current_time = {"value": 100.0}

    monkeypatch.setattr(
        "biopb_tensor_server.watcher.time.monotonic",
        lambda: current_time["value"],
    )
    monkeypatch.setattr("biopb_tensor_server.watcher.time.sleep", lambda _: None)

    # initial_immediate=False: testing the steady-state cadence (first tick after
    # one interval), not the progressive-discovery immediate-first-tick.
    watcher = PeriodicRescanWatcher(rescan_interval=1.0, initial_immediate=False)
    watcher.start({tmp_path})

    current_time["value"] = 100.5
    assert watcher.get_events(timeout=0) == []

    current_time["value"] = 101.1
    events = watcher.get_events(timeout=0)

    assert len(events) == 1
    assert events[0].event_type == WatcherEventType.RESCAN


def test_periodic_watcher_emits_immediately_by_default(monkeypatch, tmp_path):
    # Progressive discovery: the first rescan fires at once on start() (no need
    # to wait a full interval), so the background bootstrap scan begins promptly.
    current_time = {"value": 100.0}

    monkeypatch.setattr(
        "biopb_tensor_server.watcher.time.monotonic",
        lambda: current_time["value"],
    )
    monkeypatch.setattr("biopb_tensor_server.watcher.time.sleep", lambda _: None)

    watcher = PeriodicRescanWatcher(rescan_interval=30.0)  # default immediate
    watcher.start({tmp_path})

    # No clock advance: the first tick is due immediately.
    events = watcher.get_events(timeout=0)
    assert len(events) == 1
    assert events[0].event_type == WatcherEventType.RESCAN

    # ...and it reschedules for one interval out (no immediate re-fire).
    assert watcher.get_events(timeout=0) == []


def test_periodic_watcher_stops_cleanly(tmp_path):
    watcher = PeriodicRescanWatcher(rescan_interval=1.0)
    watcher.start({tmp_path})

    assert watcher.is_running() is True

    watcher.stop()

    assert watcher.is_running() is False
    assert watcher.get_events(timeout=0) == []


def test_get_watcher_returns_periodic_runtime(tmp_path):
    for watcher_type in ("auto", "watchdog", "pollvfs", "periodic", "off"):
        watcher = get_watcher(
            watcher_type=watcher_type,
            directories={tmp_path},
            poll_interval=5.0,
        )
        assert isinstance(watcher, PeriodicRescanWatcher)


def test_periodic_watcher_reschedules_after_emit(monkeypatch, tmp_path):
    current_time = {"value": 50.0}

    monkeypatch.setattr(
        "biopb_tensor_server.watcher.time.monotonic",
        lambda: current_time["value"],
    )
    monkeypatch.setattr("biopb_tensor_server.watcher.time.sleep", lambda _: None)

    watcher = PeriodicRescanWatcher(rescan_interval=2.0, initial_immediate=False)
    watcher.start({tmp_path})

    current_time["value"] = 52.1
    first = watcher.get_events(timeout=0)
    current_time["value"] = 53.0
    second = watcher.get_events(timeout=0)
    current_time["value"] = 54.2
    third = watcher.get_events(timeout=0)

    assert len(first) == 1
    assert second == []
    assert len(third) == 1


def test_get_watcher_rejects_unknown_type():
    with pytest.raises(ValueError, match="Unknown watcher type"):
        get_watcher(watcher_type="bogus")
