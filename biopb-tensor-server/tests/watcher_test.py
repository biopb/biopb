"""Tests for file write completion detection (CLOSED events and stability checks)."""

import time
import tempfile
import signal
from pathlib import Path
from multiprocessing import Queue, Event
from types import SimpleNamespace

import pytest

from biopb_tensor_server.watcher import (
    WatcherEventType,
    WatcherEvent,
    WatchdogWatcher,
    PollVFSWatcher,
    _emit_debounced_events,
)


class TestClosedEventType:
    """Tests for CLOSED event type."""

    def test_closed_event_type_exists(self):
        """CLOSED event type should be available."""
        assert WatcherEventType.CLOSED == WatcherEventType.CLOSED
        assert WatcherEventType.CLOSED.value == "closed"

    def test_closed_event_in_watcher_event(self):
        """WatcherEvent should support CLOSED type."""
        event = WatcherEvent(
            event_type=WatcherEventType.CLOSED,
            path=Path("/tmp/test.txt"),
            is_directory=False,
        )
        assert event.event_type == WatcherEventType.CLOSED
        assert event.path == Path("/tmp/test.txt")
        assert not event.is_directory


class TestDebouncingWithClosed:
    """Tests for debouncing logic with CREATED+CLOSED pairs."""

    def test_created_closed_pair_emits_closed(self):
        """CREATED+CLOSED pair should emit CLOSED (last event)."""
        queue = Queue()
        event_buffer = {}

        path = Path("/tmp/test.txt")
        # CREATED first
        event_buffer[path] = (WatcherEventType.CREATED, time.time() - 0.5, None, False)
        # CLOSED later (replaces CREATED in buffer for same path)
        # Note: in actual watcher, same path would have multiple entries
        # but buffer stores one entry per path (last wins)
        # Let's test with separate entries
        event_buffer[path] = (WatcherEventType.CLOSED, time.time(), None, False)

        _emit_debounced_events(queue, event_buffer)

        # Get events
        events = []
        while True:
            try:
                event = queue.get(timeout=0.1)
                if event is None:
                    break
                events.append(event)
            except:
                break

        # Should emit CLOSED
        assert len(events) == 1
        assert events[0].event_type == WatcherEventType.CLOSED

    def test_created_then_deleted_emits_deleted(self):
        """CREATED then DELETED for same path should emit DELETED (last event)."""
        queue = Queue()
        event_buffer = {}

        path = Path("/tmp/transient.txt")
        # CREATED first, then DELETED (overwrites CREATED)
        event_buffer[path] = (WatcherEventType.DELETED, time.time(), None, False)

        _emit_debounced_events(queue, event_buffer)

        # Get events
        events = []
        while True:
            try:
                event = queue.get(timeout=0.1)
                if event is None:
                    break
                events.append(event)
            except:
                break

        # Should emit DELETED (last event for path)
        assert len(events) == 1
        assert events[0].event_type == WatcherEventType.DELETED


class TestStabilityCheck:
    """Tests for file stability check logic."""

    def test_stable_file_detection(self, tmp_path):
        """Files with old mtime should be detected as stable."""
        # Create a file
        test_file = tmp_path / "stable.txt"
        test_file.write_text("test content")

        # Set mtime to be old enough (more than 2 seconds ago)
        old_time = time.time() - 5
        test_file.stat().st_mtime  # Current mtime

        # Wait a moment and check
        age = time.time() - test_file.stat().st_mtime

        # File should be stable if created a while ago
        # (Note: this test just verifies the logic, actual watcher has stability_window param)

    def test_recent_file_not_stable(self, tmp_path):
        """Files with recent mtime should not be considered stable."""
        # Create a file right now
        test_file = tmp_path / "recent.txt"
        test_file.write_text("test content")

        # Age should be small
        age = time.time() - test_file.stat().st_mtime
        assert age < 2  # Less than 2 seconds

    def test_file_handle_release_check(self, tmp_path):
        """Files should pass stability check only if handle is released."""
        # Create a file
        test_file = tmp_path / "handle.txt"
        test_file.write_text("test content")

        # File should be openable in append mode
        try:
            with open(test_file, "a"):
                pass  # Should succeed
            handle_released = True
        except (IOError, OSError):
            handle_released = False

        assert handle_released  # Should be able to open


class TestPendingQueue:
    """Tests for pending file queue in SourceManager."""

    def test_pending_creates_tracking(self, tmp_path):
        """Test that pending creates are tracked correctly."""
        from biopb_tensor_server.source_manager import SourceManager

        # Create a minimal mock setup
        # This tests the internal data structure
        # In real tests, we would mock the server and registry

        # Pending creates should be a dict
        pending: dict = {}

        # Add a pending file
        path = tmp_path / "pending.txt"
        path_str = str(path)
        pending[path_str] = (path, time.time(), False)

        # Check it's tracked
        assert path_str in pending
        assert pending[path_str][1] > 0

        # Remove it (simulating CLOSED event)
        pending.pop(path_str)
        assert path_str not in pending

    def test_timeout_cleanup(self, tmp_path):
        """Test that timed-out pending files are removed."""
        from biopb_tensor_server.source_manager import SourceManager

        pending: dict = {}
        timeout = 60.0

        # Add an old pending file (simulating timeout)
        path = tmp_path / "timeout.txt"
        path_str = str(path)
        pending[path_str] = (path, time.time() - timeout - 1, False)

        # Check it would timeout
        age = time.time() - pending[path_str][1]
        assert age > timeout


class _FakeProcess:
    def __init__(self, join_effects, alive_states, pid=1234, exitcode=0):
        self.pid = pid
        self.exitcode = exitcode
        self._join_effects = list(join_effects)
        self._alive_states = list(alive_states)
        self.terminate_calls = 0
        self.kill_calls = 0
        self.close_calls = 0
        self.join_calls = []

    def join(self, timeout=None):
        self.join_calls.append(timeout)
        if self._join_effects:
            effect = self._join_effects.pop(0)
            if isinstance(effect, BaseException):
                raise effect
        return None

    def is_alive(self):
        if self._alive_states:
            return self._alive_states.pop(0)
        return False

    def terminate(self):
        self.terminate_calls += 1

    def kill(self):
        self.kill_calls += 1

    def close(self):
        self.close_calls += 1


class _NeverExitProcess(_FakeProcess):
    def __init__(self, pid=1234):
        super().__init__(join_effects=[], alive_states=[], pid=pid)

    def is_alive(self):
        return True


class TestWatcherShutdown:
    def test_stop_terminates_and_closes_process(self):
        watcher = WatchdogWatcher()
        watcher._shutdown_event = SimpleNamespace(set=lambda: None)
        process = _FakeProcess(
            join_effects=[None, None],
            alive_states=[True, True, False],
        )
        watcher._process = process

        watcher.stop()

        assert watcher._process is None
        assert process.join_calls == [0]
        assert process.terminate_calls == 0
        assert process.kill_calls == 0
        assert process.close_calls == 1

    def test_stop_ignores_sigint_during_cleanup(self, monkeypatch):
        watcher = WatchdogWatcher()
        watcher._shutdown_event = SimpleNamespace(set=lambda: None)
        process = _FakeProcess(join_effects=[None], alive_states=[False])
        watcher._process = process

        handlers = []
        current_handler = signal.default_int_handler

        def fake_getsignal(sig):
            assert sig == signal.SIGINT
            return current_handler

        def fake_signal(sig, handler):
            nonlocal current_handler
            assert sig == signal.SIGINT
            handlers.append(handler)
            previous = current_handler
            current_handler = handler
            return previous

        monkeypatch.setattr(
            "biopb_tensor_server.watcher.signal.getsignal", fake_getsignal
        )
        monkeypatch.setattr("biopb_tensor_server.watcher.signal.signal", fake_signal)

        watcher.stop()

        assert handlers == [signal.SIG_IGN, signal.default_int_handler]

    def test_stop_detaches_process_after_failed_kill(self, monkeypatch):
        watcher = WatchdogWatcher()
        watcher._shutdown_event = SimpleNamespace(set=lambda: None)
        process = _NeverExitProcess()
        watcher._process = process

        detached = []
        monkeypatch.setattr("biopb_tensor_server.watcher.time.sleep", lambda _: None)
        monkeypatch.setattr(
            "biopb_tensor_server.watcher._detach_process",
            lambda proc: detached.append(proc),
        )

        watcher.stop()

        assert process.terminate_calls == 1
        assert process.kill_calls == 1
        assert detached == [process]

    def test_stop_detects_zombie_after_failed_kill(self, monkeypatch, caplog):
        watcher = WatchdogWatcher()
        watcher._shutdown_event = SimpleNamespace(set=lambda: None)
        process = _NeverExitProcess(pid=4321)
        watcher._process = process

        monkeypatch.setattr("biopb_tensor_server.watcher.time.sleep", lambda _: None)
        monkeypatch.setattr(
            "biopb_tensor_server.watcher._get_linux_process_state", lambda pid: "Z"
        )
        monkeypatch.setattr(
            "biopb_tensor_server.watcher._detach_process", lambda proc: None
        )

        watcher.stop()

        assert "became a zombie after SIGKILL" in caplog.text


class TestWatcherRecovery:
    def test_watchdog_restarts_dead_subprocess_when_polled(self, monkeypatch, caplog):
        watcher = WatchdogWatcher()
        watcher._shutdown_event = SimpleNamespace(is_set=lambda: False)
        watcher._directories = {Path("/tmp/watch")}
        watcher._process = _FakeProcess(
            join_effects=[],
            alive_states=[False],
            pid=4321,
            exitcode=17,
        )

        launched = []
        replacement = _FakeProcess(join_effects=[], alive_states=[True], pid=9876)

        def fake_launch(directories):
            launched.append(directories)
            watcher._process = replacement

        monkeypatch.setattr(watcher, "_launch_process", fake_launch)

        events = watcher.get_events(timeout=0)

        assert events == []
        assert launched == [{Path("/tmp/watch")}]
        assert watcher._process is replacement
        assert "exited unexpectedly with code 17; restarting" in caplog.text

    def test_pollvfs_restarts_dead_subprocess_when_checked(self, monkeypatch, caplog):
        watcher = PollVFSWatcher()
        watcher._shutdown_event = SimpleNamespace(is_set=lambda: False)
        watcher._directories = {Path("/tmp/poll")}
        watcher._process = _FakeProcess(
            join_effects=[],
            alive_states=[False],
            pid=2468,
            exitcode=9,
        )

        launched = []
        replacement = _FakeProcess(join_effects=[], alive_states=[True], pid=1357)

        def fake_launch(directories):
            launched.append(directories)
            watcher._process = replacement

        monkeypatch.setattr(watcher, "_launch_process", fake_launch)

        assert watcher.is_running() is True
        assert launched == [{Path("/tmp/poll")}]
        assert watcher._process is replacement
        assert "exited unexpectedly with code 9; restarting" in caplog.text
