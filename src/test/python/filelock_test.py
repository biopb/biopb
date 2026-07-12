"""Unit tests for the cross-process file lock and the atomic PID-file write.

Both harden ``biopb control start`` for the world where several processes (the
launcher, the installer, and -- once the shim starts the control on demand --
racing agent sessions) can invoke it at once. See
``biopb-mcp/docs/mcp-dedaemonization-migration.md`` and ``biopb._filelock``.
"""

import time

import pytest
from biopb._filelock import LockTimeout, file_lock


class TestFileLock:
    def test_lock_timeout_subclasses_timeouterror(self):
        # Callers may catch either; the distinct type keeps a lock timeout from
        # being confused with an unrelated TimeoutError inside the guarded block.
        assert issubclass(LockTimeout, TimeoutError)

    def test_second_acquire_blocks_until_timeout(self, tmp_path):
        lock = tmp_path / "x.lock"
        # A second os.open of the same path is a separate open-file-description
        # (POSIX flock) / a separate handle (Windows), so it is excluded even from
        # the same process -- the exclusion we rely on across processes.
        with file_lock(lock, timeout=5.0):
            t0 = time.monotonic()
            with pytest.raises(LockTimeout):
                with file_lock(lock, timeout=0.5, poll=0.05):
                    pass  # pragma: no cover - must not be reached
            # It actually waited out (roughly) the whole timeout, not failed fast.
            assert time.monotonic() - t0 >= 0.4

    def test_lock_reacquirable_after_release(self, tmp_path):
        lock = tmp_path / "x.lock"
        with file_lock(lock, timeout=1.0):
            pass
        # Released on block exit -> immediately re-acquirable.
        with file_lock(lock, timeout=1.0):
            pass

    def test_lock_released_on_exception(self, tmp_path):
        lock = tmp_path / "x.lock"
        with pytest.raises(RuntimeError):
            with file_lock(lock, timeout=1.0):
                raise RuntimeError("boom")
        # The lock must not leak when the guarded block raises.
        with file_lock(lock, timeout=1.0):
            pass

    def test_parent_dir_and_file_created(self, tmp_path):
        lock = tmp_path / "nested" / "dir" / "x.lock"
        with file_lock(lock, timeout=1.0):
            assert lock.exists()
        # The lock file persists after release (its existence conveys nothing;
        # the lock lives on the fd, not the file).
        assert lock.exists()


class TestAtomicPidWrite:
    """`biopb.cli._write_pid_file` publishes atomically and leaves no temp file."""

    def test_roundtrip_pid_and_token(self, tmp_path):
        from biopb import cli

        pid_file = tmp_path / "control.pid"
        cli._write_pid_file(pid_file, 4321, 99887766)
        pid, token = cli._read_pid_record(pid_file)
        assert pid == 4321
        assert token == 99887766

    def test_roundtrip_bare_pid_when_no_token(self, tmp_path):
        from biopb import cli

        pid_file = tmp_path / "control.pid"
        cli._write_pid_file(pid_file, 4321, None)
        pid, token = cli._read_pid_record(pid_file)
        assert pid == 4321
        assert token is None  # legacy bare-PID form -> liveness-only identity

    def test_no_temp_file_left_behind(self, tmp_path):
        from biopb import cli

        pid_file = tmp_path / "control.pid"
        cli._write_pid_file(pid_file, 4321, 5)
        # The sibling temp (".<name>-*.tmp") is renamed into place, not orphaned.
        leftovers = [p.name for p in tmp_path.iterdir() if p.name != pid_file.name]
        assert leftovers == []

    def test_overwrite_replaces_prior_record(self, tmp_path):
        from biopb import cli

        pid_file = tmp_path / "control.pid"
        cli._write_pid_file(pid_file, 1, 1)
        cli._write_pid_file(pid_file, 2, 2)
        assert cli._read_pid_record(pid_file) == (2, 2)
