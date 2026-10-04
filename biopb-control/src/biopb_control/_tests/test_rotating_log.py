"""A supervised child's log stays bounded while the child runs."""

import io
import os
import sys

from biopb_control._rotating_log import RotatingLog, pump
from biopb_control._supervisor import ServiceProcess, ServiceSpec


def test_it_rotates_while_open(tmp_path):
    path = tmp_path / "a.log"
    log = RotatingLog(path, max_bytes=100, backup_count=3)
    for i in range(20):
        log.write(f"line {i:02d} ".ljust(40, "x").encode() + b"\n")
    log.close()

    names = sorted(p.name for p in tmp_path.iterdir())
    assert names == ["a.log", "a.log.1", "a.log.2", "a.log.3"]
    # Bounded: no file is more than a limit plus the write that crossed it.
    assert all(p.stat().st_size <= 100 + 41 for p in tmp_path.iterdir())
    assert b"line 19" in path.read_bytes()


def test_a_file_already_over_the_limit_is_rotated_on_open(tmp_path):
    path = tmp_path / "a.log"
    path.write_bytes(b"x" * 500)
    RotatingLog(path, max_bytes=100).close()

    assert (tmp_path / "a.log.1").read_bytes() == b"x" * 500
    assert path.read_bytes() == b""


def test_writing_after_close_is_dropped_not_raised(tmp_path):
    log = RotatingLog(tmp_path / "a.log")
    log.close()
    log.write(b"late")
    assert (tmp_path / "a.log").read_bytes() == b""


def test_a_failed_rotation_keeps_appending(tmp_path, monkeypatch):
    path = tmp_path / "a.log"
    log = RotatingLog(path, max_bytes=50)

    def boom(*a, **k):
        raise OSError("held open")

    monkeypatch.setattr("biopb._locations.rotate_log", boom)
    for _ in range(10):
        log.write(b"y" * 30)
    log.close()
    assert path.stat().st_size == 300


class _Child(ServiceProcess):
    def __init__(self, tmp_path, code):
        super().__init__()
        self._spec = ServiceSpec(
            argv=[sys.executable, "-c", code],
            env=os.environ.copy(),
            host="127.0.0.1",
            port=1,
            log_path=tmp_path / "child.log",
            label="child",
        )

    def _service_spec(self):
        return self._spec


def test_a_chatty_child_cannot_outgrow_the_limit_and_its_last_words_survive(tmp_path):
    # stdout, stderr and a warning, none through ``logging``: what a record-only
    # rotating handler would miss.
    code = (
        "import sys, warnings\n"
        "for i in range(3000):\n"
        "    print('out', i, 'x' * 60)\n"
        "    print('err', i, 'x' * 60, file=sys.stderr)\n"
        "warnings.warn('the last warning')\n"
        "print('goodbye')\n"
    )
    child = _Child(tmp_path, code)
    child._log_sink = RotatingLog(tmp_path / "child.log", max_bytes=64 * 1024)
    child._start_process()
    child._proc.wait(timeout=30)
    child._close_log()

    logs = sorted(tmp_path.glob("child.log*"))
    assert len(logs) > 1
    # A file passes the limit by at most the one chunk the pump read.
    assert all(p.stat().st_size <= 64 * 1024 + 65536 for p in logs)
    assert b"goodbye" in (tmp_path / "child.log").read_bytes()


def test_pump_copies_until_eof(tmp_path):
    log = RotatingLog(tmp_path / "a.log")
    stream = io.BufferedReader(io.BytesIO(b"abc"))
    pump(stream, log)
    log.close()
    assert (tmp_path / "a.log").read_bytes() == b"abc"
