"""Tests for `biopb._control._endpoints` — the base-port convention and control discovery.

Two things live here, and they are load-bearing for anything that has to *find*
the control rather than start it:

- **the base-port convention.** One number places all three listeners
  (control base+3, sidecar base+4, flight base+5). The offsets are the
  container's (`entrypoint.sh`: BIOPB_BASE_PORT, sidecar +4, gRPC +5), extended
  with the control at +3 -- deliberately one scheme, not two that agree at their
  defaults and diverge the moment either base moves;

- **the runtime (discovery) record.** Once `--base-port` could move the control
  off 8813, a client that only knew 8813 would look in the wrong place. A
  serving control publishes the endpoint it bound; readers resolve
  `BIOPB_CONTROL_*` -> that record -> 8813. It is a *hint*, never proof of life:
  a crash leaves it behind, which is why it carries a pid and why every consumer
  probes.
"""

import json

import pytest
from biopb._control import _endpoints


@pytest.fixture(autouse=True)
def _isolated_state(tmp_path, monkeypatch):
    """Point the state tree at a tmpdir and clear the env overrides.

    ``BIOPB_STATE_HOME`` is what relocates the record -- `_locations.state_dir()`
    reads it, and CI sets it, so it must be *set* rather than merely trusted to
    be absent. Getting this wrong does not fail the test; it silently writes a
    bogus ``control.json`` into the developer's real state dir, where discovery
    then points every client at a port nothing is listening on.

    The two ``BIOPB_CONTROL_*`` vars sit *above* the record in precedence, so a
    stray ambient value would mask exactly what these tests assert.
    """
    monkeypatch.setenv("BIOPB_STATE_HOME", str(tmp_path))
    monkeypatch.delenv("BIOPB_CONTROL_PORT", raising=False)
    monkeypatch.delenv("BIOPB_CONTROL_HOST", raising=False)
    # Same reason, for how the user's browser reaches the control.
    monkeypatch.delenv("BIOPB_URL_PREFIX", raising=False)
    monkeypatch.delenv("BIOPB_PUBLIC_ORIGIN", raising=False)


class TestBasePortConvention:
    def test_default_base_reproduces_the_historical_ports(self):
        base = _endpoints.BASE_DEFAULT_PORT
        assert base == 8810
        assert _endpoints.control_port_for(base) == 8813
        assert _endpoints.sidecar_port_for(base) == 8814
        assert _endpoints.flight_port_for(base) == 8815

    def test_offsets_match_the_container(self):
        # entrypoint.sh: HTTP_PORT=BASE+4, GRPC_PORT=BASE+5. Drifting from these
        # would give biopb two base-port schemes indistinguishable in a bug report.
        assert _endpoints.SIDECAR_PORT_OFFSET == 4
        assert _endpoints.FLIGHT_PORT_OFFSET == 5

    def test_a_moved_base_moves_all_three_together(self):
        assert _endpoints.control_port_for(9000) == 9003
        assert _endpoints.sidecar_port_for(9000) == 9004
        assert _endpoints.flight_port_for(9000) == 9005

    def test_the_default_control_port_is_derived_not_hardcoded(self):
        assert (
            _endpoints.control_port_for(_endpoints.BASE_DEFAULT_PORT)
            == _endpoints.CONTROL_DEFAULT_PORT
        )


class TestRuntimeRecord:
    def test_absent_record_falls_back_to_the_default(self):
        assert _endpoints.read_runtime_record() == {}
        assert _endpoints.control_port() == _endpoints.CONTROL_DEFAULT_PORT
        assert _endpoints.control_host() == _endpoints.CONTROL_DEFAULT_HOST

    def test_published_endpoint_is_discovered(self):
        _endpoints.write_runtime_record("127.0.0.1", 9003, 4242)
        assert _endpoints.control_port() == 9003
        assert _endpoints.control_base_url() == "http://127.0.0.1:9003"

    def test_record_carries_the_serving_pid(self):
        # `biopb control status` uses it to tell a live foreground control (which
        # writes no pid file) from a record a crashed one left behind.
        _endpoints.write_runtime_record("127.0.0.1", 9003, 4242)
        assert _endpoints.read_runtime_record()["pid"] == 4242

    def test_record_carries_a_create_time_token(self):
        """The pid alone cannot survive pid reuse; the token is what identifies.

        Only a crash strands a record -- a clean stop retracts it -- so a stale
        record is exactly the case where the pid may since have been recycled.
        """
        import os

        from biopb.lifecycle.proc import process_create_time

        _endpoints.write_runtime_record("127.0.0.1", 9003, os.getpid())
        record = _endpoints.read_runtime_record()
        assert "create_time" in record
        # None only where the platform has no cheap create-time (readers then
        # degrade to liveness); elsewhere it must name *this* process.
        assert record["create_time"] == process_create_time(os.getpid())

    def test_remove_retracts_it(self):
        _endpoints.write_runtime_record("127.0.0.1", 9003, 4242)
        _endpoints.remove_runtime_record()
        assert _endpoints.read_runtime_record() == {}
        assert _endpoints.control_port() == _endpoints.CONTROL_DEFAULT_PORT

    def test_remove_is_idempotent(self):
        _endpoints.remove_runtime_record()  # no raise on an absent file

    def test_env_outranks_the_record(self, monkeypatch):
        # An explicit override must still win over a discovered value.
        _endpoints.write_runtime_record("127.0.0.1", 9003, 4242)
        monkeypatch.setenv("BIOPB_CONTROL_PORT", "7777")
        monkeypatch.setenv("BIOPB_CONTROL_HOST", "10.0.0.9")
        assert _endpoints.control_port() == 7777
        assert _endpoints.control_host() == "10.0.0.9"

    @pytest.mark.parametrize(
        "content", ["", "not json", "[]", '{"port": "not-an-int"}', '{"port": null}']
    )
    def test_malformed_record_degrades_to_the_default(self, content):
        """A corrupt record must never raise out of `control_port()`.

        Every client calls it before it can reach the control at all, so a hard
        failure here would break discovery entirely rather than degrade to 8813.
        """
        from biopb._locations import control_runtime_file

        path = control_runtime_file()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        assert _endpoints.control_port() == _endpoints.CONTROL_DEFAULT_PORT

    def test_undeterminable_home_degrades_to_the_default(self, monkeypatch):
        """Not even being able to *locate* the record must not raise.

        On Windows `Path.home()` raises RuntimeError when the environment has no
        USERPROFILE/HOMEPATH -- a scrubbed-env service, or any test that clears
        `os.environ`. POSIX falls back to the passwd database and never sees it,
        so this is simulated rather than provoked: it caught a real CI failure
        that no Linux run could reproduce.
        """
        import pathlib

        monkeypatch.delenv("BIOPB_STATE_HOME")

        def _no_home():
            raise RuntimeError("Could not determine home directory.")

        monkeypatch.setattr(pathlib.Path, "home", staticmethod(_no_home))
        assert _endpoints.read_runtime_record() == {}
        assert _endpoints.control_port() == _endpoints.CONTROL_DEFAULT_PORT
        assert _endpoints.control_host() == _endpoints.CONTROL_DEFAULT_HOST
        _endpoints.remove_runtime_record()  # no raise either

    def test_write_is_atomic(self):
        """Written via a temp file + replace, like the pid file.

        A client polling for the control must never observe a half-written record
        and parse it as "none".
        """
        _endpoints.write_runtime_record("127.0.0.1", 9003, 4242)
        from biopb._locations import control_runtime_file

        path = control_runtime_file()
        assert json.loads(path.read_text())["port"] == 9003
        # No temp file left behind.
        assert not list(path.parent.glob("*.tmp"))

    def test_each_writer_gets_its_own_temp(self, monkeypatch):
        """Two publishers must not share a scratch file.

        With a fixed temp name, both truncate and write it before either
        renames, and the record that lands can be a mix of the two. Only
        `control start` holds the start lock, so a foreground `control run` on
        the same state dir really can race it.
        """
        from biopb._locations import control_runtime_file

        path = control_runtime_file()
        path.parent.mkdir(parents=True, exist_ok=True)
        seen = set()
        real_mkstemp = _endpoints.tempfile.mkstemp

        def _spy(*args, **kwargs):
            fd, name = real_mkstemp(*args, **kwargs)
            seen.add(name)
            return fd, name

        monkeypatch.setattr(_endpoints.tempfile, "mkstemp", _spy)
        _endpoints.write_runtime_record("127.0.0.1", 9003, 4242)
        _endpoints.write_runtime_record("127.0.0.1", 9004, 4243)
        assert len(seen) == 2

    def test_failed_write_leaves_no_debris(self, monkeypatch):
        """A full disk must not strew temp files beside the record.

        The record is republished on every serve, so a leak here accumulates.
        """
        from biopb._locations import control_runtime_file

        path = control_runtime_file()
        path.parent.mkdir(parents=True, exist_ok=True)

        def _boom(*_a, **_k):
            raise OSError("No space left on device")

        monkeypatch.setattr(_endpoints.os, "replace", _boom)
        with pytest.raises(OSError):
            _endpoints.write_runtime_record("127.0.0.1", 9003, 4242)
        assert not list(path.parent.glob("*.tmp"))


class TestConnectHost:
    """A control bound to a wildcard is *dialed* over loopback."""

    @pytest.mark.parametrize(
        "bound, dialed",
        [
            ("0.0.0.0", "http://127.0.0.1:9003"),
            ("::", "http://[::1]:9003"),
            ("::1", "http://[::1]:9003"),
            ("10.1.2.3", "http://10.1.2.3:9003"),
        ],
    )
    def test_a_wildcard_bind_is_dialed_over_loopback(self, bound, dialed):
        _endpoints.write_runtime_record(bound, 9003, 4242)
        assert _endpoints.control_base_url() == dialed

    def test_the_bind_address_itself_is_still_what_the_record_says(self):
        """Only the connect URL is rewritten: control_host() is also what a
        control binds by default, where a wildcard must stay a wildcard."""
        _endpoints.write_runtime_record("0.0.0.0", 9003, 4242)
        assert _endpoints.control_host() == "0.0.0.0"


class TestUserBaseUrl:
    """How the *user's browser* reaches the control (biopb/biopb#1195)."""

    def test_a_plain_control_is_reached_where_this_machine_connects(self):
        _endpoints.write_runtime_record("127.0.0.1", 9003, 4242)
        assert _endpoints.user_base_url() == "http://127.0.0.1:9003"
        assert _endpoints.user_base_url() == _endpoints.control_base_url()

    def test_no_record_at_all_is_the_default(self):
        assert _endpoints.user_base_url() == _endpoints.control_base_url()

    def test_the_record_carries_neither_field_unless_set(self):
        _endpoints.write_runtime_record("127.0.0.1", 9003, 4242)
        record = _endpoints.read_runtime_record()
        assert "url_prefix" not in record and "public_origin" not in record

    def test_a_prefix_alone_is_a_path(self):
        _endpoints.write_runtime_record(
            "0.0.0.0", 9003, 4242, url_prefix="/node/h/9003"
        )
        assert _endpoints.public_url_prefix() == "/node/h/9003"
        assert _endpoints.public_origin() == ""
        assert _endpoints.user_base_url() == "/node/h/9003"
        # ...and the machine's own address is unaffected.
        assert _endpoints.control_base_url() == "http://127.0.0.1:9003"

    def test_prefix_and_origin_make_an_absolute_link(self):
        _endpoints.write_runtime_record(
            "0.0.0.0",
            9003,
            4242,
            url_prefix="/node/h/9003",
            public_origin="https://portal.example.edu",
        )
        assert _endpoints.user_base_url() == "https://portal.example.edu/node/h/9003"

    def test_an_origin_without_a_prefix_is_a_root_published_control(self):
        _endpoints.write_runtime_record(
            "127.0.0.1", 9003, 4242, public_origin="https://viewer.example.edu"
        )
        assert _endpoints.user_base_url() == "https://viewer.example.edu"

    def test_env_outranks_the_record(self, monkeypatch):
        _endpoints.write_runtime_record(
            "127.0.0.1",
            9003,
            4242,
            url_prefix="/from-record",
            public_origin="https://record.example.edu",
        )
        monkeypatch.setenv("BIOPB_URL_PREFIX", "/from-env")
        monkeypatch.setenv("BIOPB_PUBLIC_ORIGIN", "https://env.example.edu")
        assert _endpoints.user_base_url() == "https://env.example.edu/from-env"

    def test_slashes_are_tidied(self, monkeypatch):
        monkeypatch.setenv("BIOPB_URL_PREFIX", "node//h/9003/")
        monkeypatch.setenv("BIOPB_PUBLIC_ORIGIN", "https://p.example.edu/")
        assert _endpoints.user_base_url() == "https://p.example.edu/node/h/9003"

    def test_a_root_prefix_is_no_prefix(self, monkeypatch):
        monkeypatch.setenv("BIOPB_URL_PREFIX", "/")
        assert _endpoints.public_url_prefix() == ""
        assert _endpoints.user_base_url() == _endpoints.control_base_url()

    @pytest.mark.parametrize("junk", [None, 7, ["a"], {"x": 1}])
    def test_a_malformed_record_field_is_ignored(self, junk):
        from biopb._locations import control_runtime_file

        _endpoints.write_runtime_record("127.0.0.1", 9003, 4242)
        path = control_runtime_file()
        record = json.loads(path.read_text())
        record["url_prefix"] = junk
        record["public_origin"] = junk
        path.write_text(json.dumps(record))
        assert _endpoints.user_base_url() == _endpoints.control_base_url()
