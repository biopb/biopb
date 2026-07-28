"""Unit tests for biopb.tensor.cli diagnostic commands.

Uses typer.testing.CliRunner with mocked TensorFlightClient to avoid
requiring a live server.
"""

import json
import os
import socket
import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from biopb.tensor.cli import _parse_slice_hint, app
from biopb.tensor.descriptor_pb2 import DataSourceDescriptor, TensorDescriptor
from typer.testing import CliRunner

runner = CliRunner()


@pytest.fixture(autouse=True)
def _no_ambient_deployment(monkeypatch, tmp_path):
    """Resolve endpoints against nothing, not against the developer's own box.

    Since biopb/biopb#615 an omitted ``--server`` is *resolved* (env -> the
    control plane -> the default) rather than defaulted to a constant. Without
    this fixture a machine with a control plane running would feed these tests a
    live endpoint, and the state dir would hand them a real credential file.
    """
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setenv("HOME", str(tmp_path))
    for var in ("BIOPB_TENSOR_URL", "BIOPB_TENSOR_TOKEN"):
        monkeypatch.delenv(var, raising=False)
    with socket.socket() as sock:  # a port nothing is listening on
        sock.bind(("127.0.0.1", 0))
        free_port = sock.getsockname()[1]
    monkeypatch.setenv("BIOPB_CONTROL_HOST", "127.0.0.1")
    monkeypatch.setenv("BIOPB_CONTROL_PORT", str(free_port))


def _build_mock_client() -> MagicMock:
    """Build a mock TensorFlightClient for testing."""
    mock_client = MagicMock()

    # Create mock source and tensor descriptors
    tensor_desc_1 = TensorDescriptor(
        array_id="pos_0",
        shape=[512, 512],
        dtype="uint8",
    )
    tensor_desc_2 = TensorDescriptor(
        array_id="pos_1",
        shape=[512, 512],
        dtype="uint16",
    )

    source_desc = DataSourceDescriptor(
        source_id="my-source",
        tensors=[tensor_desc_1, tensor_desc_2],
    )

    # Mock list_sources
    mock_client.list_sources.return_value = {
        "my-source": source_desc,
    }

    # Mock get_source_metadata
    mock_client.get_source_metadata.return_value = {
        "multiscales": [
            {"name": "0", "datasets": [{"path": "0"}]},
        ],
        "axes": [
            {"name": "y", "type": "space"},
            {"name": "x", "type": "space"},
        ],
    }

    # Mock cache_info
    mock_client.cache_info.return_value = {
        "size_bytes": 1_000_000,
        "max_bytes": 100_000_000,
        "item_count": 42,
        "hits": 100,
        "misses": 50,
    }

    # Mock get_tensor to return a dask-like mock
    mock_arr = MagicMock()
    mock_arr.shape = (512, 512)
    mock_arr.dtype = "uint8"
    mock_arr.size = 512 * 512
    mock_arr.min.return_value.compute.return_value = 10
    mock_arr.max.return_value.compute.return_value = 200
    mock_arr.mean.return_value.compute.return_value = 100.5
    mock_client.get_tensor.return_value = mock_arr

    return mock_client


class TestQueryCommand:
    """Tests for the 'query' command."""

    def test_query_lists_sources(self):
        """Test that query lists available sources."""
        with patch("biopb.tensor.cli.TensorFlightClient") as mock_fc_class:
            mock_client = _build_mock_client()
            mock_fc_class.return_value = mock_client

            result = runner.invoke(app, ["query"])

            assert result.exit_code == 0
            assert "my-source" in result.stdout
            assert "pos_0" in result.stdout
            assert "pos_1" in result.stdout
            assert "[512, 512]" in result.stdout
            mock_client.close.assert_called_once()

    def test_query_with_custom_server(self):
        """Test that query respects --server option."""
        with patch("biopb.tensor.cli.TensorFlightClient") as mock_fc_class:
            mock_client = _build_mock_client()
            mock_fc_class.return_value = mock_client

            result = runner.invoke(app, ["query", "--server", "grpc://custom:9000"])

            assert result.exit_code == 0
            mock_fc_class.assert_called_once_with(
                location="grpc://custom:9000",
                cache_bytes=100_000_000,
                token=None,
                tls_ca_pem=None,
            )

    def test_query_shows_cache_info(self):
        """Test that query displays cache statistics."""
        with patch("biopb.tensor.cli.TensorFlightClient") as mock_fc_class:
            mock_client = _build_mock_client()
            mock_fc_class.return_value = mock_client

            result = runner.invoke(app, ["query"])

            assert result.exit_code == 0
            assert "hits=100" in result.stdout
            assert "misses=50" in result.stdout

    def test_query_handles_empty_sources(self):
        """Test that query handles empty source list gracefully."""
        with patch("biopb.tensor.cli.TensorFlightClient") as mock_fc_class:
            mock_client = _build_mock_client()
            mock_client.list_sources.return_value = {}
            mock_fc_class.return_value = mock_client

            result = runner.invoke(app, ["query"])

            assert result.exit_code == 0
            assert "No sources found" in result.stderr

    def test_query_connection_error(self):
        """Test that query handles connection errors."""
        with patch("biopb.tensor.cli.TensorFlightClient") as mock_fc_class:
            mock_fc_class.side_effect = ConnectionError("Connection refused")

            result = runner.invoke(app, ["query"])

            assert result.exit_code == 1
            # Classified by type and named with the endpoint's origin, so the
            # sentence says which address failed and where it came from.
            assert "grpc://127.0.0.1:8815" in result.stderr
            assert "no control plane answered" in result.stderr


class TestMetadataCommand:
    """Tests for the 'metadata' command."""

    def test_metadata_lists_tensors(self):
        """Test that metadata lists tensors in a source."""
        with patch("biopb.tensor.cli.TensorFlightClient") as mock_fc_class:
            mock_client = _build_mock_client()
            mock_fc_class.return_value = mock_client

            result = runner.invoke(app, ["metadata", "my-source"])

            assert result.exit_code == 0
            assert "pos_0" in result.stdout
            assert "pos_1" in result.stdout
            mock_client.close.assert_called_once()

    def test_metadata_shows_ome_metadata(self):
        """Test that metadata displays source-level OME metadata."""
        with patch("biopb.tensor.cli.TensorFlightClient") as mock_fc_class:
            mock_client = _build_mock_client()
            mock_fc_class.return_value = mock_client

            result = runner.invoke(app, ["metadata", "my-source"])

            assert result.exit_code == 0
            assert "multiscales" in result.stdout
            assert "axes" in result.stdout

    def test_metadata_with_specific_tensor(self):
        """Test that --tensor option shows detailed descriptor."""
        with patch("biopb.tensor.cli.TensorFlightClient") as mock_fc_class:
            mock_client = _build_mock_client()
            mock_fc_class.return_value = mock_client

            result = runner.invoke(app, ["metadata", "my-source", "--tensor", "pos_0"])

            assert result.exit_code == 0
            assert "Tensor Descriptor: pos_0" in result.stdout
            assert "uint8" in result.stdout

    def test_metadata_source_not_found(self):
        """Test that metadata handles missing source."""
        with patch("biopb.tensor.cli.TensorFlightClient") as mock_fc_class:
            mock_client = _build_mock_client()
            mock_fc_class.return_value = mock_client

            result = runner.invoke(app, ["metadata", "nonexistent"])

            assert result.exit_code == 1
            assert "Source not found" in result.stderr

    def test_metadata_tensor_not_found(self):
        """Test that --tensor handles missing tensor."""
        with patch("biopb.tensor.cli.TensorFlightClient") as mock_fc_class:
            mock_client = _build_mock_client()
            mock_fc_class.return_value = mock_client

            result = runner.invoke(
                app, ["metadata", "my-source", "--tensor", "nonexistent"]
            )

            assert result.exit_code == 1
            assert "Tensor not found" in result.stderr


class TestStatsCommand:
    """Tests for the 'stats' command."""

    def test_stats_computes_values(self):
        """Test that stats computes min, max, mean."""
        with patch("biopb.tensor.cli.TensorFlightClient") as mock_fc_class:
            with patch("biopb.tensor.cli.dask.compute") as mock_compute:
                mock_client = _build_mock_client()
                mock_fc_class.return_value = mock_client
                mock_compute.return_value = (10, 200, 100.5)

                result = runner.invoke(app, ["stats", "my-source/pos_0"])

                assert result.exit_code == 0
                assert "min" in result.stdout
                assert "max" in result.stdout
                assert "mean" in result.stdout
                assert "10" in result.stdout
                assert "200" in result.stdout
                assert "100.5" in result.stdout
                mock_client.close.assert_called_once()

    def test_stats_with_slice(self):
        """Test that stats respects --slice option."""
        with patch("biopb.tensor.cli.TensorFlightClient") as mock_fc_class:
            mock_client = _build_mock_client()
            mock_fc_class.return_value = mock_client

            result = runner.invoke(
                app,
                ["stats", "my-source/pos_0", "--slice", "0:100,0:100"],
            )

            assert result.exit_code == 0
            # Verify get_tensor was called with the slice
            call_args = mock_client.get_tensor.call_args
            assert call_args[1]["slice_hint"] == (slice(0, 100), slice(0, 100))

    def test_stats_missing_tensor(self):
        """Test that stats handles missing tensor."""
        with patch("biopb.tensor.cli.TensorFlightClient") as mock_fc_class:
            mock_client = _build_mock_client()
            mock_client.get_tensor.side_effect = ValueError("Tensor not found")
            mock_fc_class.return_value = mock_client

            result = runner.invoke(app, ["stats", "my-source/nonexistent"])

            assert result.exit_code == 1
            assert "Failed to compute statistics" in result.stderr

    def test_stats_displays_shape_and_dtype(self):
        """Test that stats shows tensor shape and dtype."""
        with patch("biopb.tensor.cli.TensorFlightClient") as mock_fc_class:
            mock_client = _build_mock_client()
            mock_fc_class.return_value = mock_client

            result = runner.invoke(app, ["stats", "my-source/pos_0"])

            assert result.exit_code == 0
            assert "[512, 512]" in result.stdout
            assert "uint8" in result.stdout


class TestParseSliceHint:
    """Tests for _parse_slice_hint helper function."""

    def test_parse_valid_slice(self):
        """Test parsing valid slice specifications."""
        result = _parse_slice_hint("0:100,50:150")
        assert result == (slice(0, 100), slice(50, 150))

    def test_parse_slice_with_missing_start(self):
        """Test parsing slice with missing start."""
        result = _parse_slice_hint(":100,50:")
        assert result == (slice(None, 100), slice(50, None))

    def test_parse_empty_slice(self):
        """Test that empty string returns None."""
        result = _parse_slice_hint("")
        assert result is None

    def test_parse_none_slice(self):
        """Test that None returns None."""
        result = _parse_slice_hint(None)
        assert result is None

    def test_parse_slice_with_spaces(self):
        """Test that spaces are handled."""
        result = _parse_slice_hint("  0:100 , 50:150  ")
        assert result == (slice(0, 100), slice(50, 150))

    def test_parse_invalid_format(self):
        """Test that invalid format raises BadParameter."""
        from typer import BadParameter

        with pytest.raises(BadParameter):
            _parse_slice_hint("0-100")

    def test_parse_single_dimension(self):
        """Test parsing single-dimension slice."""
        result = _parse_slice_hint("10:20")
        assert result == (slice(10, 20),)


class TestArrayIdFirstAddressing:
    """The CLI must address tensors by array_id alone (identity policy), not the
    deprecated ``(source_id, tensor_id)`` pair. See biopb/biopb#75."""

    def test_get_passes_array_id_as_single_argument(self):
        """`get` forwards the raw array_id positionally to get_tensor."""
        with patch("biopb.tensor.cli.TensorFlightClient") as mock_fc_class:
            mock_client = _build_mock_client()
            # The pickle output path serializes the returned array, so hand back
            # a real (picklable) array instead of the MagicMock default.
            mock_client.get_tensor.return_value = np.zeros((4, 4), dtype="uint8")
            mock_fc_class.return_value = mock_client

            with tempfile.TemporaryDirectory() as tmp:
                out = os.path.join(tmp, "out.pkl")
                result = runner.invoke(app, ["get", "my-source/pos_0", "-o", out])

            assert result.exit_code == 0, result.stderr
            call_args = mock_client.get_tensor.call_args
            # array_id passed as the single first positional argument; no
            # deprecated source_id=/tensor_id= keywords.
            assert call_args.args[0] == "my-source/pos_0"
            assert "source_id" not in call_args.kwargs
            assert "tensor_id" not in call_args.kwargs

    def test_stats_passes_array_id_as_single_argument(self):
        """`stats` forwards the raw array_id positionally to get_tensor."""
        with patch("biopb.tensor.cli.TensorFlightClient") as mock_fc_class:
            mock_client = _build_mock_client()
            mock_fc_class.return_value = mock_client

            result = runner.invoke(app, ["stats", "my-source/pos_0"])

            assert result.exit_code == 0, result.stderr
            call_args = mock_client.get_tensor.call_args
            assert call_args.args[0] == "my-source/pos_0"
            assert "source_id" not in call_args.kwargs
            assert "tensor_id" not in call_args.kwargs


class TestCliIntegration:
    """Integration-level CLI tests."""

    def test_help_messages(self):
        """Test that all commands have helpful help text."""
        result = runner.invoke(app, ["query", "--help"])
        assert result.exit_code == 0
        assert "Data sources" in result.stdout or "sources" in result.stdout

        result = runner.invoke(app, ["metadata", "--help"])
        assert result.exit_code == 0

        result = runner.invoke(app, ["get", "--help"])
        assert result.exit_code == 0

        result = runner.invoke(app, ["stats", "--help"])
        assert result.exit_code == 0

    def test_app_version(self):
        """Test that the app has a name and help text."""
        assert app.info.name == "tensor"
        assert app.info.help is not None


class TestCacheStatsCommand:
    """`biopb tensor cache-stats` — the server's cache, asked over Flight.

    It lived under `biopb server` until biopb/biopb#615, with an endpoint resolver
    of its own; it now dials through the same one as every other command here.
    """

    _STATS = {
        "hits": 80,
        "misses": 20,
        "evictions": 3,
        "pending_waits": 0,
        "oversized_skips": 0,
        "ref_held_evictions_skipped": 0,
        "total_entries": 12,
        "total_bytes": 5 * 1024 * 1024,
        "max_bytes": 512 * 1024 * 1024,
        "pool_stats": {
            "unified-tiny": {"hits": 50, "misses": 10, "segments": 2, "bytes": 1048576},
        },
    }

    def _run(self, *args, stats=None):
        with patch("biopb.tensor.cli.TensorFlightClient") as mock_fc_class:
            client = MagicMock()
            client.cache_stats.return_value = self._STATS if stats is None else stats
            mock_fc_class.return_value = client
            result = runner.invoke(app, ["cache-stats", *args])
        return result, mock_fc_class, client

    def test_table_renders_hit_rate_and_pools(self):
        result, _, client = self._run()
        assert result.exit_code == 0, result.output
        assert "Cache Statistics" in result.stdout
        assert "80.0%" in result.stdout  # 80/(80+20)
        assert "Per-pool Statistics" in result.stdout
        assert "unified-tiny" in result.stdout
        client.close.assert_called_once()

    def test_json_emits_the_raw_dict(self):
        result, _, _ = self._run("--json")
        assert result.exit_code == 0, result.output
        payload = json.loads(result.stdout.strip().splitlines()[-1])
        assert payload["hits"] == 80
        assert payload["pool_stats"]["unified-tiny"]["segments"] == 2

    def test_an_empty_answer_is_not_rendered_as_zeros(self):
        # The server answered with no stats: a cache that was never initialized,
        # which is a different thing from every failure and now says so.
        result, _, _ = self._run(stats={})
        assert result.exit_code == 1
        assert "no cache statistics" in result.stderr

    def test_it_asks_for_no_client_side_cache(self):
        # A throwaway client asking about the *server's* cache must not allocate
        # one of its own.
        _, mock_fc_class, _ = self._run()
        assert mock_fc_class.call_args.kwargs["cache_bytes"] == 0

    def test_hit_rate_guards_an_empty_cache(self):
        from biopb.tensor import cli as tensor_cli

        assert tensor_cli._hit_rate(0, 0) == "n/a"
        assert tensor_cli._hit_rate(3, 1) == "75.0%"

    def test_explicit_token_reaches_the_client(self):
        with patch("biopb.tensor.cli.TensorFlightClient") as mock_fc_class:
            mock_fc_class.return_value.cache_stats.return_value = self._STATS
            result = runner.invoke(app, ["cache-stats", "--token", "secret"])
        assert result.exit_code == 0, result.output
        assert mock_fc_class.call_args.kwargs["token"] == "secret"

    def test_it_dials_the_endpoint_it_resolved(self):
        """End to end with only the Flight client — the true boundary — replaced.

        Nothing stubs the resolver: #615's crash (and the misreports before it)
        all lived in the function every other cache-stats test used to replace,
        so this asserts on the location the client was actually constructed with.
        """
        with patch("biopb.tensor.cli.TensorFlightClient") as mock_fc_class:
            mock_fc_class.return_value.cache_stats.return_value = self._STATS
            result = runner.invoke(app, ["cache-stats"])
        assert result.exit_code == 0, result.output
        kwargs = mock_fc_class.call_args.kwargs
        # No control answered (see the autouse fixture), so this is the default
        # endpoint -- base+5, derived, not the literal 8815 spelled in a command.
        from biopb import _data_plane

        assert kwargs["location"] == _data_plane.default_url()
        assert kwargs["token"] is None

    def test_the_control_plane_decides_the_endpoint(self, monkeypatch):
        """A published endpoint wins over the default — #615's central claim."""
        monkeypatch.setattr(
            "biopb._data_plane.control_grpc_url",
            lambda timeout=1.0: "grpc://127.0.0.1:9915",
        )
        with patch("biopb.tensor.cli.TensorFlightClient") as mock_fc_class:
            mock_fc_class.return_value.cache_stats.return_value = self._STATS
            result = runner.invoke(app, ["cache-stats"])
        assert result.exit_code == 0, result.output
        assert mock_fc_class.call_args.kwargs["location"] == "grpc://127.0.0.1:9915"

    def test_an_auth_failure_is_not_reported_as_unreachable(self):
        """#615 fault 3: every failure rendered as "server unreachable".

        The server *answered* — it refused the dial — so the message has to name
        the token, not send the reader looking for a dead process.
        """
        import pyarrow.flight as flight

        with patch("biopb.tensor.cli.TensorFlightClient") as mock_fc_class:
            mock_fc_class.side_effect = flight.FlightUnauthenticatedError("no token")
            result = runner.invoke(app, ["cache-stats"])
        assert result.exit_code == 1
        assert "requires an access token" in result.stderr
        assert "unreachable" not in result.stderr.lower()

    def test_an_explicit_endpoint_is_told_why_the_credential_file_was_skipped(self):
        """Otherwise the reader knows a credential exists and blames the file.

        A named endpoint is deliberately not dialed with the control's credential
        (that token belongs to the plane the control owns), so the message has to
        say the file was *unused*, not missing or broken.
        """
        import pyarrow.flight as flight

        with patch("biopb.tensor.cli.TensorFlightClient") as mock_fc_class:
            mock_fc_class.side_effect = flight.FlightUnauthenticatedError("no token")
            result = runner.invoke(
                app, ["cache-stats", "--server", "grpc://data.mylab.example:8815"]
            )
        assert result.exit_code == 1
        assert "requires an access token" in result.stderr
        assert "credential" in result.stderr
        assert "--token" in result.stderr

    def test_a_rejected_token_says_so(self):
        import pyarrow.flight as flight

        with patch("biopb.tensor.cli.TensorFlightClient") as mock_fc_class:
            mock_fc_class.side_effect = flight.FlightUnauthenticatedError("bad")
            result = runner.invoke(app, ["cache-stats", "--token", "wrong"])
        assert result.exit_code == 1
        assert "rejected the token" in result.stderr

    def test_a_genuinely_unreachable_server_says_unreachable(self):
        import pyarrow.flight as flight

        with patch("biopb.tensor.cli.TensorFlightClient") as mock_fc_class:
            mock_fc_class.side_effect = flight.FlightUnavailableError("refused")
            result = runner.invoke(app, ["cache-stats"])
        assert result.exit_code == 1
        assert "Cannot reach the data plane" in result.stderr
        # ... and, on the guessed default, that --server exists for the rest.
        assert "--server" in result.stderr

    def test_an_unreadable_local_cert_is_not_reported_as_an_auth_problem(
        self, monkeypatch
    ):
        """A LocalTrustError stringifies as "Permission denied" (biopb/biopb#610).

        Classified by type, so it names the certificate rather than a token.
        """
        from biopb import _data_plane

        monkeypatch.setattr(
            "biopb._data_plane.control_grpc_url",
            lambda timeout=1.0: "grpcs://127.0.0.1:8815",
        )
        result = runner.invoke(app, ["cache-stats"])  # no cert in the state dir
        assert result.exit_code == 1
        # Rich hard-wraps, so match on words rather than the whole sentence.
        assert "certificate" in result.stderr and "cert init" in result.stderr
        assert "token" not in result.stderr
        assert _data_plane.LocalTrustError is not None  # the type, not a substring


class TestEveryCommandClassifiesItsFailures:
    """The classifier has to reach all five commands, not just cache-stats.

    ``TensorFlightClient`` opens its socket lazily, so a refused dial does not
    raise where the client is *built* -- it raises on the command's first RPC,
    inside the command body. Until this, only ``cache-stats`` routed that body
    through the classifier, so `query`/`metadata`/`get`/`stats` still printed
    "Error querying server: <raw exception>" for exactly the failures
    biopb/biopb#615 was filed about.
    """

    # (argv, the client method whose call is the command's first RPC)
    CASES = [
        (["query"], "list_sources"),
        (["metadata", "my-source"], "list_sources"),
        (["get", "my-source", "-o", "-"], "get_tensor_pb"),
        (["stats", "my-source"], "get_tensor"),
        (["cache-stats"], "cache_stats"),
    ]

    def _run(self, argv, method, exc):
        with patch("biopb.tensor.cli.TensorFlightClient") as mock_fc_class:
            client = mock_fc_class.return_value
            getattr(client, method).side_effect = exc
            return runner.invoke(app, argv)

    @pytest.mark.parametrize("argv,method", CASES)
    def test_a_missing_token_is_named_as_one(self, argv, method):
        import pyarrow.flight as flight

        result = self._run(argv, method, flight.FlightUnauthenticatedError("no token"))

        assert result.exit_code == 1
        assert "requires an access token" in result.stderr
        assert "unreachable" not in result.stderr.lower()

    @pytest.mark.parametrize("argv,method", CASES)
    def test_an_unreachable_plane_says_so_and_names_the_endpoint(self, argv, method):
        import pyarrow.flight as flight
        from biopb import _data_plane

        result = self._run(argv, method, flight.FlightUnavailableError("refused"))

        assert result.exit_code == 1
        assert "Cannot reach the data plane" in result.stderr
        # The origin is part of the message: a guessed default is not the same
        # failure as an endpoint the control published.
        assert _data_plane.default_url() in result.stderr

    def test_a_local_failure_keeps_the_command_s_own_words(self):
        """Not everything that goes wrong in a command body is the plane's doing.

        A bad slice or an unwritable file must not be reported as though the
        server said something -- that would just trade one misattribution for
        another.
        """
        with patch("biopb.tensor.cli.TensorFlightClient") as mock_fc_class:
            client = mock_fc_class.return_value
            client.get_tensor.side_effect = MemoryError("cannot allocate")
            result = runner.invoke(app, ["stats", "my-source"])

        assert result.exit_code == 1
        assert "Failed to compute statistics" in result.stderr
        assert "data plane" not in result.stderr
