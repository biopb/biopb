"""CLI integration tests for the biopb image commands, against an Ops server
(the ``ops_server`` fixture in conftest.py)."""

import json
import pickle
import subprocess
from pathlib import Path

import imageio
import numpy as np
from biopb.tensor.serialized_pb2 import SerializedTensor


def _biopb(*args, text=True, timeout=60):
    return subprocess.run(
        ["biopb", "image", *args], capture_output=True, text=text, timeout=timeout
    )


class TestImageCliOps:
    """Tests for 'biopb image ops'."""

    def test_ops_lists_operations_and_their_arguments(self, ops_server):
        result = _biopb("ops", "--server", f"grpc://{ops_server}")
        assert result.returncode == 0
        assert "mock_echo" in result.stdout
        assert "mock_random" in result.stdout
        assert "image: YX" in result.stdout
        assert "seed=0" in result.stdout

    def test_ops_connection_error(self):
        result = _biopb("ops", "--server", "grpc://invalid:9999", timeout=30)
        assert result.returncode == 1
        assert "error" in result.stderr.lower()


def _png(tmp_path: Path, shape=(128, 128)) -> Path:
    path = tmp_path / "input.png"
    imageio.imwrite(str(path), np.random.randint(0, 255, shape, dtype=np.uint8))
    return path


class TestImageCliProcess:
    """Tests for 'biopb image process'."""

    def test_eager_in_eager_out(self, ops_server, tmp_path):
        output = tmp_path / "out.png"
        result = _biopb(
            "process",
            str(_png(tmp_path, (256, 256))),
            "--op",
            "mock_echo",
            "--output",
            str(output),
            "--server",
            f"grpc://{ops_server}",
        )
        assert result.returncode == 0, result.stderr
        assert imageio.imread(str(output)).shape == (256, 256)

    def test_a_large_result_comes_back_by_reference(self, ops_server, tmp_path):
        # Over the 64 MB inline cap, so the server returns it through its
        # embedded tensor server.
        path = tmp_path / "large.tif"
        imageio.imwrite(str(path), np.random.rand(8192, 2049).astype(np.float32))
        result = _biopb(
            "process",
            str(path),
            "--op",
            "mock_echo",
            "--output",
            "-",
            "--server",
            f"grpc://{ops_server}",
            text=False,
            timeout=120,
        )
        assert result.returncode == 0, result.stderr
        assert SerializedTensor.FromString(result.stdout).location.startswith("grpc://")

    def test_a_reference_as_pickle(self, ops_server, tmp_path):
        result = _biopb(
            "process",
            str(_png(tmp_path)),
            "--op",
            "mock_echo_lazy",
            "--output",
            "-",
            "--format",
            "pickle",
            "--server",
            f"grpc://{ops_server}",
            text=False,
        )
        assert result.returncode == 0, result.stderr
        serialized = pickle.loads(result.stdout)
        assert isinstance(serialized, SerializedTensor)
        assert serialized.location.startswith("grpc://")

    def test_eager_to_stdout_is_refused(self, ops_server, tmp_path):
        result = _biopb(
            "process",
            str(_png(tmp_path)),
            "--op",
            "mock_echo",
            "--output",
            "-",
            "--server",
            f"grpc://{ops_server}",
        )
        assert result.returncode == 1
        assert "stdout not allowed" in result.stderr.lower()

    def test_kwargs_and_labels(self, ops_server, tmp_path):
        output = tmp_path / "labels.png"
        result = _biopb(
            "process",
            str(_png(tmp_path, (64, 48))),
            "--op",
            "mock_random",
            "--kwargs",
            '{"seed": 3}',
            "--output",
            str(output),
            "--server",
            f"grpc://{ops_server}",
        )
        assert result.returncode == 0, result.stderr
        labels = imageio.imread(str(output))
        assert labels.shape == (64, 48) and labels.max() <= 4

    def test_json_and_tensor_outputs(self, ops_server, tmp_path):
        output = tmp_path / "echo.png"
        image = _png(tmp_path)
        result = _biopb(
            "process",
            str(image),
            "--op",
            "mock_stats",
            "--kwargs",
            '{"scale": 2}',
            "--output",
            str(output),
            "--server",
            f"grpc://{ops_server}",
        )
        assert result.returncode == 0, result.stderr
        assert output.exists()
        mean = imageio.imread(str(image)).mean() * 2
        key, text = result.stdout.strip().split(": ", 1)
        assert key == "0"
        assert abs(json.loads(text)["mean"] - mean) < 1e-6

    def test_an_unknown_op_is_named(self, ops_server, tmp_path):
        result = _biopb(
            "process",
            str(_png(tmp_path)),
            "--op",
            "nope",
            "--server",
            f"grpc://{ops_server}",
        )
        assert result.returncode == 1
        assert "mock_echo" in result.stderr
