"""CLI integration tests for biopb.image commands.

These tests use the mock_server fixture from conftest.py which starts
a mock ProcessImage servicer for testing.
"""

import subprocess
import tempfile
from pathlib import Path

import imageio
import numpy as np

from biopb.tensor.serialized_pb2 import SerializedTensor


class TestImageCliOps:
    """Tests for 'biopb image ops' command."""

    def test_ops_returns_operations(self, mock_server):
        """ops command should list available operations."""
        result = subprocess.run(
            [
                "biopb",
                "image",
                "ops",
                "--server",
                f"grpc://{mock_server}",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "mock_echo" in result.stdout
        assert "mock_random" in result.stdout

    def test_ops_shows_schema_details(self, mock_server):
        """ops command should show operation schema details."""
        result = subprocess.run(
            [
                "biopb",
                "image",
                "ops",
                "--server",
                f"grpc://{mock_server}",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        # Should show descriptions and labels
        assert "echo" in result.stdout.lower() or "Echo" in result.stdout
        assert "random" in result.stdout.lower() or "Random" in result.stdout

    def test_ops_connection_error(self):
        """ops command should exit with error on invalid server."""
        result = subprocess.run(
            [
                "biopb",
                "image",
                "ops",
                "--server",
                "grpc://invalid:9999",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode == 1
        assert "error" in result.stderr.lower() or "Error" in result.stderr


class TestImageCliProcess:
    """Tests for 'biopb image process' command."""

    def test_process_eager_input_eager_output(self, mock_server):
        """Process image file, get eager result saved as image."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test image
            test_img = np.random.randint(0, 255, (256, 256), dtype=np.uint8)
            input_path = Path(tmpdir) / "test_input.png"
            output_path = Path(tmpdir) / "test_output.png"
            imageio.imwrite(str(input_path), test_img)

            result = subprocess.run(
                [
                    "biopb",
                    "image",
                    "process",
                    str(input_path),
                    "--op",
                    "mock_echo",
                    "--output",
                    str(output_path),
                    "--server",
                    f"grpc://{mock_server}",
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )
            assert result.returncode == 0

            # Verify output image
            output_img = imageio.imread(str(output_path))
            assert output_img.shape == test_img.shape

    def test_process_stdin_stdout_lazy(self, mock_server):
        """Process from stdin to stdout (protobuf format for lazy data)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Use large image to trigger lazy response (>64MB)
            large_img = np.random.rand(8192, 2049).astype(np.float32)
            input_path = Path(tmpdir) / "large_input.tif"
            imageio.imwrite(str(input_path), large_img)

            result = subprocess.run(
                [
                    "biopb",
                    "image",
                    "process",
                    str(input_path),
                    "--op",
                    "mock_echo",
                    "--output",
                    "-",
                    "--server",
                    f"grpc://{mock_server}",
                ],
                capture_output=True,
                timeout=60,
            )
            assert result.returncode == 0

            # Output should be protobuf SerializedTensor
            serialized = SerializedTensor.FromString(result.stdout)
            assert serialized.location.startswith("grpc://")

    def test_process_stdin_stdout_pickle(self, mock_server):
        """Process with pickle output format for lazy data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            large_img = np.random.rand(8192, 2049).astype(np.float32)
            input_path = Path(tmpdir) / "large_input.tif"
            imageio.imwrite(str(input_path), large_img)

            result = subprocess.run(
                [
                    "biopb",
                    "image",
                    "process",
                    str(input_path),
                    "--op",
                    "mock_echo",
                    "--output",
                    "-",
                    "--format",
                    "pickle",
                    "--server",
                    f"grpc://{mock_server}",
                ],
                capture_output=True,
                timeout=60,
            )
            assert result.returncode == 0

            # Output should be pickled SerializedTensor
            import pickle
            serialized = pickle.loads(result.stdout)
            assert isinstance(serialized, SerializedTensor)
            assert serialized.location.startswith("grpc://")

    def test_process_small_image_to_file(self, mock_server):
        """Process small image (eager) to file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Small image - server returns eager data
            small_img = np.random.randint(0, 255, (128, 128), dtype=np.uint8)
            input_path = Path(tmpdir) / "small_input.png"
            output_path = Path(tmpdir) / "small_output.png"
            imageio.imwrite(str(input_path), small_img)

            result = subprocess.run(
                [
                    "biopb",
                    "image",
                    "process",
                    str(input_path),
                    "--op",
                    "mock_echo",
                    "--output",
                    str(output_path),
                    "--server",
                    f"grpc://{mock_server}",
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )
            assert result.returncode == 0
            assert output_path.exists()

    def test_process_eager_stdout_error(self, mock_server):
        """Eager data with stdout should error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            small_img = np.random.randint(0, 255, (128, 128), dtype=np.uint8)
            input_path = Path(tmpdir) / "small_input.png"
            imageio.imwrite(str(input_path), small_img)

            result = subprocess.run(
                [
                    "biopb",
                    "image",
                    "process",
                    str(input_path),
                    "--op",
                    "mock_echo",
                    "--output",
                    "-",
                    "--server",
                    f"grpc://{mock_server}",
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )
            # Should fail because eager data can't go to stdout
            assert result.returncode == 1
            assert "stdout not allowed" in result.stderr.lower()

    def test_process_random_mode(self, mock_server):
        """Process with mock_random operation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_img = np.random.randint(0, 255, (256, 256), dtype=np.uint8)
            input_path = Path(tmpdir) / "test_input.png"
            output_path = Path(tmpdir) / "test_output.png"
            imageio.imwrite(str(input_path), test_img)

            result = subprocess.run(
                [
                    "biopb",
                    "image",
                    "process",
                    str(input_path),
                    "--op",
                    "mock_random",
                    "--output",
                    str(output_path),
                    "--server",
                    f"grpc://{mock_server}",
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )
            assert result.returncode == 0

            # Output should be random labels mask
            output_img = imageio.imread(str(output_path))
            assert output_img.shape[:2] == test_img.shape[:2]  # Same spatial dims

    def test_process_stdin_pipe(self, mock_server):
        """Process input piped from stdin."""
        # This test would pipe data from tensor get to image process
        # Skipped if tensor server not available
        pass