"""Debug and introspection utilities for biopb services."""

import functools
import logging
import os
import platform
import subprocess
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

@dataclass
class ServiceStats:
    """Thread-safe request statistics tracker."""

    request_count: int = 0
    error_count: int = 0
    total_latency_ms: float = 0.0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def record_request(self, latency_ms: float, error: bool = False):
        """Record a request with its latency."""
        with self._lock:
            self.request_count += 1
            self.total_latency_ms += latency_ms
            if error:
                self.error_count += 1

    def get_stats(self) -> dict:
        """Get current statistics."""
        with self._lock:
            avg_latency = (
                self.total_latency_ms / self.request_count
                if self.request_count > 0
                else 0.0
            )
            return {
                "request_count": self.request_count,
                "error_count": self.error_count,
                "total_latency_ms": round(self.total_latency_ms, 2),
                "avg_latency_ms": round(avg_latency, 2),
            }

    def reset(self):
        """Reset statistics."""
        with self._lock:
            self.request_count = 0
            self.error_count = 0
            self.total_latency_ms = 0.0


# Global stats instance
_global_stats = ServiceStats()


def get_stats() -> ServiceStats:
    """Get the global service statistics tracker."""
    return _global_stats


def get_gpu_memory_info() -> Optional[dict]:
    """Get GPU memory information using nvidia-smi.

    Returns:
        dict with memory info or None if NVIDIA GPU not available.
    """
    try:
        # Query nvidia-smi for GPU info
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.used,memory.free", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
        )

        if result.returncode != 0:
            return None

        # Parse output: "GPU Name, Total, Used, Free"
        lines = result.stdout.strip().split("\n")
        if not lines or not lines[0]:
            return None

        # Use first GPU
        parts = lines[0].split(", ")
        if len(parts) < 4:
            return None

        name = parts[0].strip()
        total_mb = float(parts[1].strip())
        used_mb = float(parts[2].strip())
        free_mb = float(parts[3].strip())

        return {
            "device": name,
            "total_mb": round(total_mb, 2),
            "allocated_mb": round(used_mb, 2),  # used = allocated in nvidia-smi terms
            "reserved_mb": round(used_mb, 2),   # approximate
            "free_mb": round(free_mb, 2),
        }
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
        return None
    except Exception:
        return None


def get_system_info() -> dict:
    """Get system information including CPU, memory, and platform details.

    Returns:
        dict with system information.
    """
    info = {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "cpu_count": os.cpu_count(),
    }

    # Try to get memory info
    try:
        with open("/proc/meminfo") as f:
            meminfo = {}
            for line in f:
                parts = line.split(":")
                if len(parts) == 2:
                    key = parts[0].strip()
                    value = parts[1].strip().split()[0]
                    meminfo[key] = int(value)
            if "MemTotal" in meminfo:
                info["memory_total_mb"] = round(meminfo["MemTotal"] / 1024, 2)
            if "MemAvailable" in meminfo:
                info["memory_available_mb"] = round(meminfo["MemAvailable"] / 1024, 2)
    except (FileNotFoundError, PermissionError):
        pass

    # Add GPU info if available
    gpu_info = get_gpu_memory_info()
    if gpu_info:
        info["gpu"] = gpu_info

    return info


def get_model_info(model: Any) -> dict:
    """Extract model metadata.

    Args:
        model: The model instance

    Returns:
        dict with model information.
    """
    info = {"model_type": type(model).__name__}

    # Try common attributes
    for attr in ["model_type", "name", "version", "__version__"]:
        if hasattr(model, attr):
            val = getattr(model, attr)
            if isinstance(val, str):
                info[attr] = val

    return info


def profile_memory(func: Callable) -> Callable:
    """Decorator to profile memory usage of a function.

    Logs memory delta before and after function execution.
    Uses nvidia-smi for GPU memory tracking.
    Only logs at DEBUG level (skipped if log level >= INFO).
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Skip profiling if not DEBUG level
        if not logger.isEnabledFor(logging.DEBUG):
            return func(*args, **kwargs)

        gpu_before = get_gpu_memory_info()
        start_time = time.perf_counter()

        try:
            result = func(*args, **kwargs)
            error = None
        except Exception as e:
            error = e
            result = None

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        gpu_after = get_gpu_memory_info()

        # Log memory delta
        if gpu_before and gpu_after:
            delta = gpu_after["allocated_mb"] - gpu_before["allocated_mb"]
            logger.debug(
                f"{func.__name__}: {elapsed_ms:.2f}ms, GPU memory delta: {delta:+.2f}MB"
            )
        else:
            logger.debug(f"{func.__name__}: {elapsed_ms:.2f}ms")

        if error:
            raise error

        return result

    return wrapper


def timed(func: Callable) -> Callable:
    """Decorator to time function execution and record in stats.

    Only logs at DEBUG level (skipped if log level >= INFO).
    Stats are always recorded regardless of log level.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        error = False

        try:
            result = func(*args, **kwargs)
        except Exception:
            error = True
            raise
        finally:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            _global_stats.record_request(elapsed_ms, error)

            # Log timing only at DEBUG level
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"{func.__name__}: {elapsed_ms:.2f}ms")

        return result

    return wrapper
