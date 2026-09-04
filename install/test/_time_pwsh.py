"""Time one PowerShell spawn, with and without the module-analysis cache vars.

Not a test -- a diagnostic the Windows CI leg runs so the cost of that leg is a
number in the log rather than something the next person rediscovers by watching a
job hit its timeout.

The Windows leg spawns a fresh Windows PowerShell 5.1 per case, each dot-sourcing
biopb-engine.ps1. That measured ~22s per spawn on a stock runner against ~0.2s
for the bash cases interleaved with it. The cause was not the engine and not
Defender: conftest hands the interpreter a controlled environment, and with
LOCALAPPDATA/APPDATA missing from it PowerShell can neither read nor write its
module analysis cache, so every spawn rebuilt it by walking every module on a
runner that ships Az, AWS and friends. A developer machine has few enough modules
that the same code path costs nothing, which is why it only ever showed up here.

Prints both numbers. The first is what the suite actually pays; the second is what
it would pay if those variables were dropped from _pwsh_base_env again.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import conftest  # noqa: E402


def timed(label: str, runs: int = 3) -> float:
    """Best of `runs`, so a cold first spawn does not stand in for the steady cost."""
    best = None
    for _ in range(runs):
        started = time.monotonic()
        conftest.pwsh("1 | Out-Null")
        elapsed = time.monotonic() - started
        best = elapsed if best is None else min(best, elapsed)
    print(f"  {label:<34} {best:6.2f}s")
    return best


def main() -> int:
    if conftest.PWSH is None:
        print("no PowerShell interpreter; nothing to time")
        return 0

    print(f"Timing {conftest.PWSH}")
    print("  (dot-sources biopb-engine.ps1 once per spawn, as every case does)")
    with_cache = timed("with LOCALAPPDATA/APPDATA")

    # Drop them from the REAL environment, which is what _pwsh_base_env copies
    # from -- so the second number is the old behaviour, measured, not asserted.
    for var in ("LOCALAPPDATA", "APPDATA"):
        os.environ.pop(var, None)
    without_cache = timed("without them (the old cost)")

    if without_cache > 0:
        print(f"\n  speedup: {without_cache / max(with_cache, 1e-6):.1f}x")
    # Never fails the job: this measures the runner, and a runner that is slow for
    # some other reason is a thing to read about, not a red build. The suite's own
    # job timeout is what actually guards the cost.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
