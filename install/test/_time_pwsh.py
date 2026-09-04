"""Print what one PowerShell spawn costs, so this leg's cost stays visible.

Not a test -- a diagnostic the Windows CI leg runs before the suite, because the
cost of that leg is invisible in a green run right up until it walks into a job
timeout, which is how it was found the first time.

What happened: the Windows leg spawns a fresh Windows PowerShell 5.1 per case,
each dot-sourcing biopb-engine.ps1. That cost ~22s per spawn -- 879s for the
suite -- against ~0.2s for the bash cases interleaved with it. The cause was the
module analysis cache: conftest hands the interpreter a controlled environment,
and with LOCALAPPDATA/APPDATA missing from it PowerShell could not WRITE that
cache, so every spawn rebuilt it by walking every module on a runner that ships
Az, AWS and friends. Passing them (conftest._pwsh_base_env) took the same suite
to 17s on the same runner -- measured across two runs, which is the only place
that comparison can honestly be made.

It cannot be made HERE, and an earlier version of this file tried: once any spawn
has written the cache, dropping the variables again is fast because the file is
already on disk. That version printed "speedup: 0.7x" next to a 50x real-world
win, which is worse than printing nothing. So this prints one number -- the
steady-state spawn cost -- and leaves the comparison to the commit history.

A number near 0.3s is healthy. A number near 20s means the cache is being
rebuilt per spawn again, and the suite is about to take a quarter of an hour.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import conftest  # noqa: E402

# Above this, something has gone wrong with the interpreter rather than the tests
# -- the 22s regression this exists to catch is two orders of magnitude over it.
SLOW_SPAWN_SECONDS = 5.0


def main() -> int:
    if conftest.PWSH is None:
        print("no PowerShell interpreter on PATH; nothing to time")
        return 0

    print(f"Timing {conftest.PWSH}")
    print("  (one spawn, dot-sourcing biopb-engine.ps1, as every case does)")

    # Best of several: the first spawn on a cold machine pays for warming the
    # cache once, which is expected and not what this is watching for.
    best = min(_one_spawn() for _ in range(3))
    print(f"  steady-state spawn: {best:.2f}s")

    if best > SLOW_SPAWN_SECONDS:
        print(
            f"\n  WARNING: over {SLOW_SPAWN_SECONDS:.0f}s per spawn. The whole suite is"
            f" ~{best * 43 / 60:.0f} min at this rate."
            "\n  Check that _pwsh_base_env still passes LOCALAPPDATA/APPDATA."
        )
    # Never fails the job: this measures the runner. The job timeout is what
    # actually guards the cost; this only says why, in advance.
    return 0


def _one_spawn() -> float:
    started = time.monotonic()
    conftest.pwsh("1 | Out-Null")
    return time.monotonic() - started


if __name__ == "__main__":
    raise SystemExit(main())
