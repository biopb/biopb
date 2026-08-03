"""Canonical skill frontmatter contract — stdlib only.

Tolerant on read (coercion, inference), strict on the result. All skill-file
format variation is absorbed here so what the author is held to is one thing.
See ``biopb-mcp/docs/skill-interface.md`` §1 and §5.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass

# Current authoring dialect. Older skill files declare a lower spec_version and
# are up-converted by migrate() in _validate, so what the gate checks is uniform.
CURRENT_SPEC_VERSION = 1

# Tags are lowercased and coerced to a list, but NOT checked against a fixed
# vocabulary: a closed set needs an edit for every new topic, and it fails the PR
# that introduces one -- for a judgment ("is this the right category?") the
# reviewer curating the catalog is already making by hand.

# Body structure every skill must carry, as normalized H2 headings (lowercased,
# surrounding punctuation stripped). Order is not enforced, extra sections are
# allowed. These are the sections a small model needs and cannot infer --
# especially "when not to use" -- and every one of them is answerable from the
# workflow the author has just run.
REQUIRED_SECTIONS = (
    "when to use",
    "when not to use",
    "parameters",
    "steps",
)

# Allowed, never required. Both are only worth writing from evidence, and a
# required section gets filled whether or not the author has any: asked for a
# failure table about a workflow that has not failed yet, an author writes what
# sounds plausible, and speculation is indistinguishable from observation once it
# is a table row. So they grow like a regression suite instead -- a row per
# failure someone actually hit. See docs/skill-interface.md 5b.
EVIDENCE_SECTIONS = (
    "failure modes",
    "next steps",
)

SEMVER = re.compile(r"^\d+\.\d+\.\d+$")
KEBAB = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")


@dataclass
class SkillEntry:
    """What validation yields per skill — strict and canonical.

    Narrower than the old published catalog entry: ``url`` and ``sha256``
    described where to fetch a body and how to verify it, and there is no fetch
    any more. ``updated`` is gone with them — it was derived from git history to
    stamp the generated index, and there is no index.
    """

    id: str
    title: str
    description: str
    tags: list
    version: str
    spec_version: int
    # What step 1 verifies against `server_status`, named `requires:` until the
    # rename. It informs the agent; it has never gated anything, and the old
    # name promised otherwise -- see skill-interface.md §4.
    checklist: list

    def to_dict(self) -> dict:
        return asdict(self)


def coerce_list(v) -> list:
    """Accept a list, a comma-separated string, a scalar, or None."""
    if v is None:
        return []
    if isinstance(v, list):
        return v
    if isinstance(v, str):
        return [s.strip() for s in v.split(",") if s.strip()]
    return [v]
