"""The skills authoring gate — structure, contract and retrieval.

The deterministic layers of ``biopb-mcp/docs/skill-testing.md``: no agent, no session, no
display. They run against the skills this package ships
(``biopb_mcp/mcp/_skills_data/*.md``), which is why they live here rather than
beside the website that used to publish them — see §1 of that doc.

Two readers exist for the same files, on purpose. ``mcp/_skills.py`` is
**tolerant**: it is on the agent's path, so a malformed file must degrade to a
skipped entry rather than an error. :mod:`._validate` here is **strict**: it is
on the author's path, where a missing section or a bad ``checklist:`` token
should stop the PR.
"""
