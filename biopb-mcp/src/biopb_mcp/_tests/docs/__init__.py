"""The deterministic layers of the knowledge store: packaging, seed, contracts.

No agent, no session, no display. They run against the store this package ships
(``biopb_mcp/mcp/_docs_data/*.md``).

There is no schema layer any more, and that is the point of the redesign: a doc
is a markdown file with optional frontmatter, and the tolerant reader in
``mcp/_docs.py`` is the only reader. What is left here is what a reader cannot
check -- that the files reach the wheel, that the seed index and the seed agree,
and that the third-party APIs a body quotes still look the way it says.
"""
