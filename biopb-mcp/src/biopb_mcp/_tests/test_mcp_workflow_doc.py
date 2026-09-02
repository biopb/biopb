"""Tests for the workflow document parser (_workflow_doc.py).

What is under test is the contract the agent writes against: which fences are
cells, what happens to everything else, and whether a document survives the trip
to disk and back -- which is what the draft/retry loop rests on.
"""

import json

import pytest

from biopb_mcp.mcp import _workflow_doc as doc

_DOC = """# Count foci

Load the stack and threshold it. The threshold is 0.4 because the background
peak sits at 0.3.

```python
import numpy as np
from biopb_mcp.workflow_env import workflow_env

client, ops = workflow_env()
```

Run the saved notebook from a shell with:

```bash
jupyter nbconvert --execute wf.ipynb
```

```python
print("done")
```
"""


class TestMarkdown:
    def test_fences_are_cells_and_everything_else_is_prose(self):
        blocks = doc.parse(_DOC)
        assert [b["kind"] for b in blocks] == ["markdown", "code", "markdown", "code"]
        assert blocks[1]["text"].startswith("import numpy as np")
        assert blocks[3]["text"] == 'print("done")'

    def test_a_fence_in_another_language_stays_prose(self):
        # It is a command someone might type, not a cell to run.
        blocks = doc.parse(_DOC)
        assert "nbconvert" in blocks[2]["text"]
        assert "```bash" in blocks[2]["text"]
        assert "nbconvert" not in "\n".join(doc.code_cells(blocks))

    def test_the_title_is_read_from_the_document(self):
        # Written once, by the author, in the document -- not asked for again.
        assert doc.title_of(doc.parse(_DOC)) == "Count foci"
        assert doc.title_of(doc.parse("```python\n1\n```"), "fallback") == "fallback"

    def test_a_document_with_no_cells_is_refused(self):
        with pytest.raises(doc.DocumentError, match="no ```python cells"):
            doc.parse("# Title\n\nAll prose.\n")

    def test_an_empty_document_is_refused(self):
        with pytest.raises(doc.DocumentError, match="empty"):
            doc.parse("   \n")

    def test_an_unclosed_fence_runs_to_the_end_rather_than_failing(self):
        # The agent's typo. "Your last cell is long" is more use than refusing
        # the whole run.
        blocks = doc.parse("# T\n\n```python\na = 1\nb = 2\n")
        assert doc.code_cells(blocks) == ["a = 1\nb = 2"]

    def test_blank_prose_between_two_cells_is_not_a_block(self):
        blocks = doc.parse("```python\na = 1\n```\n\n\n```python\na = 2\n```\n")
        assert [b["kind"] for b in blocks] == ["code", "code"]


class TestNotebookInput:
    """The saved `.ipynb` comes back: that is the round trip at the point a
    person would use it -- they edited the file the last pass wrote."""

    def _nb(self, *cells):
        return json.dumps({"cells": list(cells), "nbformat": 4})

    def test_a_notebook_parses_to_the_same_shape(self):
        blocks = doc.parse(
            self._nb(
                {"cell_type": "markdown", "source": ["# Count foci\n", "\n", "Prose."]},
                {"cell_type": "code", "source": ["a = 2\n"]},
            )
        )
        assert [b["kind"] for b in blocks] == ["markdown", "code"]
        assert blocks[1]["text"] == "a = 2"
        assert doc.title_of(blocks) == "Count foci"

    def test_an_empty_cell_is_dropped(self):
        blocks = doc.parse(
            self._nb(
                {"cell_type": "code", "source": []},
                {"cell_type": "code", "source": "a = 2"},
            )
        )
        assert len(blocks) == 1

    def test_a_notebook_with_no_code_is_refused(self):
        with pytest.raises(doc.DocumentError, match="no code cells"):
            doc.parse(self._nb({"cell_type": "markdown", "source": ["hi"]}))

    def test_broken_json_says_so_rather_than_being_read_as_markdown(self):
        with pytest.raises(doc.DocumentError, match="Not readable"):
            doc.parse('{"cells": [')


def test_a_document_survives_the_trip_to_disk_and_back():
    # The draft is written in this spelling and sent back verbatim, so what
    # round-trips has to be the same document -- otherwise a retry silently
    # verifies something else.
    once = doc.parse(_DOC)
    twice = doc.parse(doc.to_markdown(once))
    assert once == twice
