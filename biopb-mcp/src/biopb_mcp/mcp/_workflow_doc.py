"""The workflow document: what `verify_workflow` is handed, as blocks.

The tool takes a *document*, not a list of cells with a list of notes beside it.
A document is what the agent is good at writing and what a person can edit, and
it keeps prose from being a tool parameter -- every heading, aside and caption a
workflow wants would otherwise be another argument with another alignment rule.

Two input spellings, both parsed here:

* **Markdown with fenced ``python`` blocks** (jupytext's ``md`` flavour). What
  the agent writes: prose is prose, code is a fence. This is also what the draft
  on disk stays, so a document read back is a document that can be sent back.
* **``.ipynb`` JSON.** What the *reader* has after a run passes. Accepting it is
  what closes the loop: the user edits the saved notebook in VS Code, hands it
  back, and it is verified as it stands rather than retyped.

A fence in some other language is prose about a command, not a cell, so it stays
inside its markdown block.
"""

import json

#: Fence info strings that make a block a code cell. Anything else -- ``bash``,
#: ``text``, no info at all -- is a fenced quotation inside the prose.
_CODE_LANGS = frozenset({"python", "py", "python3"})

_FENCE = "```"


class DocumentError(ValueError):
    """The document could not be read as one."""


def parse(text):
    """Parse *text* into ``[{"kind": "markdown"|"code", "text": ...}, ...]``.

    Blank markdown between two cells is dropped; blank code is not possible.
    """
    if text is None or not text.strip():
        raise DocumentError("The document is empty.")
    stripped = text.lstrip()
    if stripped.startswith("{"):
        return _from_ipynb(stripped)
    return _from_markdown(text)


def _from_markdown(text):
    blocks = []
    prose = []
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if not line.lstrip().startswith(_FENCE):
            prose.append(line)
            i += 1
            continue
        info = line.strip()[len(_FENCE) :].strip().lower()
        body, close, i = _read_fence(lines, i + 1)
        if info in _CODE_LANGS:
            _flush(blocks, prose)
            blocks.append({"kind": "code", "text": "\n".join(body).strip("\n")})
        else:
            # Not a cell: a fenced quotation. Kept verbatim, closing fence and
            # all, so the prose renders as it was written.
            prose.append(line)
            prose.extend(body)
            if close is not None:
                prose.append(close)
    _flush(blocks, prose)
    if not any(b["kind"] == "code" for b in blocks):
        raise DocumentError("The document has no ```python cells.")
    return blocks


def _read_fence(lines, i):
    """Consume a fenced block from *i*; return ``(body, closing line, next i)``.

    An unclosed fence runs to the end of the document rather than raising: it is
    the agent's typo, and reporting it as "your last cell is long" is more use
    than refusing the whole run.
    """
    body = []
    while i < len(lines):
        if lines[i].lstrip().startswith(_FENCE):
            return body, lines[i], i + 1
        body.append(lines[i])
        i += 1
    return body, None, i


def _flush(blocks, prose):
    text = "\n".join(prose).strip("\n")
    prose.clear()
    if text.strip():
        blocks.append({"kind": "markdown", "text": text})


def _from_ipynb(text):
    try:
        nb = json.loads(text)
    except ValueError as exc:
        raise DocumentError(f"Not readable as a notebook: {exc}") from None
    cells = nb.get("cells")
    if not isinstance(cells, list):
        raise DocumentError("Not a notebook: no cells.")
    blocks = []
    for cell in cells:
        if not isinstance(cell, dict):
            continue
        kind = "code" if cell.get("cell_type") == "code" else "markdown"
        source = cell.get("source") or ""
        body = ("".join(source) if isinstance(source, list) else str(source)).strip(
            "\n"
        )
        if body.strip():
            blocks.append({"kind": kind, "text": body})
    if not any(b["kind"] == "code" for b in blocks):
        raise DocumentError("The notebook has no code cells.")
    return blocks


def code_cells(blocks):
    """Just the code, in order -- what actually gets run."""
    return [b["text"] for b in blocks if b["kind"] == "code"]


def to_markdown(blocks):
    """Render *blocks* back to the markdown spelling, for the draft on disk."""
    out = []
    for block in blocks:
        if block["kind"] == "code":
            out.append(f"{_FENCE}python\n{block['text']}\n{_FENCE}")
        else:
            out.append(block["text"])
    return "\n\n".join(out) + "\n"


def title_of(blocks, default=""):
    """The document's own ``# `` heading, or *default*.

    Read from the document rather than asked for separately: the title is part
    of what the agent wrote, and a second parameter is a second thing to keep in
    step with it.
    """
    for block in blocks:
        if block["kind"] != "markdown":
            continue
        for line in block["text"].splitlines():
            if line.startswith("# "):
                return line[2:].strip()
        break
    return default
