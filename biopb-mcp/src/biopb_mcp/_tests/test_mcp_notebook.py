"""Tests for the audit-notebook serializer (_notebook.py).

Pure-function tests: build_notebook turns a list of _jobs.export() snapshots into
an nbformat-v4 dict. No kernel, no Qt.
"""

import json

from biopb_mcp.mcp import _notebook


def _snap(**kw):
    base = {
        "job_id": "job-1",
        "code": "x = 1\nx",
        "status": "ok",
        "stdout": "hello\n",
        "result_text": "1",
        "error_text": "",
        "cancel_reason": None,
        "elapsed": 0.1,
        "created": 1_700_000_000.0,
    }
    base.update(kw)
    return base


def test_empty_session_is_valid_notebook():
    nb = _notebook.build_notebook([])
    assert nb["nbformat"] == 4
    # title + bootstrap + "no jobs" note, all serializable.
    json.dumps(nb)
    assert nb["cells"][0]["cell_type"] == "markdown"
    assert any("build_ops" in "".join(c["source"]) for c in nb["cells"])


def test_one_job_cell_structure():
    nb = _notebook.build_notebook([_snap()])
    code = [c for c in nb["cells"] if c["cell_type"] == "code"]
    # bootstrap + one job
    assert len(code) == 2
    job = code[-1]
    src = "".join(job["source"])
    assert "# [job-1 · ok · 0.1s ·" in src  # audit header comment
    assert "x = 1" in src
    assert job["metadata"]["biopb"]["job_id"] == "job-1"
    # stdout -> stream, result_text -> execute_result
    kinds = {o["output_type"] for o in job["outputs"]}
    assert kinds == {"stream", "execute_result"}


def test_interrupted_job_kept_as_code_with_reason_in_output():
    reason = "Interrupted by user via the observe web UI."
    nb = _notebook.build_notebook(
        [_snap(status="interrupted", error_text=reason, cancel_reason=reason)]
    )
    job = [c for c in nb["cells"] if c["cell_type"] == "code"][-1]
    assert job["cell_type"] == "code"  # not demoted to markdown
    assert "interrupted" in "".join(job["source"])
    stderr = [o for o in job["outputs"] if o.get("name") == "stderr"]
    assert stderr and reason in "".join(stderr[0]["text"])


def test_ordering_and_count_in_intro():
    jobs = [_snap(job_id="job-1"), _snap(job_id="job-2", code="y = 2")]
    nb = _notebook.build_notebook(jobs)
    code = [c for c in nb["cells"] if c["cell_type"] == "code"]
    assert "job-1" in "".join(code[1]["source"])
    assert "job-2" in "".join(code[2]["source"])
    assert "2 job(s)" in "".join(nb["cells"][0]["source"])


def test_headless_note_and_filename():
    nb = _notebook.build_notebook([], headless=True)
    assert "headless" in "".join(nb["cells"][0]["source"]).lower()
    assert _notebook.suggested_filename().endswith(".ipynb")
    assert _notebook.suggested_filename().startswith("biopb-mcp-session-")
