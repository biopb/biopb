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
    assert "# [job-1 · mcp · ok · 0.1s ·" in src  # audit header comment
    assert "x = 1" in src
    assert job["metadata"]["biopb"]["job_id"] == "job-1"
    # Who ran it, in both places. A record predating `origin` reads as the
    # agent, which is what every pre-console session was.
    assert job["metadata"]["biopb"]["origin"] == "mcp"


def test_a_user_cell_is_attributed_in_the_export():
    # The export is an audit, so a human's cell must not read as the agent's:
    # `mask = mask > 0.7` from the observe page is indistinguishable otherwise.
    nb = _notebook.build_notebook([_snap(origin="user")])
    job = [c for c in nb["cells"] if c["cell_type"] == "code"][-1]
    assert "· user ·" in "".join(job["source"])
    assert job["metadata"]["biopb"]["origin"] == "user"
    # stdout -> stream, result_text -> execute_result
    kinds = {o["output_type"] for o in job["outputs"]}
    assert kinds == {"stream", "execute_result"}


def test_intent_becomes_a_markdown_cell_above_its_code():
    # The code is the only thing the session records natively; why it was run
    # exists nowhere unless it was passed in. It gets its own cell so it reads
    # as prose, and so a chat turn has a shape waiting for it.
    nb = _notebook.build_notebook([_snap(intent="find the drift between t0 and t1")])
    kinds = [c["cell_type"] for c in nb["cells"]]
    assert kinds[-2:] == ["markdown", "code"]
    note = "".join(nb["cells"][-2]["source"])
    assert "find the drift between t0 and t1" in note
    assert "job-1" in note and "mcp" in note
    assert (
        nb["cells"][-1]["metadata"]["biopb"]["intent"]
        == "find the drift between t0 and t1"
    )


def test_a_job_without_intent_gets_no_note_cell():
    # Optional, and an empty note would be worse than none: the export must not
    # grow a blank cell per job for a field nobody filled. A record predating
    # the field has no key at all, which must read the same as an empty one.
    for snap in (_snap(), _snap(intent=""), _snap(intent="   ")):
        nb = _notebook.build_notebook([snap])
        assert [c["cell_type"] for c in nb["cells"]] == ["markdown", "code", "code"]
        assert nb["cells"][-1]["metadata"]["biopb"]["intent"] == ""


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


def test_suggested_filename():
    assert _notebook.suggested_filename().endswith(".ipynb")
    assert _notebook.suggested_filename().startswith("biopb-mcp-session-")


def _record(**kw):
    base = {
        "title": "Count foci per cell",
        "created": 1_700_000_000.0,
        "status": "ok",
        "added_layers": [],
        "cells": [
            {
                "code": "a = 2",
                "status": "ok",
                "stdout": "",
                "result_text": "",
                "error_text": "",
                "elapsed": 0.1,
            },
            {
                "code": "print(a * 3)\na * 3",
                "status": "ok",
                "stdout": "6\n",
                "result_text": "6",
                "error_text": "",
                "elapsed": 0.2,
            },
        ],
    }
    base.update(kw)
    return base


def test_workflow_notebook_is_title_bootstrap_then_the_cells():
    nb = _notebook.build_workflow_notebook(_record())
    json.dumps(nb)
    assert [c["cell_type"] for c in nb["cells"]] == ["markdown", "code", "code", "code"]
    assert "Count foci per cell" in "".join(nb["cells"][0]["source"])
    assert "build_ops" in "".join(nb["cells"][1]["source"])


def test_a_workflow_cell_carries_no_audit_header():
    # An audit cell needs provenance because a reader has to know who ran it; a
    # workflow has one author, and a banner per cell is noise in a document
    # someone is meant to read and edit.
    nb = _notebook.build_workflow_notebook(_record())
    src = "".join(nb["cells"][-1]["source"])
    assert src == "print(a * 3)\na * 3"
    kinds = {o["output_type"] for o in nb["cells"][-1]["outputs"]}
    assert kinds == {"stream", "execute_result"}


def test_the_intro_states_what_the_run_did_and_did_not_prove():
    intro = "".join(_notebook.build_workflow_notebook(_record())["cells"][0]["source"])
    assert "scratch namespace" in intro
    assert "2 cell(s)" in intro
    # The residual is named, not hidden.
    assert "viewer" in intro and "sys.modules" in intro


def test_added_layers_are_reported_when_there_are_any():
    plain = "".join(_notebook.build_workflow_notebook(_record())["cells"][0]["source"])
    assert "added to the live viewer" not in plain
    noted = "".join(
        _notebook.build_workflow_notebook(_record(added_layers=["foci", "nuclei"]))[
            "cells"
        ][0]["source"]
    )
    assert "`foci`, `nuclei`" in noted


def test_an_empty_record_still_builds():
    # The export route 404s on a missing record, so this is defence against a
    # shape that arrived anyway rather than a path anyone takes.
    nb = _notebook.build_workflow_notebook(None)
    json.dumps(nb)
    assert "Verified workflow" in "".join(nb["cells"][0]["source"])


def test_suggested_workflow_filename_slugs_the_title():
    name = _notebook.suggested_workflow_filename("Count Foci / cell (v2)")
    assert name.startswith("biopb-count-foci-cell-v2-")
    assert name.endswith(".ipynb")
    # A title that slugs to nothing must still give a usable filename.
    assert _notebook.suggested_workflow_filename("///").startswith("biopb-workflow-")
    assert _notebook.suggested_workflow_filename("").startswith("biopb-workflow-")


def test_the_bootstrap_cell_is_valid_python():
    # It is shipped as a string and never imported, so nothing else would catch
    # a syntax error in it until someone opened the notebook.
    compile(_notebook.BOOTSTRAP_SRC, "<bootstrap>", "exec")


def test_the_bootstrap_cell_loads_kernel_plugins_after_the_handles():
    # A workflow calling `rolling_ball.subtract_background(...)` verified fine —
    # the scratch namespace has the plugins, because they are in the bootstrap
    # baseline — and then failed on a fresh kernel, because this cell rebuilt
    # every handle except them. Ordered last, like the kernel's own step 7b, so
    # a plugin can reference the handles above it.
    src = _notebook.BOOTSTRAP_SRC
    assert "_load_namespace_plugins" in src
    assert src.index("napari.Viewer()") < src.index("_load_namespace_plugins")


def test_both_intros_say_the_plugins_come_from_the_reader_s_machine():
    audit = "".join(_notebook.build_notebook([])["cells"][0]["source"])
    workflow = "".join(
        _notebook.build_workflow_notebook(_record())["cells"][0]["source"]
    )
    assert "kernel plugins" in audit
    # The workflow export claims reproducibility, so it owes the sharper note:
    # the plugins are the reader's, not the session's.
    assert "~/.config/biopb/kernel" in workflow and "NameError" in workflow
