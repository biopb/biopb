"""``view_figure`` and the note that names it."""

from unittest.mock import MagicMock

import pytest

from biopb_mcp._tests.conftest import call_tool as _tool
from biopb_mcp.mcp import _app, _kernel_rpc, _scratch, _server
from biopb_mcp.mcp._job_log import JobLog


def test_a_result_names_its_figures_and_the_tool():
    note = _kernel_rpc._format_execute_result(
        {"status": "ok", "stdout": "n=3\n", "job_id": "job-7", "figures": 2}
    )
    assert "2 figures" in note and "view_figure('job-7')" in note
    plain = _kernel_rpc._format_execute_result({"status": "ok", "stdout": "n=3\n"})
    assert "figure" not in plain


@pytest.fixture
def host(monkeypatch):
    host = MagicMock()
    host.jobs = JobLog(host_session="host")
    monkeypatch.setattr(_app, "_require_kernel_host", lambda: (host, None))
    return host


def _job_with_figure(log, request="c1"):
    log.on_iopub(
        {
            "header": {"msg_type": "execute_input"},
            "parent_header": {"msg_id": request, "session": "client"},
            "content": {"code": "plot()"},
        }
    )
    log.on_iopub(
        {
            "header": {"msg_type": "display_data"},
            "parent_header": {"msg_id": request},
            "content": {"data": {"image/png": "UE5H"}},
        }
    )
    (rec,) = log.export()
    return rec["job_id"]


def test_view_figure_returns_the_png_as_an_image(host):
    job_id = _job_with_figure(host.jobs)
    (block,) = _tool(_server.view_figure, job_id)
    assert block.type == "image" and block.mimeType == "image/png"
    assert block.data == "UE5H"


def test_view_figure_says_why_there_is_none(host):
    job_id = _job_with_figure(host.jobs)
    (block,) = _tool(_server.view_figure, job_id, index=5)
    assert "1 figure" in block.text
    (block,) = _tool(_server.view_figure, "job-nope")
    assert "no such job" in block.text


def test_a_verification_figure_is_asked_for_by_cell(host, monkeypatch):
    monkeypatch.setattr(_scratch, "figure", lambda j, c, i=0: ("UE5H", 1))
    (block,) = _tool(_server.view_figure, "verify-1")
    assert "pass cell=N" in block.text
    (block,) = _tool(_server.view_figure, "verify-1", cell=2)
    assert block.type == "image"


def test_the_verification_ledger_names_a_cells_figures():
    text = _server._format_verification(
        {"status": "ok", "cells": [{"status": "ok", "elapsed": 1, "figures": 2}]},
        "verify-1",
        "/x.ipynb",
    )
    assert "2 figures (view_figure('verify-1', cell=1))" in text
