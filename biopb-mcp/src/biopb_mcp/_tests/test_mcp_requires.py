"""Unit tests for `biopb_mcp.mcp._requires` (no kernel/display needed)."""


class TestInstallTargetIsDecidedByTheEnv:
    """`## Versions` has to answer "install where, how, and does it last?".

    The agent can read `sys.executable`; what it cannot see is that biopb's own
    deployment is a uv tool env the next upgrade rebuilds.
    """

    def _lines(self, tmp_path, *, receipt, has_pip):
        from biopb_mcp.mcp import _requires

        if receipt:
            (tmp_path / "uv-receipt.toml").write_text("[tool]\n", encoding="utf-8")
        return "\n".join(
            _requires.versions_status_lines(
                prefix=tmp_path,
                executable="/env/bin/python",
                has_pip=has_pip,
                version="9.9.9",
            )
        )

    def test_a_plain_env_gets_pip_and_no_warning(self, tmp_path):
        report = self._lines(tmp_path, receipt=False, has_pip=True)
        assert "biopb-mcp: 9.9.9" in report
        assert "/env/bin/python -m pip install <pkg>" in report
        assert "uv-managed" not in report  # the user's env; theirs to keep

    def test_no_pip_falls_back_to_uv(self, tmp_path):
        report = self._lines(tmp_path, receipt=False, has_pip=False)
        assert "uv pip install --python /env/bin/python" in report

    def test_a_uv_tool_env_warns_even_though_it_has_pip(self, tmp_path):
        # The real deployment carries pip transitively, so a pip probe would take
        # the `-m pip` branch and say nothing about the rebuild -- the receipt is
        # what identifies the env, not the absence of pip.
        report = self._lines(tmp_path, receipt=True, has_pip=True)
        assert "uv pip install --python /env/bin/python" in report
        assert "-m pip install" not in report
        assert "extra-packages.txt" in report  # the durable half of the fix
