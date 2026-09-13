"""`gitview analyze` must reject a missing API key before doing any work."""

from pathlib import Path

from click.testing import CliRunner

from gitview.cli import cli
from tests.test_graph import synthetic_repo  # noqa: F401


def test_missing_key_fails_before_extraction(synthetic_repo, tmp_path, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    out = tmp_path / "out"
    result = CliRunner().invoke(cli, ["analyze", "-r", str(synthetic_repo), "-o", str(out),
                                      "--backend", "anthropic"])
    assert result.exit_code != 0
    assert "API key required" in result.output
    assert not (out / "repo_history.jsonl").exists()
