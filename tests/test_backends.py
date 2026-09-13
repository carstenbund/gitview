"""Tests for the Claude Code CLI backend and router auto-detection."""

import json
import os
import stat
from pathlib import Path

import pytest

from gitview.backends import LLMMessage
from gitview.backends.claude_cli_backend import ClaudeCLIBackend, fold_messages
from gitview.backends.router import LLMBackend, LLMRouter


def _fake_claude(tmp_path: Path, result: str = "OK", is_error: bool = False) -> Path:
    """A stand-in `claude` that records its argv and stdin and prints a print-mode JSON envelope."""
    log = tmp_path / "calls.json"
    exe = tmp_path / "bin" / "claude"
    exe.parent.mkdir(exist_ok=True)
    payload = json.dumps({"type": "result", "subtype": "success", "is_error": is_error,
                          "result": result, "stop_reason": "end_turn", "session_id": "s",
                          "usage": {"input_tokens": 10, "cache_read_input_tokens": 5, "output_tokens": 3},
                          "modelUsage": {"claude-sonnet-5": {}}})
    exe.write_text(
        "#!/usr/bin/env python3\n"
        "import json, sys\n"
        f"json.dump({{'argv': sys.argv[1:], 'stdin': sys.stdin.read(), 'cwd': __import__('os').getcwd(),\n"
        f"           'env_has_claudecode': 'CLAUDECODE' in __import__('os').environ}}, open({str(log)!r}, 'w'))\n"
        f"print({payload!r})\n")
    exe.chmod(exe.stat().st_mode | stat.S_IEXEC)
    return log


def test_fold_messages_prepends_system_and_labels_multi_turn():
    single = fold_messages([LLMMessage("system", "be terse"), LLMMessage("user", "hi")])
    assert single.startswith("System instructions:\nbe terse") and single.endswith("hi")
    multi = fold_messages([LLMMessage("user", "a"), LLMMessage("assistant", "b"), LLMMessage("user", "c")])
    assert "[user]\na" in multi and "[assistant]\nb" in multi


def test_generate_runs_cli_in_print_mode(tmp_path, monkeypatch):
    log = _fake_claude(tmp_path, result="a summary")
    monkeypatch.setenv("PATH", f"{tmp_path / 'bin'}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("CLAUDECODE", "1")
    resp = ClaudeCLIBackend(model="haiku").generate([LLMMessage("user", "summarise")])
    assert resp.content == "a summary"
    assert resp.usage == {"prompt_tokens": 15, "completion_tokens": 3, "total_tokens": 18}
    assert resp.model == "claude-sonnet-5"
    call = json.loads(log.read_text())
    assert call["stdin"] == "summarise"
    assert "-p" in call["argv"] and "--model" in call["argv"] and "haiku" in call["argv"]
    assert "--no-session-persistence" in call["argv"]
    assert call["cwd"] != os.getcwd()           # neutral cwd: no CLAUDE.md / hooks of the analysed repo
    assert call["env_has_claudecode"] is False  # nested-session markers stripped


def test_generate_surfaces_cli_error(tmp_path, monkeypatch):
    _fake_claude(tmp_path, result="Not logged in", is_error=True)
    monkeypatch.setenv("PATH", f"{tmp_path / 'bin'}{os.pathsep}{os.environ['PATH']}")
    with pytest.raises(RuntimeError, match="Not logged in"):
        ClaudeCLIBackend().generate([LLMMessage("user", "x")])


def test_generate_errors_when_cli_missing(tmp_path, monkeypatch):
    monkeypatch.setenv("PATH", str(tmp_path))
    with pytest.raises(RuntimeError, match="not on PATH"):
        ClaudeCLIBackend().generate([LLMMessage("user", "x")])


def test_router_prefers_keys_then_cli_then_ollama(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("PATH", str(tmp_path))
    assert LLMRouter().backend_type == LLMBackend.OLLAMA
    _fake_claude(tmp_path)
    monkeypatch.setenv("PATH", str(tmp_path / "bin"))
    router = LLMRouter()
    assert router.backend_type == LLMBackend.CLAUDE_CLI and router.model == "sonnet"
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    assert LLMRouter().backend_type == LLMBackend.ANTHROPIC
    assert LLMRouter(backend="claude-cli", model="haiku").model == "haiku"
