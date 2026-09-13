"""Claude Code CLI backend.

Routes generation through the locally installed ``claude`` CLI (Claude Code)
in print mode, so a user with a Claude Pro/Max plan can run GitView without a
separate ``ANTHROPIC_API_KEY``; usage is billed to the plan.

Design notes:

- The prompt is written to the CLI's stdin, not passed as an argument, so
  long phase summaries never hit the OS argument-length limit.
- A ``system`` message is folded into the user turn. Recent CLIs (>= 2.1)
  still layer their own coding-agent context over ``--system-prompt``, so the
  flag is not a reliable sole authority; an explicit preamble is.
- The invocation mirrors Graphify's proven one (``-p --output-format json
  --no-session-persistence``, prompt on stdin, current working directory kept):
  a fresh temporary cwd can trip Claude Code's workspace-trust check, which
  print mode cannot answer, and an empty ``--tools`` list is not needed for a
  text completion. Session-persistence is off so no transcript is written.
- ``max_tokens`` and ``temperature`` have no CLI equivalent and are ignored.
"""

import json
import os
import shutil
import subprocess
from typing import List, Optional

from .base import BaseLLMBackend, LLMMessage, LLMResponse

DEFAULT_MODEL = "sonnet"
#: Environment variables that mark a running Claude Code session; a nested
#: print-mode call must not inherit them.
_SESSION_ENV = ("CLAUDECODE", "CLAUDE_CODE_ENTRYPOINT", "CLAUDE_CODE_SESSION_ID",
                "CLAUDE_CODE_CHILD_SESSION", "CLAUDE_CODE_MESSAGING_SOCKET",
                "CLAUDE_CODE_MESSAGING_TOKEN")


def claude_cli_available() -> bool:
    """True when a ``claude`` executable is on PATH."""
    return shutil.which("claude") is not None


def fold_messages(messages: List[LLMMessage]) -> str:
    """Flatten a message list into one prompt for a single print-mode turn."""
    system = [m.content for m in messages if m.role == "system"]
    turns = [m for m in messages if m.role != "system"]
    parts: List[str] = []
    if system:
        parts.append("System instructions:\n" + "\n\n".join(system) + "\n\n---\n")
    if len(turns) == 1:
        parts.append(turns[0].content)
    else:
        for m in turns:
            parts.append(f"[{m.role}]\n{m.content}")
    return "\n".join(parts)


class ClaudeCLIBackend(BaseLLMBackend):
    """Generate through ``claude -p --output-format json``."""

    def __init__(self, model: str = DEFAULT_MODEL, temperature: float = 0.7,
                 executable: str = "claude", timeout: int = 600, **kwargs):
        super().__init__(model or DEFAULT_MODEL, temperature, **kwargs)
        self.executable = executable
        self.timeout = timeout

    def generate(self, messages: List[LLMMessage], max_tokens: int = 2000,
                 **kwargs) -> LLMResponse:
        exe = shutil.which(self.executable)
        if not exe:
            raise RuntimeError(
                f"'{self.executable}' is not on PATH; install Claude Code and run `claude /login`, "
                "or choose another backend")
        prompt = fold_messages(messages)
        env = {k: v for k, v in os.environ.items() if k not in _SESSION_ENV}
        cmd = [exe, "-p", "--output-format", "json", "--no-session-persistence",
               "--model", self.model]
        proc = subprocess.run(cmd, input=prompt, capture_output=True, text=True,
                              env=env, timeout=self.timeout)
        shown = " ".join(cmd)
        stderr = proc.stderr.strip()[-800:]
        if proc.returncode != 0 and not proc.stdout.strip():
            raise RuntimeError(f"claude CLI failed (exit {proc.returncode}) running `{shown}`:\n{stderr}")
        try:
            data = json.loads(proc.stdout)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"claude CLI returned non-JSON output running `{shown}`:\n"
                f"stdout: {proc.stdout[:300]!r}\nstderr: {stderr}") from exc
        if data.get("is_error"):
            raise RuntimeError(f"claude CLI error running `{shown}`: {data.get('result')}"
                               + (f"\nstderr: {stderr}" if stderr else ""))
        raw_usage = data.get("usage") or {}
        usage: Optional[dict] = None
        if raw_usage:
            prompt_tokens = int(raw_usage.get("input_tokens", 0) or 0) \
                + int(raw_usage.get("cache_read_input_tokens", 0) or 0) \
                + int(raw_usage.get("cache_creation_input_tokens", 0) or 0)
            completion_tokens = int(raw_usage.get("output_tokens", 0) or 0)
            usage = {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                     "total_tokens": prompt_tokens + completion_tokens}
        return LLMResponse(
            content=str(data.get("result", "")),
            model=str((data.get("modelUsage") and next(iter(data["modelUsage"]), None)) or self.model),
            usage=usage,
            metadata={"stop_reason": data.get("stop_reason"),
                      "session_id": data.get("session_id"),
                      "total_cost_usd": data.get("total_cost_usd")},
        )
