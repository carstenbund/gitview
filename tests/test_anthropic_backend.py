"""Sampling parameters must not be sent to models that reject them."""

from types import SimpleNamespace

import pytest

from gitview.backends import LLMMessage
from gitview.backends.anthropic_backend import (
    AnthropicBackend, accepts_temperature, response_text, thinking_request_fields,
)


class _Messages:
    def __init__(self, reject_temperature):
        self.calls = []
        self.reject = reject_temperature

    def create(self, **kw):
        self.calls.append(kw)
        if self.reject and 'temperature' in kw:
            try:
                import httpx                # anthropic 0.x
            except ImportError:             # anthropic 1.x is built on httpx2
                import httpx2 as httpx
            from anthropic import BadRequestError
            resp = httpx.Response(400, request=httpx.Request("POST", "https://api.anthropic.com/v1/messages"))
            raise BadRequestError("`temperature` is deprecated for this model.", response=resp, body=None)
        return SimpleNamespace(content=[SimpleNamespace(text="ok")], model=kw['model'],
                               usage=SimpleNamespace(input_tokens=1, output_tokens=1),
                               stop_reason="end_turn")


def _backend(model, reject):
    b = AnthropicBackend(model=model, api_key="k")
    b.client = SimpleNamespace(messages=_Messages(reject))
    return b


@pytest.mark.parametrize("model,expected", [
    ("claude-sonnet-5", False), ("claude-opus-5", False), ("claude-opus-4-8", False),
    ("claude-sonnet-4-6", True), ("claude-haiku-4-5", True), ("claude-sonnet-4-5-20250929", True),
])
def test_accepts_temperature_by_model(model, expected):
    assert accepts_temperature(model) is expected


def test_known_model_never_sends_temperature():
    b = _backend("claude-sonnet-5", reject=True)
    assert b.generate([LLMMessage("user", "hi")]).content == "ok"
    assert 'temperature' not in b.client.messages.calls[0]


def test_older_model_sends_temperature():
    b = _backend("claude-sonnet-4-6", reject=False)
    b.generate([LLMMessage("user", "hi")])
    assert b.client.messages.calls[0]['temperature'] == 0.7


def test_unanticipated_rejection_retries_without_and_remembers():
    b = _backend("claude-future-9", reject=True)
    assert b.generate([LLMMessage("user", "hi")]).content == "ok"
    assert [('temperature' in c) for c in b.client.messages.calls] == [True, False]
    b.generate([LLMMessage("user", "again")])
    assert 'temperature' not in b.client.messages.calls[-1]


def test_text_is_collected_from_text_blocks_only():
    blocks = [SimpleNamespace(type="thinking", thinking="..."),
              SimpleNamespace(type="text", text="Hello "),
              SimpleNamespace(type="text", text="world")]
    assert response_text(blocks) == "Hello world"
    assert response_text([SimpleNamespace(type="thinking", thinking="x")]) == ""


@pytest.mark.parametrize("model,expected", [
    ("claude-sonnet-5", {"thinking": {"type": "disabled"}}),
    ("claude-opus-5", {"thinking": {"type": "disabled"}}),
    ("claude-fable-5-1", {"max_tokens": 8000}),
    ("claude-sonnet-4-5-20250929", {}),
    ("claude-haiku-4-5", {}),
])
def test_thinking_fields_by_model(model, expected):
    assert thinking_request_fields(model, 1200) == expected


def test_generate_survives_a_leading_thinking_block():
    class _Thinking(_Messages):
        def create(self, **kw):
            self.calls.append(kw)
            return SimpleNamespace(content=[SimpleNamespace(type="thinking", thinking="hmm"),
                                            SimpleNamespace(type="text", text="summary")],
                                   model=kw["model"],
                                   usage=SimpleNamespace(input_tokens=1, output_tokens=1),
                                   stop_reason="end_turn")
    b = AnthropicBackend(model="claude-sonnet-5", api_key="k")
    b.client = SimpleNamespace(messages=_Thinking(False))
    out = b.generate([LLMMessage("user", "hi")], max_tokens=1200)
    assert out.content == "summary"
    assert b.client.messages.calls[0]["thinking"] == {"type": "disabled"}
    assert "temperature" not in b.client.messages.calls[0]
