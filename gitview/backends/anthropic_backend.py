"""Anthropic Claude backend."""

from typing import List, Optional

from anthropic import Anthropic, BadRequestError

from .base import BaseLLMBackend, LLMMessage, LLMResponse

#: Model families that reject the sampling parameters (`temperature`, `top_p`,
#: `top_k`) with a 400 "deprecated for this model". Everything from Opus 4.7 /
#: Sonnet 5 onwards; Opus 4.6, Sonnet 4.6, Haiku 4.5 and older still accept them.
_NO_SAMPLING_PREFIXES = (
    "claude-opus-5", "claude-opus-4-7", "claude-opus-4-8",
    "claude-sonnet-5", "claude-fable", "claude-mythos",
)


def accepts_temperature(model: str) -> bool:
    return not str(model).startswith(_NO_SAMPLING_PREFIXES)


class AnthropicBackend(BaseLLMBackend):
    """Anthropic Claude backend."""

    def __init__(self, model: str, api_key: str, temperature: float = 0.7, **kwargs):
        """
        Initialize Anthropic backend.

        Args:
            model: Claude model identifier
            api_key: Anthropic API key
            temperature: Temperature for generation
            **kwargs: Additional parameters
        """
        super().__init__(model, temperature, **kwargs)
        self.client = Anthropic(api_key=api_key)
        self._send_temperature = accepts_temperature(model)

    def generate(self, messages: List[LLMMessage], max_tokens: int = 2000,
                **kwargs) -> LLMResponse:
        """
        Generate completion using Claude.

        Args:
            messages: List of messages
            max_tokens: Maximum tokens to generate
            **kwargs: Additional generation parameters

        Returns:
            LLMResponse object
        """
        # Convert messages to Anthropic format
        anthropic_messages = [
            {"role": msg.role, "content": msg.content}
            for msg in messages
        ]

        request = dict(model=self.model, max_tokens=max_tokens, messages=anthropic_messages)
        if self._send_temperature:
            request['temperature'] = kwargs.get('temperature', self.temperature)

        # Call Anthropic API. If a model we did not anticipate rejects the
        # sampling parameter, drop it and retry once, then remember.
        try:
            response = self.client.messages.create(**request)
        except BadRequestError as exc:
            if 'temperature' in request and 'temperature' in str(exc):
                self._send_temperature = False
                request.pop('temperature')
                response = self.client.messages.create(**request)
            else:
                raise

        # Extract usage info
        usage = {
            'prompt_tokens': response.usage.input_tokens,
            'completion_tokens': response.usage.output_tokens,
            'total_tokens': response.usage.input_tokens + response.usage.output_tokens
        }

        # Return standardized response
        return LLMResponse(
            content=response.content[0].text,
            model=response.model,
            usage=usage,
            metadata={'stop_reason': response.stop_reason}
        )
