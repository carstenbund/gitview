"""Anthropic Claude backend."""

from typing import List, Optional

from anthropic import Anthropic

from .base import BaseLLMBackend, LLMMessage, LLMResponse


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

        # Call Anthropic API
        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=kwargs.get('temperature', self.temperature),
            messages=anthropic_messages
        )

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
