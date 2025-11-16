"""OpenAI backend."""

from typing import List, Optional

from openai import OpenAI

from .base import BaseLLMBackend, LLMMessage, LLMResponse


class OpenAIBackend(BaseLLMBackend):
    """OpenAI GPT backend."""

    def __init__(self, model: str, api_key: str, temperature: float = 0.7, **kwargs):
        """
        Initialize OpenAI backend.

        Args:
            model: OpenAI model identifier (e.g., "gpt-4", "gpt-3.5-turbo")
            api_key: OpenAI API key
            temperature: Temperature for generation
            **kwargs: Additional parameters
        """
        super().__init__(model, temperature, **kwargs)
        self.client = OpenAI(api_key=api_key)

    def generate(self, messages: List[LLMMessage], max_tokens: int = 2000,
                **kwargs) -> LLMResponse:
        """
        Generate completion using OpenAI.

        Args:
            messages: List of messages
            max_tokens: Maximum tokens to generate
            **kwargs: Additional generation parameters

        Returns:
            LLMResponse object
        """
        # Convert messages to OpenAI format
        openai_messages = [
            {"role": msg.role, "content": msg.content}
            for msg in messages
        ]

        # Call OpenAI API
        response = self.client.chat.completions.create(
            model=self.model,
            messages=openai_messages,
            max_tokens=max_tokens,
            temperature=kwargs.get('temperature', self.temperature)
        )

        # Extract usage info
        usage = None
        if response.usage:
            usage = {
                'prompt_tokens': response.usage.prompt_tokens,
                'completion_tokens': response.usage.completion_tokens,
                'total_tokens': response.usage.total_tokens
            }

        # Return standardized response
        return LLMResponse(
            content=response.choices[0].message.content,
            model=response.model,
            usage=usage,
            metadata={
                'finish_reason': response.choices[0].finish_reason,
                'created': response.created
            }
        )
