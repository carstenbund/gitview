"""Simplified LLM router for GitView."""

import os
from enum import Enum
from typing import Optional, List

from .base import BaseLLMBackend, LLMMessage
from .anthropic_backend import AnthropicBackend
from .ollama_backend import OllamaBackend
from .openai_backend import OpenAIBackend


class LLMBackend(str, Enum):
    """Supported LLM backends."""

    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    OLLAMA = "ollama"


class LLMRouter:
    """Routes LLM requests to appropriate backend."""

    # Default models for each backend
    DEFAULT_MODELS = {
        LLMBackend.ANTHROPIC: "claude-sonnet-4-5-20250929",
        LLMBackend.OPENAI: "gpt-4",
        LLMBackend.OLLAMA: "llama3",
    }

    def __init__(self, backend: Optional[str] = None, model: Optional[str] = None,
                 api_key: Optional[str] = None, **kwargs):
        """
        Initialize LLM router.

        Args:
            backend: Backend to use ('anthropic', 'openai', 'ollama')
            model: Model identifier (uses defaults if not specified)
            api_key: API key for the backend (if required)
            **kwargs: Additional backend-specific parameters
        """
        # Determine backend
        if backend:
            self.backend_type = LLMBackend(backend.lower())
        else:
            # Auto-detect from environment
            if os.environ.get('ANTHROPIC_API_KEY'):
                self.backend_type = LLMBackend.ANTHROPIC
            elif os.environ.get('OPENAI_API_KEY'):
                self.backend_type = LLMBackend.OPENAI
            else:
                self.backend_type = LLMBackend.OLLAMA

        # Determine model
        self.model = model or self.DEFAULT_MODELS[self.backend_type]

        # Determine API key
        if api_key:
            self.api_key = api_key
        else:
            # Try to get from environment
            if self.backend_type == LLMBackend.ANTHROPIC:
                self.api_key = os.environ.get('ANTHROPIC_API_KEY')
            elif self.backend_type == LLMBackend.OPENAI:
                self.api_key = os.environ.get('OPENAI_API_KEY')
            else:
                self.api_key = None  # Not needed for Ollama

        # Additional parameters
        self.kwargs = kwargs

        # Create backend
        self._backend: Optional[BaseLLMBackend] = None

    def _get_backend(self) -> BaseLLMBackend:
        """Get or create backend instance."""
        if self._backend is None:
            temperature = self.kwargs.get('temperature', 0.7)

            if self.backend_type == LLMBackend.ANTHROPIC:
                if not self.api_key:
                    raise ValueError(
                        "Anthropic API key required. Set ANTHROPIC_API_KEY environment variable "
                        "or pass api_key parameter."
                    )
                self._backend = AnthropicBackend(
                    model=self.model,
                    api_key=self.api_key,
                    temperature=temperature
                )

            elif self.backend_type == LLMBackend.OPENAI:
                if not self.api_key:
                    raise ValueError(
                        "OpenAI API key required. Set OPENAI_API_KEY environment variable "
                        "or pass api_key parameter."
                    )
                self._backend = OpenAIBackend(
                    model=self.model,
                    api_key=self.api_key,
                    temperature=temperature
                )

            elif self.backend_type == LLMBackend.OLLAMA:
                ollama_url = self.kwargs.get('ollama_url', 'http://localhost:11434')
                self._backend = OllamaBackend(
                    model=self.model,
                    api_url=ollama_url,
                    temperature=temperature
                )

        return self._backend

    def generate(self, messages: List[LLMMessage], max_tokens: int = 2000, **kwargs):
        """
        Generate completion.

        Args:
            messages: List of messages
            max_tokens: Maximum tokens to generate
            **kwargs: Additional generation parameters

        Returns:
            LLMResponse object
        """
        backend = self._get_backend()
        return backend.generate(messages, max_tokens, **kwargs)

    def generate_text(self, prompt: str, max_tokens: int = 2000, **kwargs) -> str:
        """
        Generate completion from a simple text prompt.

        Args:
            prompt: Text prompt
            max_tokens: Maximum tokens to generate
            **kwargs: Additional generation parameters

        Returns:
            Generated text
        """
        messages = [LLMMessage(role="user", content=prompt)]
        response = self.generate(messages, max_tokens, **kwargs)
        return response.content

    def __repr__(self) -> str:
        """String representation."""
        return f"LLMRouter(backend={self.backend_type.value}, model={self.model})"


def create_router(backend: Optional[str] = None, model: Optional[str] = None,
                 api_key: Optional[str] = None, **kwargs) -> LLMRouter:
    """
    Create an LLM router.

    Args:
        backend: Backend to use ('anthropic', 'openai', 'ollama')
        model: Model identifier
        api_key: API key for the backend
        **kwargs: Additional backend parameters

    Returns:
        LLMRouter instance
    """
    return LLMRouter(backend=backend, model=model, api_key=api_key, **kwargs)
