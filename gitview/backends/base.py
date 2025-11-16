"""Base LLM backend interface."""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from dataclasses import dataclass


@dataclass
class LLMMessage:
    """Standard message format for LLM APIs."""

    role: str  # "user", "assistant", "system"
    content: str


@dataclass
class LLMResponse:
    """Standard response format from LLM APIs."""

    content: str
    model: str
    usage: Optional[Dict[str, int]] = None  # Token usage stats
    metadata: Optional[Dict[str, Any]] = None  # Additional metadata


class BaseLLMBackend(ABC):
    """Base class for LLM backends."""

    def __init__(self, model: str, temperature: float = 0.7, **kwargs):
        """
        Initialize backend.

        Args:
            model: Model identifier
            temperature: Temperature for generation (0.0 - 1.0)
            **kwargs: Additional backend-specific parameters
        """
        self.model = model
        self.temperature = temperature
        self.kwargs = kwargs

    @abstractmethod
    def generate(self, messages: List[LLMMessage], max_tokens: int = 2000,
                **kwargs) -> LLMResponse:
        """
        Generate completion from messages.

        Args:
            messages: List of messages
            max_tokens: Maximum tokens to generate
            **kwargs: Additional generation parameters

        Returns:
            LLMResponse object
        """
        pass

    def __repr__(self) -> str:
        """String representation."""
        return f"{self.__class__.__name__}(model={self.model}, temperature={self.temperature})"
