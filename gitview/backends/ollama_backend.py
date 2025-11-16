"""Ollama backend for local LLM inference."""

import json
from typing import List, Optional

import requests

from .base import BaseLLMBackend, LLMMessage, LLMResponse


class OllamaBackend(BaseLLMBackend):
    """Ollama local LLM backend."""

    def __init__(self, model: str, api_url: str = "http://localhost:11434",
                 temperature: float = 0.7, **kwargs):
        """
        Initialize Ollama backend.

        Args:
            model: Ollama model identifier (e.g., "llama3", "mistral")
            api_url: Ollama API URL
            temperature: Temperature for generation
            **kwargs: Additional parameters
        """
        super().__init__(model, temperature, **kwargs)
        self.api_url = api_url.rstrip('/')

    def generate(self, messages: List[LLMMessage], max_tokens: int = 2000,
                **kwargs) -> LLMResponse:
        """
        Generate completion using Ollama.

        Args:
            messages: List of messages
            max_tokens: Maximum tokens to generate
            **kwargs: Additional generation parameters

        Returns:
            LLMResponse object
        """
        # Convert messages to Ollama format
        ollama_messages = [
            {"role": msg.role, "content": msg.content}
            for msg in messages
        ]

        # Prepare request
        payload = {
            "model": self.model,
            "messages": ollama_messages,
            "stream": False,
            "options": {
                "temperature": kwargs.get('temperature', self.temperature),
                "num_predict": max_tokens,
            }
        }

        # Call Ollama API
        try:
            response = requests.post(
                f"{self.api_url}/api/chat",
                json=payload,
                timeout=120  # 2 minute timeout for local inference
            )
            response.raise_for_status()
            data = response.json()

            # Extract content
            content = data.get('message', {}).get('content', '')

            # Extract usage if available
            usage = None
            if 'prompt_eval_count' in data and 'eval_count' in data:
                usage = {
                    'prompt_tokens': data.get('prompt_eval_count', 0),
                    'completion_tokens': data.get('eval_count', 0),
                    'total_tokens': data.get('prompt_eval_count', 0) + data.get('eval_count', 0)
                }

            # Return standardized response
            return LLMResponse(
                content=content,
                model=data.get('model', self.model),
                usage=usage,
                metadata={
                    'done': data.get('done', False),
                    'total_duration': data.get('total_duration'),
                }
            )

        except requests.exceptions.RequestException as e:
            raise RuntimeError(
                f"Ollama API error: {e}\n"
                f"Make sure Ollama is running at {self.api_url}\n"
                f"Start it with: ollama serve"
            ) from e
