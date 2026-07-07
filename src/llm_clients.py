"""Abstract client for querying LLMs and a Gemini implementation."""

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

from google import genai
from google.genai import types

from src.config import get_api_key


@dataclass
class LLMResponse:
    prompt_id: str
    model_name: str
    prompt_text: str
    response_text: str
    success: bool
    error_message: Optional[str] = None
    latency_seconds: Optional[float] = None


class LLMClient(ABC):
    def __init__(self, model_name: str):
        self.model_name = model_name

    @abstractmethod
    def _call_model(self, prompt_text: str) -> str:
        ...

    def query(self, prompt_id: str, prompt_text: str) -> LLMResponse:
        # Time the model call and catch errors, so a single failure does not stop the whole experiment.
        start_time = time.time()
        try:
            response_text = self._call_model(prompt_text)
            latency = time.time() - start_time
            return LLMResponse(
                prompt_id=prompt_id,
                model_name=self.model_name,
                prompt_text=prompt_text,
                response_text=response_text,
                success=True,
                latency_seconds=latency,
            )
        except Exception as e:
            latency = time.time() - start_time
            return LLMResponse(
                prompt_id=prompt_id,
                model_name=self.model_name,
                prompt_text=prompt_text,
                response_text="",
                success=False,
                error_message=str(e),
                latency_seconds=latency,
            )


class GeminiClient(LLMClient):
    def __init__(self, model_name: str = "gemini-2.5-flash"):
        super().__init__(model_name)
        api_key = get_api_key("google")
        self.client = genai.Client(api_key=api_key)

    def _call_model(self, prompt_text: str) -> str:
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt_text,
            config=types.GenerateContentConfig(
                temperature=0.7,
                max_output_tokens=2048,
            ),
        )
        return response.text