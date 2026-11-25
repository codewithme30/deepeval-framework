"""Model wrappers.

1. get_judge_model(): the LLM that SCORES answers. Name comes from config (`judge_model`).
2. LiveChatModel: a DeepEvalBaseLLM wrapper around the app's live OpenAI model. DeepEval's
   benchmarks (MMLU, TruthfulQA, ...) need the model under test in this shape.
"""
from __future__ import annotations

import json
from typing import Any

from deepeval.models import DeepEvalBaseLLM, GPTModel

from framework.config_loader import Config


def get_judge_model(config: Config) -> GPTModel:
    return GPTModel(model=config.judge_model, temperature=config.get("judge_temperature", 0))


class LiveChatModel(DeepEvalBaseLLM):
    """Wraps the live app model so benchmarks can call `generate(prompt, schema=...)`."""

    def __init__(self, model_name: str):
        self.model_name = model_name  # must be set before super().__init__ calls load_model()
        super().__init__(model_name)

    def load_model(self):
        from app.llm import get_client

        return get_client()

    def generate(self, prompt: str, schema: Any = None) -> Any:
        kwargs: dict[str, Any] = {}
        if schema is not None:
            prompt = f"{prompt}\n\nRespond ONLY with JSON matching this schema:\n{json.dumps(schema.model_json_schema())}"
            kwargs["response_format"] = {"type": "json_object"}
        response = self.model.chat.completions.create(
            model=self.model_name, messages=[{"role": "user", "content": prompt}], temperature=0, **kwargs
        )
        text = response.choices[0].message.content or ""
        if schema is None:
            return text
        try:
            return schema.model_validate_json(text)
        except Exception as exc:  # benchmarks catch TypeError and retry without a schema
            raise TypeError(f"Model did not return valid {schema.__name__} JSON") from exc

    async def a_generate(self, prompt: str, schema: Any = None) -> Any:
        return self.generate(prompt, schema)

    def get_model_name(self) -> str:
        return self.model_name


class FixedAnswerModel(DeepEvalBaseLLM):
    """A 'model' that always gives the same multiple-choice answer. Used for a designed-to-fail
    benchmark demo: the answer is never right, so the score is 0 without spending any tokens."""

    def __init__(self, answer: str = "Z"):
        self.answer = answer
        super().__init__("fixed-answer")

    def load_model(self):
        return self

    def generate(self, prompt: str, schema: Any = None) -> Any:
        return schema(answer=self.answer) if schema is not None else self.answer

    async def a_generate(self, prompt: str, schema: Any = None) -> Any:
        return self.generate(prompt, schema)

    def get_model_name(self) -> str:
        return "fixed-answer"
