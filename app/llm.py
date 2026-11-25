"""Thin OpenAI helper used by the app in LIVE mode only (stub mode never imports a network client)."""
from __future__ import annotations

from functools import lru_cache
from typing import Any


@lru_cache(maxsize=1)
def get_client():
    """Create the OpenAI client lazily so stub mode works with no OpenAI key at all."""
    from openai import OpenAI

    return OpenAI()


def chat(messages: list[dict[str, Any]], model: str, **kwargs: Any):
    """One chat-completion call. Returns the first choice's message object."""
    response = get_client().chat.completions.create(model=model, messages=messages, **kwargs)
    return response.choices[0].message
