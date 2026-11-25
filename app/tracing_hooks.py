"""Glue between the mock app and DeepEval tracing.

The app's key functions are decorated with `@observe` (retriever / llm / tool / agent spans).
To evaluate a single component, a test registers metrics for that span here; the app then
attaches them to the live span together with a test case describing what the span did.

Why a registry instead of `@observe(metrics=[...])`? The metrics must come from
`framework.metric_factory` (config-driven), which does not exist yet when the app module is
imported. Registering at run time keeps thresholds out of the app code.
"""
from __future__ import annotations

from typing import Any

from deepeval.test_case import LLMTestCase, ToolCall
from deepeval.tracing import update_current_span

_SPAN_METRICS: dict[str, list[Any]] = {}


def set_span_metrics(**metrics_by_span: list[Any]) -> None:
    """Register metrics per span key: retriever, generator, planner, tool, agent."""
    _SPAN_METRICS.update(metrics_by_span)


def clear_span_metrics() -> None:
    _SPAN_METRICS.clear()


def attach(span_key: str, **test_case_fields: Any) -> None:
    """Describe the active span as an LLMTestCase and attach any metrics registered for it."""
    metrics = _SPAN_METRICS.get(span_key) or None
    if metrics is None and not test_case_fields:
        return
    update_current_span(test_case=LLMTestCase(**test_case_fields), metrics=metrics)


def to_tool_calls(records: list[Any]) -> list[ToolCall]:
    """Convert the agent's ToolRecord objects into DeepEval ToolCall objects."""
    return [
        ToolCall(name=r.name, input_parameters=dict(r.arguments), output=r.output)
        for r in records
    ]
