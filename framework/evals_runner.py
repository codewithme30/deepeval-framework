"""Helpers shared by tests that run many cases at once.

* eval_configs(): DeepEval async / cache / error settings, taken from config/settings.yaml.
* run_evals_iterator(): drives `dataset.evals_iterator(...)` and RETURNS its EvaluationResult
  (a plain `for` loop discards it, so pytest would have nothing to assert on).
* assert_all_passed(): turn a failed EvaluationResult into a readable pytest failure.
* run_trace_evals(): ONE entry point for "run the app under tracing, then judge the trace/spans"
  that works both under plain `pytest` and under `deepeval test run` (see its docstring for why
  two code paths are needed).
"""
from __future__ import annotations

from typing import Any, Callable

from deepeval import assert_test
from deepeval.evaluate.configs import AsyncConfig, CacheConfig, DisplayConfig, ErrorConfig
from deepeval.evaluate.types import EvaluationResult
from deepeval.utils import get_is_running_deepeval

from framework.config_loader import Config


def eval_configs(config: Config) -> dict[str, Any]:
    """Keyword arguments accepted by both `evaluate()` and `dataset.evals_iterator()`."""
    ev = config.get("evaluation", {})
    return {
        "async_config": AsyncConfig(
            run_async=bool(ev.get("async_mode", True)),
            max_concurrent=int(ev.get("max_concurrent", 5)),
            throttle_value=float(ev.get("throttle_value", 0)),
        ),
        "cache_config": CacheConfig(use_cache=bool(ev.get("use_cache", False)), write_cache=bool(ev.get("write_cache", True))),
        "error_config": ErrorConfig(ignore_errors=bool(ev.get("ignore_errors", False))),
        "display_config": DisplayConfig(show_indicator=False, print_results=False, inspect_after_run=False),
    }


def run_evals_iterator(dataset, app_fn: Callable[[str], Any], metrics=None, **kwargs: Any) -> EvaluationResult:
    """Run `app_fn(golden.input)` for every golden, evaluate, and return the EvaluationResult."""
    iterator = dataset.evals_iterator(metrics=metrics, **kwargs)
    while True:
        try:
            golden = next(iterator)
        except StopIteration as stop:
            return stop.value
        app_fn(golden.input)


def failed_metrics(result: EvaluationResult) -> list[str]:
    lines = []
    for test_result in result.test_results:
        for md in test_result.metrics_data or []:
            if not md.success:
                lines.append(f"[{test_result.name}] {md.name}: score={md.score} threshold={md.threshold} reason={md.reason}")
    return lines


def assert_all_passed(result: EvaluationResult) -> None:
    assert result is not None and result.test_results, "No test results were produced"
    evaluated = sum(len(t.metrics_data or []) for t in result.test_results)
    assert evaluated > 0, "No metric was evaluated (were metrics attached to the trace or to a span?)"
    failures = failed_metrics(result)
    assert not failures, "Metrics below threshold:\n" + "\n".join(failures)


def run_trace_evals(goldens, app_fn: Callable[[str], Any], metrics=None, **eval_kwargs: Any) -> None:
    """Run `app_fn` on every golden under tracing and assert every attached metric passes.

    `metrics` are TRACE-level metrics (judge the whole run). Metrics registered on individual
    spans via app/tracing_hooks.py are judged too. Raises AssertionError on any failure.

    Why two paths? DeepEval has two execution models for traces and they do not mix:
      * plain `pytest`          -> `dataset.evals_iterator(...)` drives the loop.
      * `deepeval test run`     -> a plugin wraps every test in its own trace, and the metrics
                                   are applied with `assert_test(golden=..., metrics=...)`.
                                   (evals_iterator yields NO results inside that wrapper.)
    Tests call this one function and work under both, so CI (which uses `deepeval test run`) and
    a developer's plain `pytest` behave the same.
    """
    from deepeval.dataset import EvaluationDataset

    goldens = list(goldens.goldens if isinstance(goldens, EvaluationDataset) else goldens)
    if get_is_running_deepeval():
        for golden in goldens:
            app_fn(golden.input)
            assert_test(golden=golden, metrics=metrics)
    else:
        result = run_evals_iterator(EvaluationDataset(goldens=goldens), app_fn, metrics=metrics, **eval_kwargs)
        assert_all_passed(result)
