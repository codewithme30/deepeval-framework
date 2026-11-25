"""
FEATURE: Standard LLM benchmarks
================================

WHAT IT IS
    Benchmarks are public exam papers for language models. MMLU asks multiple-choice questions
    across school and professional subjects; TruthfulQA checks whether a model repeats popular
    myths. DeepEval downloads the questions, quizzes YOUR model, and reports accuracy.

WHY IT MATTERS
    Everything else in this suite tests our APP. Benchmarks test the MODEL underneath it. They
    answer "did swapping to a cheaper or newer model make it dumber?" before you ship the change.
    They are NOT a substitute for the app-specific tests in files 01-09.

HOW THE MODEL IS PLUGGED IN
    Benchmarks need the model in DeepEval's own shape. `framework/model_wrappers.py` wraps the
    app's live OpenAI model (name from config `live_model`) as a `DeepEvalBaseLLM`.

COST AND SKIPPING
    Benchmarks make many model calls, so they only run when BOTH are true:
        * APP_MODE=live   (stub mode has no real model to measure)
        * the --run-benchmarks flag is passed (or RUN_BENCHMARKS=1 is set)
    Otherwise the quiz tests skip with a clear message. Size is capped by config/settings.yaml
    (benchmark_* limits: tasks, shots, problems per task). CI runs this file weekly.

HOW TO READ PASS / FAIL
    `overall_score` is accuracy from 0 to 1. The test passes when it reaches the threshold in
    thresholds.yaml (benchmark_mmlu, benchmark_truthful_qa). With only a handful of questions the
    score is coarse (each question is worth 1/N), so the thresholds are deliberately modest.
    The xfail demo quizzes a "model" that always answers Z, scoring zero.

HOW TO RUN
    pytest -m benchmark --app-mode live --run-benchmarks
    APP_MODE=live RUN_BENCHMARKS=1 deepeval test run tests/test_10_benchmarks.py
    make benchmarks
"""
import pytest
from deepeval.benchmarks import MMLU, TruthfulQA
from deepeval.benchmarks.modes import TruthfulQAMode
from deepeval.benchmarks.tasks import MMLUTask, TruthfulQATask

from app import llm
from framework.model_wrappers import FixedAnswerModel, LiveChatModel

pytestmark = [pytest.mark.benchmark, pytest.mark.slow]


@pytest.fixture()
def benchmark_gate(run_benchmarks, config):
    """Skip unless we are in live mode AND the user opted in with --run-benchmarks."""
    if config.app_mode != "live":
        pytest.skip("Benchmarks measure a real model: run with APP_MODE=live (or --app-mode live).")
    if not run_benchmarks:
        pytest.skip("Benchmarks are expensive: pass --run-benchmarks (or set RUN_BENCHMARKS=1) to run them.")


@pytest.fixture()
def live_model(benchmark_gate, config) -> LiveChatModel:
    return LiveChatModel(config.live_model)


def _mmlu(config, n_shots=None) -> MMLU:
    return MMLU(
        tasks=[MMLUTask[name] for name in config.require("limits.benchmark_mmlu_tasks")],
        n_shots=n_shots or int(config.require("limits.benchmark_mmlu_n_shots")),
        n_problems_per_task=int(config.require("limits.benchmark_n_problems_per_task")),
    )


# ======================================================================================
# 0. Set-up checks that cost nothing (run in every mode)
# ======================================================================================
@pytest.mark.smoke
@pytest.mark.no_judge
def test_benchmark_sizes_come_from_config_and_stay_small(config):
    assert [MMLUTask[n] for n in config.require("limits.benchmark_mmlu_tasks")]
    assert [TruthfulQATask[n] for n in config.require("limits.benchmark_truthfulqa_tasks")]
    assert 1 <= config.require("limits.benchmark_mmlu_n_shots") <= 5
    assert config.require("limits.benchmark_n_problems_per_task") <= 20, "keep benchmark runs cheap"


@pytest.mark.smoke
@pytest.mark.no_judge
def test_live_model_wrapper_follows_the_deepeval_model_interface(config, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-not-a-real-key")  # building the client makes no network call
    llm.get_client.cache_clear()
    try:
        model = LiveChatModel(config.live_model)
        assert model.get_model_name() == config.live_model
        assert callable(model.generate) and callable(model.a_generate)
    finally:
        llm.get_client.cache_clear()


# ======================================================================================
# 1. The quizzes (live mode + --run-benchmarks only)
# ======================================================================================
def test_model_meets_the_mmlu_accuracy_threshold(live_model, config, thresholds):
    benchmark = _mmlu(config)
    benchmark.evaluate(model=live_model)
    assert benchmark.overall_score >= thresholds["benchmark_mmlu"], (
        f"MMLU accuracy {benchmark.overall_score:.0%} is below {thresholds['benchmark_mmlu']:.0%}"
    )


def test_model_meets_the_truthfulqa_accuracy_threshold(live_model, config, thresholds):
    benchmark = TruthfulQA(
        tasks=[TruthfulQATask[n] for n in config.require("limits.benchmark_truthfulqa_tasks")],
        mode=TruthfulQAMode.MC1,
        n_problems_per_task=int(config.require("limits.benchmark_n_problems_per_task")),
    )
    benchmark.evaluate(model=live_model)
    assert benchmark.overall_score >= thresholds["benchmark_truthful_qa"], (
        f"TruthfulQA accuracy {benchmark.overall_score:.0%} is below {thresholds['benchmark_truthful_qa']:.0%}"
    )


# ======================================================================================
# 2. DESIGNED TO FAIL
# ======================================================================================
@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="demonstrates a failing benchmark: a model that always answers 'Z' scores 0% on MMLU",
)
def test_model_that_always_answers_z_fails_the_mmlu_threshold(benchmark_gate, config, thresholds):
    benchmark = _mmlu(config)
    benchmark.evaluate(model=FixedAnswerModel("Z"))
    assert benchmark.overall_score >= thresholds["benchmark_mmlu"], (
        f"MMLU accuracy {benchmark.overall_score:.0%} is below {thresholds['benchmark_mmlu']:.0%}"
    )
