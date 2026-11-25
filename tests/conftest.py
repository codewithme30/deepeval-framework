"""Shared fixtures, command-line options and skip rules for every test file.

New to pytest? A *fixture* is a reusable piece of setup. A test asks for one simply by
naming it as an argument, e.g. `def test_x(rag_app, metric_factory): ...`.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

# Make `app` and `framework` importable no matter where pytest is started from.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# We do not send usage data to DeepEval / DeepTeam telemetry from this demo.
os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")
os.environ.setdefault("DEEPTEAM_TELEMETRY_OPT_OUT", "YES")

from deepeval.dataset import EvaluationDataset  # noqa: E402

from app.agent import ClaimsAgent  # noqa: E402
from app.chatbot import ClaimsChatbot  # noqa: E402
from app.rag import RAGPipeline  # noqa: E402
from framework.config_loader import load_config, load_thresholds, openai_key_present  # noqa: E402
from framework.evals_runner import eval_configs  # noqa: E402
from framework.metric_factory import MetricFactory  # noqa: E402

NO_KEY_REASON = (
    "OPENAI_API_KEY is not set. This test uses an LLM judge (always a real OpenAI model, even in "
    "stub mode). Copy .env.example to .env and add your key."
)


# --------------------------------------------------------------------------------------
# Command-line options
# --------------------------------------------------------------------------------------
def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("deepeval-framework")
    group.addoption("--app-mode", choices=["stub", "live"], default=None,
                    help="Override app_mode from config/settings.yaml (stub = canned, live = OpenAI).")
    group.addoption("--run-benchmarks", action="store_true", default=False,
                    help="Allow the expensive benchmark tests (test_10) to run. Needs --app-mode live.")
    group.addoption("--max-goldens", type=int, default=None,
                    help="Cap on synthetic goldens per generation call (default: limits.max_goldens).")


def pytest_configure(config: pytest.Config) -> None:
    mode = config.getoption("--app-mode")
    if mode:
        os.environ["APP_MODE"] = mode  # read by Config.app_mode


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Two global rules.

    1. No OPENAI_API_KEY -> skip (do not crash) every test that needs the LLM judge.
    2. APP_MODE=live -> make the designed-to-fail demos NON-strict. Their "bad" answers are canned
       in the stub, so a real model may answer them correctly and the demo would XPASS. In stub
       mode (the default) they stay strict.
    """
    if not openai_key_present():
        skip = pytest.mark.skip(reason=NO_KEY_REASON)
        for item in items:
            if item.get_closest_marker("no_judge") is None:
                item.add_marker(skip)

    if os.getenv("APP_MODE", load_config().app_mode).lower() == "live":
        for item in items:
            marker = item.get_closest_marker("xfail")
            if marker is not None and marker.kwargs.get("strict"):
                relaxed = {**marker.kwargs, "strict": False}
                item.add_marker(pytest.mark.xfail(*marker.args, **relaxed), append=False)


# --------------------------------------------------------------------------------------
# Configuration fixtures
# --------------------------------------------------------------------------------------
@pytest.fixture(scope="session")
def config():
    return load_config()


@pytest.fixture(scope="session")
def thresholds():
    return load_thresholds()


@pytest.fixture(scope="session")
def metric_factory(config, thresholds) -> MetricFactory:
    return MetricFactory(config, thresholds)


@pytest.fixture(scope="session")
def eval_kwargs(config) -> dict:
    """async / cache / error settings for evaluate() and evals_iterator()."""
    return eval_configs(config)


@pytest.fixture(scope="session")
def max_goldens(request, config) -> int:
    return request.config.getoption("--max-goldens") or int(config.require("limits.max_goldens"))


@pytest.fixture(scope="session")
def run_benchmarks(request) -> bool:
    """True when the user opted in: --run-benchmarks, or RUN_BENCHMARKS=1 (the `deepeval test run`
    CLI cannot forward custom pytest flags, so CI and the Makefile use the env var)."""
    return bool(request.config.getoption("--run-benchmarks")) or os.getenv("RUN_BENCHMARKS") == "1"


@pytest.fixture(scope="session")
def reports_dir(config) -> Path:
    return config.reports_dir


# --------------------------------------------------------------------------------------
# The system under test
# --------------------------------------------------------------------------------------
@pytest.fixture(scope="session")
def rag_app(config) -> RAGPipeline:
    return RAGPipeline(config)


@pytest.fixture(scope="session")
def agent_app(config) -> ClaimsAgent:
    return ClaimsAgent(config)


@pytest.fixture()
def chatbot_app(config) -> ClaimsChatbot:
    """A NEW chatbot per test, because it remembers the conversation."""
    return ClaimsChatbot(config)


@pytest.fixture()
def sample_dataset(config) -> EvaluationDataset:
    """The curated goldens in data/datasets/goldens.json as an EvaluationDataset."""
    dataset = EvaluationDataset()
    dataset.add_goldens_from_json_file(file_path=str(config.path("paths.datasets_dir") / "goldens.json"))
    return dataset
