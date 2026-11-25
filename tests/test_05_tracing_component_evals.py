"""
FEATURE: Tracing and component-level evaluation
===============================================

WHAT IT IS
    Tracing records every step an app takes as a tree of "spans":

        query (pipeline)
        |-- retrieve   [retriever span]  finds documents
        |-- generate   [llm span]        writes the answer
        run (agent)    [agent span]
        |-- plan       [llm span]
        |-- call_tool  [tool span]       one per tool call

    Our mock app is instrumented with `@observe` in app/rag.py and app/agent.py. Once spans exist,
    you can attach a metric to ONE span ("score the retriever on its own") instead of only
    scoring the final answer.

BLACK-BOX vs COMPONENT-LEVEL (when to use which)
    BLACK-BOX / END-TO-END   Score only (input, final output). Simple, matches what a user sees,
                             and works on any app, even one you cannot instrument. Use it for
                             release gates ("is the product good?"). Weakness: when it fails, it
                             does not say WHERE, and it can PASS while an inner part is broken.
    COMPONENT-LEVEL          Score each span separately. Use it for debugging and regression
                             tracking of individual parts (retriever, tools, prompts), and when
                             a part can be broken while the final answer still looks fine.
    Rule of thumb: gate releases on black-box tests, diagnose with component-level tests.

HOW THIS FILE ATTACHES METRICS
    app/tracing_hooks.py lets a test say "put THESE metrics on THAT span"
    (`set_span_metrics(retriever=[...], generator=[...])`). The app then calls DeepEval's
    `update_current_span(test_case=..., metrics=[...])` inside the span. Metrics still come from
    metric_factory, so thresholds stay in thresholds.yaml.

HOW TO READ PASS / FAIL
    Each span-level metric is reported with the span's name (retrieve, generate, call_tool).
    The xfail demo shows a case where the end-to-end check is green but the retriever span is red.

HOW TO RUN
    pytest -m tracing
"""
import copy

import pytest
from deepeval import assert_test
from deepeval.dataset import Golden
from deepeval.tracing import current_trace_context, observe

from app.tracing_hooks import clear_span_metrics, set_span_metrics
from framework.evals_runner import run_trace_evals
from framework.test_case_builders import build_rag_test_case

pytestmark = pytest.mark.tracing

RENTAL_Q = "How much does rental car reimbursement pay?"
PAYOUT_Q = "I have a $4,200 auto claim on policy POL-100234. What would the estimated payout be?"


@pytest.fixture(autouse=True)
def _clean_span_metrics():
    """Span metrics are global to the app, so always reset them between tests."""
    clear_span_metrics()
    yield
    clear_span_metrics()


class _WrongDocsRetriever:
    """A deliberately broken search: always returns two unrelated documents."""

    def __init__(self, real):
        self._pairs = [(n, d) for n, d in zip(real.names, real.docs) if n.startswith(("07_", "08_"))]

    def search(self, question, top_k):
        return self._pairs[:top_k]


@pytest.fixture()
def rag_with_broken_retriever(rag_app):
    broken = copy.copy(rag_app)  # shares config, replaces only the retriever
    broken.retriever = _WrongDocsRetriever(rag_app.retriever)
    return broken


def _goldens(question: str) -> list[Golden]:
    return [Golden(input=question)]


def _span_types(*calls) -> list[str]:
    """Run the given zero-argument callables inside one trace and list the span classes created."""

    @observe(type="agent")
    def probe():
        for call in calls:
            call()
        names: list[str] = []

        def walk(span):
            names.append(type(span).__name__)
            for child in span.children:
                walk(child)

        for root in current_trace_context.get().root_spans:
            walk(root)
        return names

    return probe()


# ======================================================================================
# 1. The app really is instrumented (no judge needed)
# ======================================================================================
@pytest.mark.smoke
@pytest.mark.no_judge
def test_app_emits_retriever_llm_tool_and_agent_spans(rag_app, agent_app):
    types = _span_types(lambda: rag_app.query(RENTAL_Q), lambda: agent_app.run(PAYOUT_Q))
    assert {"RetrieverSpan", "LlmSpan", "ToolSpan", "AgentSpan"} <= set(types), types


@pytest.mark.no_judge
def test_span_level_metric_runs_on_the_tool_span_without_any_llm(agent_app, eval_kwargs):
    """Component-level evaluation needs no judge when the span metric is deterministic."""
    from deepeval.metrics import PatternMatchMetric

    # Every claim lookup must return an output containing a "status" field.
    set_span_metrics(tool=[PatternMatchMetric(pattern=r'(?s).*"status".*')])
    run_trace_evals(_goldens("What is the status of claim CLM-2001?"), agent_app.run, **eval_kwargs)


# ======================================================================================
# 2. Black-box (end-to-end) evaluation
# ======================================================================================
@pytest.mark.smoke
def test_black_box_final_answer_is_relevant(rag_app, metric_factory):
    """Looks ONLY at what the user sees: the question and the final answer."""
    assert_test(build_rag_test_case(rag_app, RENTAL_Q), [metric_factory.build("answer_relevancy")])


# ======================================================================================
# 3. Component-level evaluation: metrics on individual spans
# ======================================================================================
def test_component_level_retriever_and_generator_both_pass(rag_app, metric_factory, eval_kwargs):
    """Contextual relevancy on the RETRIEVER span, answer relevancy on the GENERATOR (llm) span."""
    set_span_metrics(
        retriever=[metric_factory.build("contextual_relevancy")],
        generator=[metric_factory.build("answer_relevancy")],
    )
    run_trace_evals(_goldens(RENTAL_Q), rag_app.query, **eval_kwargs)


def test_component_level_planner_span_is_relevant(agent_app, metric_factory, eval_kwargs):
    """The same idea for the agent: judge the planner (llm) span's output on its own."""
    set_span_metrics(planner=[metric_factory.build("answer_relevancy")])
    run_trace_evals(_goldens(PAYOUT_Q), agent_app.run, **eval_kwargs)


# ======================================================================================
# 4. Side by side: the same broken pipeline, two verdicts
# ======================================================================================
def test_black_box_alone_misses_a_broken_retriever(rag_with_broken_retriever, metric_factory, config):
    """With search broken, the STUB generator still produces a fluent, relevant answer (as an LLM
    answering from memory might), so an end-to-end relevancy check stays GREEN.
    That is exactly the weakness of black-box testing: the defect is hidden."""
    if config.app_mode != "stub":
        pytest.skip("In live mode the generator reads the (wrong) documents, so this contrast only holds in stub mode.")
    test_case = build_rag_test_case(rag_with_broken_retriever, RENTAL_Q)
    assert_test(test_case, [metric_factory.build("answer_relevancy")])


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="demonstrates component-level evaluation catching a retriever fault that the black-box test above cannot see",
)
def test_component_level_catches_a_broken_retriever(rag_with_broken_retriever, metric_factory, eval_kwargs):
    set_span_metrics(retriever=[metric_factory.build("contextual_relevancy")])
    # Fails: the retrieve span scores below the contextual_relevancy threshold.
    run_trace_evals(_goldens(RENTAL_Q), rag_with_broken_retriever.query, **eval_kwargs)
