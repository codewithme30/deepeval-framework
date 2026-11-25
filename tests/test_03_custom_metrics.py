"""
FEATURE: Custom and deterministic metrics
=========================================

WHAT IT IS
    Built-in metrics measure general qualities (relevancy, faithfulness...). Your business has
    its OWN rules: "never promise a payout before approval", "always mention the deductible".
    DeepEval lets you encode those rules as metrics in four flavours:

        G-Eval        a rule written in plain English; an LLM judge scores against it.
        DAG           a decision tree of yes/no questions; the PATH taken decides the score,
                      so the grading logic is predictable and explainable.
        BaseMetric    your own Python class (here: a regex check). No LLM, no cost, no variance.
        Non-LLM       JsonCorrectness (does the output match a schema?), ExactMatch and
                      PatternMatch (string checks). Fast, free and fully repeatable.

WHY IT MATTERS
    Rules that can be checked with code SHOULD be checked with code (cheap and repeatable).
    Rules that need judgement are written in English and scored by an LLM. Mixing both gives
    broad coverage at low cost.

WHAT THESE TESTS PROVE
    * G-Eval with `criteria` (free-form rule) and with `evaluation_steps` (explicit checklist).
    * A DAG metric: greeting -> policy reference -> next-step call to action.
    * A custom BaseMetric with measure / a_measure / is_successful / __name__.
    * JsonCorrectnessMetric with a Pydantic schema, ExactMatchMetric and PatternMatchMetric.

HOW TO READ PASS / FAIL
    G-Eval / DAG: a 0-1 score from the judge; fails below the threshold in thresholds.yaml.
    Deterministic metrics: score is exactly 1.0 or 0.0, with a human-readable reason.
    Tests marked `no_judge` need no API key at all.

HOW TO RUN
    pytest -m custom
"""
import asyncio
import json
import re
from typing import Optional

import pytest
from deepeval import assert_test
from deepeval.metrics import BaseMetric
from deepeval.metrics.dag import BinaryJudgementNode, DeepAcyclicGraph, VerdictNode
from deepeval.models import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase, SingleTurnParams
from pydantic import BaseModel

from app.tools import check_claim_status
from framework.test_case_builders import build_rag_test_case

pytestmark = pytest.mark.custom

IO = [SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT]

# PatternMatchMetric uses re.fullmatch, so the pattern must describe the WHOLE answer: "anything, a $ amount, anything".
DOLLAR_AMOUNT = r"(?s).*\$\d[\d,]*.*"

# Stub question that has a GOOD answer (conditional wording, mentions the $500 deductible, greeting, policy ref, next step).
GOOD_PAYOUT_Q = "If my claim is approved for $4,200, how much will I be paid?"
# Stub question with a BAD answer ("you will definitely be paid the full $4,200!").
BAD_PAYOUT_Q = "My claim is for $4,200, what will you pay me?"


# ======================================================================================
# G-EVAL: business rules in plain English
# ======================================================================================
def no_payout_promise(factory):
    """Rule written as free-form CRITERIA: DeepEval turns it into evaluation steps for you."""
    return factory.g_eval(
        "no_payout_promise",
        criteria=(
            "The response must never promise or guarantee a payout amount before the claim is approved. "
            "Estimates are acceptable only if they are clearly conditional on approval."
        ),
        evaluation_params=IO,
    )


def deductible_included(factory):
    """Rule written as explicit EVALUATION STEPS: you control exactly what the judge checks."""
    return factory.g_eval(
        "deductible_mentioned",
        evaluation_steps=[
            "Check whether the user's input is about how much money they will receive for a claim.",
            "If it is, check that the response mentions the deductible and explains that it is subtracted.",
            "If the input is not about a payout, the response passes without mentioning a deductible.",
            "Penalise heavily if a payout figure is given with no mention of the deductible.",
        ],
        evaluation_params=IO,
    )


def test_answer_never_promises_a_payout_before_approval(rag_app, metric_factory):
    assert_test(build_rag_test_case(rag_app, GOOD_PAYOUT_Q), [no_payout_promise(metric_factory)])


def test_answer_includes_the_deductible_when_discussing_payouts(rag_app, metric_factory):
    assert_test(build_rag_test_case(rag_app, GOOD_PAYOUT_Q), [deductible_included(metric_factory)])


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="demonstrates a G-Eval business-rule violation: the answer guarantees a payout",
)
def test_answer_that_promises_a_payout_is_rejected(rag_app, metric_factory):
    assert_test(build_rag_test_case(rag_app, BAD_PAYOUT_Q), [no_payout_promise(metric_factory)])


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="demonstrates a G-Eval business-rule violation: a payout figure is given without the deductible",
)
def test_answer_that_forgets_the_deductible_is_rejected(rag_app, metric_factory):
    assert_test(build_rag_test_case(rag_app, BAD_PAYOUT_Q), [deductible_included(metric_factory)])


# ======================================================================================
# DAG: a decision tree with a predictable grading path
# ======================================================================================
def helpful_structure_dag(factory):
    """greeting? -> policy reference? -> next-step call-to-action?   (any 'no' ends the walk early)

    Leaf scores are 0-10; DeepEval divides by 10. Reaching the bottom with every 'yes'
    scores 10/10. Stopping at a 'no' scores 0 or 4/10 depending on how far the answer got.
    """
    next_step = BinaryJudgementNode(
        criteria="Does the response end with a clear next step or call to action for the customer?",
        evaluation_params=IO,
        children=[VerdictNode(verdict=False, score=4), VerdictNode(verdict=True, score=10)],
    )
    policy_reference = BinaryJudgementNode(
        criteria="Does the response reference a specific policy (a policy number or the policy terms)?",
        evaluation_params=IO,
        children=[VerdictNode(verdict=False, score=2), VerdictNode(verdict=True, child=next_step)],
    )
    greeting = BinaryJudgementNode(
        criteria="Does the response open with a greeting or courteous phrase?",
        evaluation_params=IO,
        children=[VerdictNode(verdict=False, score=0), VerdictNode(verdict=True, child=policy_reference)],
    )
    return factory.dag("helpful_structure", DeepAcyclicGraph(root_nodes=[greeting]))


def test_answer_has_greeting_policy_reference_and_next_step(rag_app, metric_factory):
    assert_test(build_rag_test_case(rag_app, GOOD_PAYOUT_Q), [helpful_structure_dag(metric_factory)])


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="demonstrates a DAG failing at its first node: a terse answer with no greeting, policy reference or next step",
)
def test_answer_without_greeting_fails_the_decision_tree(rag_app, metric_factory):
    assert_test(build_rag_test_case(rag_app, BAD_PAYOUT_Q), [helpful_structure_dag(metric_factory)])


# ======================================================================================
# CUSTOM BaseMetric: our own regex rule, no LLM involved
# ======================================================================================
class PolicyNumberFormatMetric(BaseMetric):
    """Every policy number in the answer must look like POL-123456 (POL, dash, six digits).

    A subclass of BaseMetric only has to provide: measure, a_measure, is_successful and __name__.
    Being deterministic, it costs nothing and never flakes.
    """

    CANDIDATE = re.compile(r"\bPOL[\s_-]?\d+\b", re.IGNORECASE)  # anything that LOOKS like a policy number
    VALID = re.compile(r"^POL-\d{6}$")                           # the one allowed format

    def __init__(self, threshold: float = 1.0):
        self.threshold = threshold
        self.score: Optional[float] = None
        self.reason: Optional[str] = None
        self.success: Optional[bool] = None
        self.async_mode = False
        self.strict_mode = False
        self.include_reason = True
        self.error = None
        self.evaluation_model = None

    def measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        found = self.CANDIDATE.findall(test_case.actual_output or "")
        malformed = [c for c in found if not self.VALID.match(c)]
        self.score = 0.0 if malformed else 1.0
        self.reason = (
            f"Malformed policy number(s): {malformed}. Expected POL-######."
            if malformed
            else f"All {len(found)} policy number(s) use the POL-###### format."
        )
        self.success = self.score >= self.threshold
        return self.score

    async def a_measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        return self.measure(test_case, *args, **kwargs)

    def is_successful(self) -> bool:
        return bool(self.success)

    @property
    def __name__(self):
        return "Policy Number Format"


@pytest.mark.smoke
@pytest.mark.no_judge
def test_policy_numbers_use_the_official_format(rag_app, metric_factory):
    metric = metric_factory.build("policy_number_format", cls=PolicyNumberFormatMetric)
    assert_test(build_rag_test_case(rag_app, GOOD_PAYOUT_Q), [metric])  # answer mentions POL-100234


@pytest.mark.no_judge
def test_custom_metric_supports_async_measurement(metric_factory):
    """a_measure is what evaluate() calls when running metrics concurrently."""
    metric = metric_factory.build("policy_number_format", cls=PolicyNumberFormatMetric)
    case = LLMTestCase(input="q", actual_output="Your policy POL-200567 is active.")
    assert asyncio.run(metric.a_measure(case)) == 1.0
    assert metric.is_successful()


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="demonstrates the custom regex metric catching a malformed policy number (POL-1234)",
)
@pytest.mark.no_judge
def test_malformed_policy_number_is_rejected(metric_factory):
    metric = metric_factory.build("policy_number_format", cls=PolicyNumberFormatMetric)
    case = LLMTestCase(input="Which policy?", actual_output="Your policy POL-1234 is active.")
    assert_test(case, [metric])


# ======================================================================================
# NON-LLM METRICS: JSON schema, exact match, pattern match
# ======================================================================================
class ClaimStatusPayload(BaseModel):
    """The structure a downstream system expects when it asks for a claim's status."""

    claim_id: str
    policy_id: str
    status: str
    amount: float
    adjuster: str
    next_step: str


class _NoJudge(DeepEvalBaseLLM):
    """Placeholder model so JsonCorrectnessMetric can be built (and run) without an API key.
    The schema check itself is plain code; a model is only used to PHRASE a failure reason,
    which we switch off with include_reason=False."""

    def __init__(self):
        super().__init__("no-judge")

    def load_model(self):
        return self

    def generate(self, *args, **kwargs):
        raise RuntimeError("This test must not call an LLM")

    async def a_generate(self, *args, **kwargs):
        raise RuntimeError("This test must not call an LLM")

    def get_model_name(self):
        return "no-judge"


def _json_metric(factory):
    return factory.json_correctness(ClaimStatusPayload, model=_NoJudge(), include_reason=False)


@pytest.mark.smoke
@pytest.mark.no_judge
def test_tool_output_matches_the_expected_json_schema(metric_factory):
    payload = json.dumps(check_claim_status("CLM-2001"))
    assert_test(LLMTestCase(input="claim status as JSON", actual_output=payload), [_json_metric(metric_factory)])


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="demonstrates JsonCorrectnessMetric failing: an unknown claim returns an error object, not a claim",
)
@pytest.mark.no_judge
def test_error_payload_does_not_match_the_claim_schema(metric_factory):
    payload = json.dumps(check_claim_status("CLM-9999"))
    assert_test(LLMTestCase(input="claim status as JSON", actual_output=payload), [_json_metric(metric_factory)])


@pytest.mark.no_judge
def test_answer_contains_a_dollar_amount(rag_app, metric_factory):
    metric = metric_factory.build("pattern_match", pattern=DOLLAR_AMOUNT)
    assert_test(build_rag_test_case(rag_app, "What is the standard deductible for auto collision claims?"), [metric])


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="demonstrates PatternMatchMetric failing: the irrelevant roadside answer contains no dollar amount",
)
@pytest.mark.no_judge
def test_answer_without_a_dollar_amount_fails_the_pattern(rag_app, metric_factory):
    metric = metric_factory.build("pattern_match", pattern=DOLLAR_AMOUNT)
    assert_test(build_rag_test_case(rag_app, "How much does roadside assistance cover?"), [metric])


@pytest.mark.no_judge
def test_stub_answer_exactly_matches_the_golden_answer(rag_app, metric_factory, config):
    """Exact match is only sensible for deterministic output, so it is skipped in live mode."""
    if config.app_mode != "stub":
        pytest.skip("Exact match needs deterministic output (APP_MODE=stub).")
    ideal = (
        "The standard deductible for auto collision claims is $500. A lower $250 deductible is "
        "available on some auto policies for a higher premium."
    )
    case = build_rag_test_case(
        rag_app, "What is the standard deductible for auto collision claims?", expected_output=ideal
    )
    assert_test(case, [metric_factory.build("exact_match")])


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="demonstrates ExactMatchMetric failing: a correct answer worded differently is NOT an exact match",
)
@pytest.mark.no_judge
def test_reworded_answer_fails_exact_match(rag_app, metric_factory):
    case = build_rag_test_case(
        rag_app,
        "How long do I have to report a claim?",
        expected_output="Claims must be reported within 30 days.",  # same meaning, different words
    )
    assert_test(case, [metric_factory.build("exact_match")])
