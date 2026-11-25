"""
FEATURE: Pytest-native unit testing with DeepEval
=================================================

WHAT IT IS
    DeepEval lets you write AI quality checks exactly like ordinary unit tests. You build a
    "test case" (the question, the answer your app gave, and optionally the ideal answer and the
    documents it used), pick one or more "metrics" (quality yardsticks scored 0 to 1), and call
    `assert_test`. If any score is below its threshold the test fails.

WHY IT MATTERS
    Everything else in this framework is built from these few pieces. If you understand this
    file, you understand the rest. It also shows how checks plug into CI: a red test blocks a merge.

WHAT THESE TESTS PROVE
    * An LLMTestCase carries input, actual_output, expected_output, context and retrieval_context.
    * assert_test() raises when a metric scores below its threshold (that is what "fail" means).
    * A metric exposes .score, .reason and .is_successful() after .measure().
    * Several metrics can judge one test case; evaluate() judges a whole list in one call.
    * strict_mode (pass only on a perfect score) and include_reason (explanations on/off).

HOW TO READ PASS / FAIL
    PASS   = every metric scored at or above its threshold (see config/thresholds.yaml).
    FAIL   = at least one metric scored below its threshold. The failure message names the
             metric, the score and the judge's reason.
    XFAIL  = "expected failure". A test that is DESIGNED to fail to show what a failure looks
             like. It is reported as `x` and keeps the suite green. If it ever unexpectedly
             passes, pytest reports XPASS(strict) as a failure, so you notice.

HOW TO RUN
    pytest -m basics                     # this file only
    pytest -m smoke                      # the cheapest tests from every file
    deepeval test run tests/test_01_pytest_basics.py -n 4
                                         # same tests, 4 parallel workers (needs pytest-xdist,
                                         # installed with deepeval) plus a DeepEval run report

NOTE: The judge is always a real OpenAI model, so most tests need OPENAI_API_KEY even when the
app runs in stub mode. Tests marked `no_judge` run without a key.
"""
import pytest
from deepeval import assert_test, evaluate
from deepeval.test_case import LLMTestCase

from framework.test_case_builders import build_rag_test_case

pytestmark = pytest.mark.basics

QUESTION = "What is the standard deductible for auto collision claims?"
IDEAL_ANSWER = (
    "The standard deductible for auto collision claims is $500. A lower $250 deductible is "
    "available on some auto policies for a higher premium."
)
BAD_QUESTION = "Does my home policy cover flood damage?"  # the stub answers this one WRONGLY on purpose


# ------------------------------------------------------------------------------------
# 1. The anatomy of a test case (no LLM judge needed)
# ------------------------------------------------------------------------------------
@pytest.mark.smoke
@pytest.mark.no_judge
def test_llm_test_case_holds_every_field_a_metric_may_need():
    """An LLMTestCase is just a labelled bundle of facts about ONE interaction."""
    test_case = LLMTestCase(
        input=QUESTION,                                      # what the user asked
        actual_output="The deductible is $500.",             # what OUR app answered
        expected_output=IDEAL_ANSWER,                        # the ideal answer (a "golden")
        retrieval_context=["Doc: the standard deductible is $500."],  # what the retriever FOUND
        context=["The standard deductible is $500."],        # the ground truth we trust
    )
    assert test_case.input == QUESTION
    assert test_case.actual_output.endswith("$500.")
    assert test_case.expected_output == IDEAL_ANSWER
    assert test_case.retrieval_context and test_case.context


@pytest.mark.no_judge
def test_thresholds_come_from_config_not_from_test_code(metric_factory, thresholds):
    """Deterministic metrics need no judge, so this also proves the config wiring without a key."""
    exact = metric_factory.build("exact_match")
    pattern = metric_factory.build("pattern_match", pattern=r"\$\d+")
    assert exact.threshold == thresholds["exact_match"]
    assert pattern.threshold == thresholds["pattern_match"]


# ------------------------------------------------------------------------------------
# 2. assert_test: the heart of DeepEval
# ------------------------------------------------------------------------------------
@pytest.mark.smoke
def test_answer_is_relevant_to_the_question(rag_app, metric_factory):
    """Answer Relevancy = does the answer actually address what was asked?"""
    test_case = build_rag_test_case(rag_app, QUESTION)
    assert_test(test_case, [metric_factory.build("answer_relevancy")])


def test_answer_is_faithful_to_retrieved_documents(rag_app, metric_factory):
    """Faithfulness = does the answer avoid claims that the retrieved documents do not support?"""
    test_case = build_rag_test_case(rag_app, QUESTION)
    assert_test(test_case, [metric_factory.build("faithfulness")])


def test_several_metrics_judge_the_same_test_case(rag_app, metric_factory):
    """Pass a list of metrics: the test fails if ANY of them is below its threshold."""
    test_case = build_rag_test_case(rag_app, QUESTION, expected_output=IDEAL_ANSWER)
    assert_test(
        test_case,
        [
            metric_factory.build("answer_relevancy"),
            metric_factory.build("faithfulness"),
            metric_factory.build("contextual_recall"),
        ],
    )


# ------------------------------------------------------------------------------------
# 3. Standalone metric.measure(): inspect score, reason and success yourself
# ------------------------------------------------------------------------------------
def test_metric_exposes_score_reason_and_success_flag(rag_app, metric_factory):
    """measure() runs ONE metric and fills in .score, .reason and .is_successful()."""
    test_case = build_rag_test_case(rag_app, QUESTION)
    metric = metric_factory.build("answer_relevancy")
    score = metric.measure(test_case)

    assert score == metric.score
    assert 0.0 <= metric.score <= 1.0
    assert metric.reason, "include_reason is on by default, so the judge explains its score"
    assert metric.is_successful() == (metric.score >= metric.threshold)
    assert metric.is_successful(), f"score {metric.score} < threshold {metric.threshold}: {metric.reason}"


def test_include_reason_false_skips_the_explanation(rag_app, metric_factory):
    """Switch the explanation off to save tokens when you only need the number."""
    test_case = build_rag_test_case(rag_app, QUESTION)
    metric = metric_factory.build("answer_relevancy", include_reason=False)
    metric.measure(test_case)
    assert metric.score is not None
    assert metric.reason is None


def test_strict_mode_accepts_a_perfectly_supported_answer(rag_app, metric_factory):
    """strict_mode=True forces the threshold to 1.0 and the score to all-or-nothing."""
    test_case = build_rag_test_case(rag_app, QUESTION)
    metric = metric_factory.build("faithfulness", strict_mode=True)
    metric.measure(test_case)
    assert metric.threshold == 1
    assert metric.is_successful(), metric.reason


# ------------------------------------------------------------------------------------
# 4. evaluate(): many test cases in one call
# ------------------------------------------------------------------------------------
def test_bulk_evaluate_scores_a_list_of_test_cases(rag_app, metric_factory, eval_kwargs):
    """evaluate() runs every metric on every test case (concurrently) and returns the results."""
    questions = [
        QUESTION,
        "How long do I have to report a claim?",
        "How much does rental car reimbursement pay?",
    ]
    test_cases = [build_rag_test_case(rag_app, q) for q in questions]
    result = evaluate(
        test_cases=test_cases,
        metrics=[metric_factory.build("answer_relevancy")],
        **eval_kwargs,
    )
    assert len(result.test_results) == len(test_cases)
    failures = [t.name for t in result.test_results if not t.success]
    assert not failures, f"These test cases failed: {failures}"


# ------------------------------------------------------------------------------------
# 5. DESIGNED TO FAIL: see what a failure looks like
# ------------------------------------------------------------------------------------
@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="demonstrates a failing assert_test: the stub says flood IS covered, the policy says it is NOT",
)
def test_unfaithful_answer_fails_assert_test(rag_app, metric_factory):
    test_case = build_rag_test_case(rag_app, BAD_QUESTION)
    assert_test(test_case, [metric_factory.build("faithfulness")])


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="demonstrates reading a failing score yourself: .is_successful() is False for a wrong answer",
)
def test_failing_metric_reports_low_score_and_a_reason(rag_app, metric_factory):
    test_case = build_rag_test_case(rag_app, BAD_QUESTION)
    metric = metric_factory.build("faithfulness")
    metric.measure(test_case)
    assert metric.is_successful(), f"score={metric.score} reason={metric.reason}"
