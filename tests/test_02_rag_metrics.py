"""
FEATURE: RAG evaluation (Retrieval-Augmented Generation)
========================================================

WHAT IT IS
    A RAG app answers in two stages:
        1. RETRIEVER  - finds the policy documents that look relevant to the question.
        2. GENERATOR  - writes an answer using those documents.
    A wrong final answer can be caused by EITHER stage, and the fix is different for each.
    DeepEval's RAG metrics are built to tell the two apart.

WHY IT MATTERS
    "The chatbot gave a wrong answer" is not actionable. "The retriever never found the
    rental-car document" (fix search) vs "the generator ignored the document" (fix prompt/model)
    is. Splitting the score by stage tells you where to spend engineering time.

THE METRICS (which stage each one blames)
    Retriever side                              Generator side
    - Contextual Precision: relevant docs        - Faithfulness: does the answer stick to the
      ranked ABOVE irrelevant ones?                retrieved documents (no invented facts)?
    - Contextual Recall: did retrieval find      - Answer Relevancy: does the answer address
      ALL the facts the ideal answer needs?        the question that was asked?
    - Contextual Relevancy: is the retrieved     - Hallucination: does the answer contradict
      text actually about the question?            a trusted "ground truth" context?
    - RAGAS: a combined score from the RAGAS library (optional install, see test below).

HOW TO READ PASS / FAIL
    Low CONTEXTUAL scores with a good answer   -> RETRIEVER problem (fix search / chunking).
    Good contextual scores but low Faithfulness -> GENERATOR problem (fix prompt / model).
    The two `xfail` demos below show one of each on purpose.

HOW TO RUN
    pytest -m rag
    make rag
"""
import pytest
from deepeval import assert_test
from deepeval.test_case import LLMTestCase

from framework.test_case_builders import build_rag_test_case

pytestmark = pytest.mark.rag

SMOKE = pytest.mark.smoke

# Questions whose canned stub answers are GOOD and match data/policies/.
DEDUCTIBLE_Q = "What is the standard deductible for auto collision claims?"
DEDUCTIBLE_IDEAL = (
    "The standard deductible for auto collision claims is $500. A lower $250 deductible is "
    "available on some auto policies for a higher premium."
)
DEDUCTIBLE_TRUTH = [
    "The standard deductible for auto collision and comprehensive claims is $500. "
    "A lower deductible of $250 is available on some auto policies for a higher premium."
]
REPORT_Q = "How long do I have to report a claim?"
RENTAL_Q = "How much does rental car reimbursement pay?"
RENTAL_IDEAL = (
    "Rental car reimbursement pays up to $40 per day for a maximum of 30 days, as long as "
    "rental coverage is on your policy."
)
# Questions whose stub answers are BAD on purpose.
FLOOD_Q = "Does my home policy cover flood damage?"
FLOOD_IDEAL = "No. Flood damage is excluded from home policies."
FLOOD_TRUTH = ["Flood damage is not covered by auto or home policies."]
ROADSIDE_Q = "How much does roadside assistance cover?"


def _doc(rag_app, filename: str) -> str:
    """Fetch one policy document by file name (used to hand-build good or bad retrievals)."""
    return rag_app.retriever.docs[rag_app.retriever.names.index(filename)]


# ======================================================================================
# GENERATOR-SIDE METRICS (pass)
# ======================================================================================
@SMOKE
def test_answer_is_faithful_to_policy_documents(rag_app, metric_factory):
    """Every claim in the answer must be supported by the retrieved documents."""
    assert_test(build_rag_test_case(rag_app, DEDUCTIBLE_Q), [metric_factory.build("faithfulness")])


@pytest.mark.parametrize(
    "question",
    [
        pytest.param(DEDUCTIBLE_Q, marks=SMOKE, id="deductible"),
        pytest.param(REPORT_Q, id="report-deadline"),
        pytest.param(RENTAL_Q, id="rental-car"),
    ],
)
def test_answer_is_relevant_to_the_question(rag_app, metric_factory, question):
    """The answer should address what was asked, without wandering off-topic."""
    assert_test(build_rag_test_case(rag_app, question), [metric_factory.build("answer_relevancy")])


def test_answer_does_not_contradict_ground_truth_context(rag_app, metric_factory):
    """Hallucination compares the answer with `context` (facts WE trust), not with retrieved text."""
    test_case = build_rag_test_case(rag_app, DEDUCTIBLE_Q, context=DEDUCTIBLE_TRUTH)
    assert_test(test_case, [metric_factory.build("hallucination")])


# ======================================================================================
# RETRIEVER-SIDE METRICS (pass)
# ======================================================================================
def test_retriever_ranks_relevant_documents_first(rag_app, metric_factory):
    """Contextual Precision rewards putting the useful document at rank 1, not rank 3."""
    test_case = build_rag_test_case(rag_app, DEDUCTIBLE_Q, expected_output=DEDUCTIBLE_IDEAL)
    assert_test(test_case, [metric_factory.build("contextual_precision")])


def test_retriever_finds_every_fact_the_ideal_answer_needs(rag_app, metric_factory):
    """Contextual Recall: can each sentence of the ideal answer be traced to a retrieved document?"""
    test_case = build_rag_test_case(rag_app, RENTAL_Q, expected_output=RENTAL_IDEAL)
    assert_test(test_case, [metric_factory.build("contextual_recall")])


def test_retrieved_documents_are_about_the_question(rag_app, metric_factory):
    """Contextual Relevancy: how much of the retrieved text is on-topic (penalises noisy retrieval)?"""
    assert_test(build_rag_test_case(rag_app, RENTAL_Q), [metric_factory.build("contextual_relevancy")])


def test_generator_stays_honest_when_retrieval_is_poor(rag_app, metric_factory):
    """The retriever failed (wrong documents) but the generator admitted it. Faithfulness PASSES:
    the generator behaved correctly given what it was given. Compare with the xfail retriever demo below."""
    test_case = LLMTestCase(
        input=RENTAL_Q,
        actual_output="I'm sorry, I couldn't find that in our policy documents.",
        retrieval_context=[_doc(rag_app, "07_cancellation_and_renewal.md")],
    )
    assert_test(test_case, [metric_factory.build("faithfulness")])


# ======================================================================================
# COMBINED RAGAS METRIC (optional dependency)
# ======================================================================================
def test_combined_ragas_score_is_above_threshold(rag_app, metric_factory):
    """RAGAS bundles precision, recall, answer relevancy and faithfulness into one number.
    DeepEval 4.x keeps it at `deepeval.metrics.ragas.RagasMetric` (no longer exported from
    `deepeval.metrics`) and it needs extra packages. Skipped unless they are installed:
        pip install ragas datasets langchain-openai
    """
    pytest.importorskip("ragas", reason="ragas not installed (optional)")
    pytest.importorskip("datasets", reason="datasets not installed (optional)")
    pytest.importorskip("langchain_core", reason="langchain-core not installed (optional)")
    from deepeval.metrics.ragas import RagasMetric

    test_case = build_rag_test_case(rag_app, DEDUCTIBLE_Q, expected_output=DEDUCTIBLE_IDEAL)
    assert_test(test_case, [RagasMetric(threshold=metric_factory.threshold("ragas"), model=metric_factory.config.judge_model)])


# ======================================================================================
# DESIGNED TO FAIL: one failure per pipeline stage
# ======================================================================================
@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="demonstrates a RETRIEVER problem: the retrieved documents lack the facts needed (low contextual recall)",
)
def test_retriever_problem_documents_miss_the_needed_facts(rag_app, metric_factory):
    # We hand the pipeline two UNRELATED documents, as if search had failed. The ideal answer
    # talks about rental cars, but nothing in the context mentions them -> recall is low.
    test_case = LLMTestCase(
        input=RENTAL_Q,
        actual_output="I'm sorry, I couldn't find that in our policy documents.",
        expected_output=RENTAL_IDEAL,
        retrieval_context=[_doc(rag_app, "07_cancellation_and_renewal.md"), _doc(rag_app, "04_exclusions.md")],
    )
    assert_test(test_case, [metric_factory.build("contextual_recall")])


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="demonstrates a RETRIEVER problem: retrieved text is off-topic (low contextual relevancy)",
)
def test_retriever_problem_documents_are_off_topic(rag_app, metric_factory):
    test_case = LLMTestCase(
        input=RENTAL_Q,
        actual_output="I'm sorry, I couldn't find that in our policy documents.",
        retrieval_context=[_doc(rag_app, "07_cancellation_and_renewal.md"), _doc(rag_app, "08_privacy_and_fraud.md")],
    )
    assert_test(test_case, [metric_factory.build("contextual_relevancy")])


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="demonstrates a GENERATOR problem: retrieval is fine but the answer says flood IS covered (low faithfulness)",
)
def test_generator_problem_answer_contradicts_retrieved_documents(rag_app, metric_factory):
    test_case = build_rag_test_case(rag_app, FLOOD_Q, expected_output=FLOOD_IDEAL)
    # The retriever did its job: the exclusions document (which says flood is NOT covered) is in the context.
    assert any("flood" in doc.lower() for doc in test_case.retrieval_context)
    assert_test(test_case, [metric_factory.build("faithfulness")])


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="demonstrates a hallucination: the answer contradicts the trusted ground-truth context",
)
def test_hallucinated_answer_contradicts_ground_truth(rag_app, metric_factory):
    test_case = build_rag_test_case(rag_app, FLOOD_Q, context=FLOOD_TRUTH)
    assert_test(test_case, [metric_factory.build("hallucination")])


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="demonstrates an irrelevant answer: it talks about company history instead of roadside assistance",
)
def test_irrelevant_answer_fails_answer_relevancy(rag_app, metric_factory):
    assert_test(build_rag_test_case(rag_app, ROADSIDE_Q), [metric_factory.build("answer_relevancy")])
