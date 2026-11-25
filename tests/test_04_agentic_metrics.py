"""
FEATURE: Agent evaluation (tool-calling AI)
===========================================

WHAT IT IS
    An "agent" does not just answer; it DECIDES which tools to use (look up a policy, check a
    claim, calculate a payout), calls them, and then writes the answer. Agents can fail in ways
    a plain chatbot cannot: wrong tool, right tool with wrong inputs, pointless repeated calls,
    or a good plan that is not followed.

WHY IT MATTERS
    A wrong tool call can quote the wrong customer's claim or compute the wrong payout, even when
    the final sentence sounds confident. These metrics check the PROCESS, not just the prose.

TWO KINDS OF AGENT METRIC
    Test-case metrics (work on an ordinary LLMTestCase, run with assert_test):
        ToolCorrectnessMetric      did it call the expected tools (optionally in order, with the
                                   expected arguments)?
        ArgumentCorrectnessMetric  were the arguments sensible for the user's request?
    TRACE-ONLY metrics (need the full execution trace, i.e. every step the agent took; they
    cannot work from a final answer alone, so they MUST run through tracing with
    `dataset.evals_iterator(metrics=[...])`):
        TaskCompletionMetric       did the whole run achieve the user's goal?
        PlanQualityMetric          is the plan logical, complete and efficient?
        PlanAdherenceMetric        did the agent follow its own plan?
        StepEfficiencyMetric       did it avoid redundant or unnecessary steps?
    Our mock agent is instrumented with `@observe` (see app/agent.py), so each run produces a trace
    with an agent span, planner (llm) spans and tool spans.

HOW TO READ PASS / FAIL
    Score near 1.0 = the agent behaved as intended. The `xfail` demos show the three classic
    agent bugs: a WRONG TOOL, WRONG ARGUMENTS and REDUNDANT STEPS.

HOW TO RUN
    pytest -m agent
    make agents
"""
import pytest
from deepeval import assert_test
from deepeval.dataset import Golden
from deepeval.test_case import ToolCall, ToolCallParams

from framework.evals_runner import run_trace_evals
from framework.test_case_builders import build_agent_test_case

pytestmark = pytest.mark.agent

STATUS_Q = "What is the status of claim CLM-2001?"
PAYOUT_Q = "I have a $4,200 auto claim on policy POL-100234. What would the estimated payout be?"
WRONG_TOOL_Q = "Can you check the status of my claim CLM-2002?"          # stub calls the POLICY tool by mistake
REDUNDANT_Q = "Is claim CLM-2003 approved yet?"                          # stub calls the same tool 3 times
WRONG_ARGS_Q = "Estimate the payout on a $6,000 claim for policy POL-200567."  # stub types $600 and a $250 deductible

STATUS_TOOLS = [ToolCall(name="check_claim_status", input_parameters={"claim_id": "CLM-2001"})]
PAYOUT_TOOLS = [
    ToolCall(name="lookup_policy", input_parameters={"policy_id": "POL-100234"}),
    ToolCall(name="calculate_payout", input_parameters={"claim_amount": 4200, "deductible": 500}),
]


def run_trace_eval(agent_app, question: str, metric, eval_kwargs):
    """Run the agent under tracing so trace-only metrics can see every step it took."""
    run_trace_evals([Golden(input=question)], agent_app.run, metrics=[metric], **eval_kwargs)


# ======================================================================================
# TEST-CASE METRICS: ToolCorrectness and ArgumentCorrectness
# ======================================================================================
@pytest.mark.smoke
def test_agent_calls_the_right_tool_for_a_claim_status_question(agent_app, metric_factory):
    test_case = build_agent_test_case(agent_app, STATUS_Q, expected_tools=STATUS_TOOLS)
    assert_test(test_case, [metric_factory.build("tool_correctness")])


def test_agent_calls_tools_in_order_with_the_expected_arguments(agent_app, metric_factory):
    """Also compares the arguments (INPUT_PARAMETERS) and insists on the expected ORDER."""
    test_case = build_agent_test_case(agent_app, PAYOUT_Q, expected_tools=PAYOUT_TOOLS)
    metric = metric_factory.build(
        "tool_correctness",
        evaluation_params=[ToolCallParams.INPUT_PARAMETERS],
        should_consider_ordering=True,
    )
    assert_test(test_case, [metric])


def test_agent_arguments_are_correct_for_the_users_request(agent_app, metric_factory):
    """ArgumentCorrectness asks an LLM judge: given the user's request, are the arguments right?"""
    test_case = build_agent_test_case(agent_app, PAYOUT_Q)
    assert_test(test_case, [metric_factory.build("argument_correctness")])


# ======================================================================================
# TRACE-ONLY METRICS (run through tracing + evals_iterator)
# ======================================================================================
@pytest.mark.smoke
def test_agent_completes_the_users_task(agent_app, metric_factory, eval_kwargs):
    metric = metric_factory.build("task_completion")
    run_trace_eval(agent_app, PAYOUT_Q, metric, eval_kwargs)


@pytest.mark.parametrize("metric_key", ["plan_quality", "plan_adherence", "step_efficiency"])
def test_agent_plan_and_steps_are_sound(agent_app, metric_factory, eval_kwargs, metric_key):
    """One test per trace-only metric, all on the same well-behaved run."""
    metric = metric_factory.build(metric_key)
    run_trace_eval(agent_app, PAYOUT_Q, metric, eval_kwargs)


# ======================================================================================
# DESIGNED TO FAIL: the three classic agent bugs
# ======================================================================================
@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="demonstrates a WRONG TOOL: the agent calls lookup_policy for a claim-status question",
)
def test_agent_that_calls_the_wrong_tool_is_caught(agent_app, metric_factory):
    expected = [ToolCall(name="check_claim_status", input_parameters={"claim_id": "CLM-2002"})]
    test_case = build_agent_test_case(agent_app, WRONG_TOOL_Q, expected_tools=expected)
    assert_test(test_case, [metric_factory.build("tool_correctness")])


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="demonstrates WRONG ARGUMENTS: claim amount 600 instead of 6000 and the wrong deductible",
)
def test_agent_that_passes_wrong_arguments_is_caught(agent_app, metric_factory):
    test_case = build_agent_test_case(agent_app, WRONG_ARGS_Q)
    assert_test(test_case, [metric_factory.build("argument_correctness")])


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="demonstrates REDUNDANT STEPS: the same claim lookup is repeated three times plus an unneeded policy lookup",
)
def test_agent_that_repeats_steps_is_inefficient(agent_app, metric_factory, eval_kwargs):
    metric = metric_factory.build("step_efficiency")
    run_trace_eval(agent_app, REDUNDANT_Q, metric, eval_kwargs)


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="demonstrates an unfinished task: the wrong tool means the agent never reports the claim status",
)
def test_agent_that_looks_up_the_wrong_thing_does_not_complete_the_task(agent_app, metric_factory, eval_kwargs):
    metric = metric_factory.build("task_completion")
    run_trace_eval(agent_app, WRONG_TOOL_Q, metric, eval_kwargs)
