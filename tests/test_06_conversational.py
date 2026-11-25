"""
FEATURE: Multi-turn conversation evaluation
===========================================

WHAT IT IS
    Real support chats are not one question and one answer. Quality shows up ACROSS turns:
    does the assistant remember what the customer said earlier, stay in character, and actually
    finish the job? DeepEval models this with a ConversationalTestCase: an ordered list of
    `Turn(role="user" | "assistant", content=...)` plus optional context such as the assistant's
    `chatbot_role` and the `scenario` being tested.

WHY IT MATTERS
    A bot can answer every single message correctly and still frustrate people, for example by
    asking for a policy number the customer gave two messages ago, or by drifting out of its role
    and giving legal advice. Single-turn tests cannot see these failures.

THE METRICS
    TurnRelevancyMetric            is each assistant reply relevant to the conversation so far?
    KnowledgeRetentionMetric       does the assistant remember facts the user already gave?
    RoleAdherenceMetric            does it stay in the assigned `chatbot_role`?
    ConversationCompletenessMetric were the user's goals actually satisfied by the end?
    ConversationalGEval            your own plain-English rule applied to the whole chat (here: empathy)

SIMULATED USERS
    Writing conversations by hand does not scale. `ConversationSimulator` plays an LLM-driven
    "customer" (from a scenario + persona) against our chatbot to generate fresh conversations,
    which we then score with the same metrics. The number of conversations and turns is capped
    in config/settings.yaml (limits.max_simulated_conversations, limits.max_user_simulations).

HOW TO READ PASS / FAIL
    Scores are per conversation. The xfail demos include a bot that FORGETS the policy number,
    gives an OFF-TOPIC answer, leaves the user's request INCOMPLETE, breaks its ROLE by giving
    legal advice, and answers a distressed customer with NO EMPATHY.

HOW TO RUN
    pytest -m conversational
    make conversational
"""
import pytest
from deepeval import assert_test
from deepeval.dataset import EvaluationDataset
from deepeval.simulator import ConversationSimulator
from deepeval.test_case import ConversationalTestCase, MultiTurnParams, Turn

from app.chatbot import CHATBOT_ROLE
from framework.test_case_builders import build_conversation

pytestmark = pytest.mark.conversational

POLICY_INTRO = "Hi, my policy number is POL-100234 and I had a fender bender yesterday."
NEXT_STEPS = "Yes please, what do I need to do first?"

# --- scripted customer messages (the stub chatbot has canned replies for every one of them) ---
GOOD_RETENTION = [POLICY_INTRO, NEXT_STEPS, "Can you remind me which policy number I gave you?"]
FORGETFUL = [POLICY_INTRO, NEXT_STEPS, "Could you confirm the policy I told you about earlier?"]
OFF_TOPIC = [POLICY_INTRO, "Does my policy include rental car coverage?"]
ON_TOPIC = [POLICY_INTRO, "How much does rental car reimbursement pay?"]
COMPLETE = ["I want to file a claim for hail damage. How do I start and how long will it take?"]
INCOMPLETE = ["I want to file a claim for a stolen car. How do I start and how long will it take?"]
LEGAL_OK = ["Should I sue the other driver?"]
LEGAL_BAD = ["Is it worth suing the other driver for my injuries?"]
EMPATHETIC = ["My car was damaged in a crash last night."]
COLD = ["My house was damaged by a fire last night."]


def empathy_rule(factory):
    """A conversational G-Eval: any plain-English rule, judged over the whole conversation."""
    return factory.conversational_g_eval(
        "conversational_geval",
        name="Empathy",
        criteria=(
            "When the customer reports damage, loss or an accident, the assistant acknowledges how "
            "they feel with a sympathetic phrase and offers to help, instead of replying coldly."
        ),
        evaluation_params=[MultiTurnParams.CONTENT, MultiTurnParams.ROLE],
    )


# ======================================================================================
# 0. Anatomy of a conversational test case (no judge needed)
# ======================================================================================
@pytest.mark.smoke
@pytest.mark.no_judge
def test_conversation_is_built_from_turn_objects(chatbot_app):
    conversation = build_conversation(
        chatbot_app,
        GOOD_RETENTION,
        scenario="Customer reports an accident and later asks the bot to repeat their policy number.",
        expected_outcome="The bot repeats POL-100234 without asking again.",
    )
    assert isinstance(conversation, ConversationalTestCase)
    assert [t.role for t in conversation.turns] == ["user", "assistant"] * 3
    assert all(isinstance(t, Turn) for t in conversation.turns)
    assert conversation.chatbot_role == CHATBOT_ROLE
    assert "POL-100234" in conversation.turns[-1].content


@pytest.mark.no_judge
def test_chatbot_keeps_history_between_turns(chatbot_app):
    """Free-form (non-canned) input: the stub still finds the policy number earlier in the chat."""
    chatbot_app.reply("My policy number is POL-300891.")
    assert "POL-300891" in chatbot_app.reply("What policy number did I mention?")


# ======================================================================================
# 1. Conversation metrics (pass)
# ======================================================================================
@pytest.mark.smoke
def test_assistant_remembers_the_policy_number_given_earlier(chatbot_app, metric_factory):
    conversation = build_conversation(chatbot_app, GOOD_RETENTION)
    assert_test(conversation, [metric_factory.build("knowledge_retention")])


def test_every_assistant_reply_is_relevant_to_the_conversation(chatbot_app, metric_factory):
    conversation = build_conversation(chatbot_app, ON_TOPIC)
    assert_test(conversation, [metric_factory.build("turn_relevancy")])


def test_assistant_stays_in_role_and_declines_legal_advice(chatbot_app, metric_factory):
    conversation = build_conversation(chatbot_app, LEGAL_OK)
    assert_test(conversation, [metric_factory.build("role_adherence")])


def test_conversation_satisfies_the_users_request(chatbot_app, metric_factory):
    conversation = build_conversation(chatbot_app, COMPLETE)
    assert_test(conversation, [metric_factory.build("conversation_completeness")])


def test_assistant_is_empathetic_when_customer_reports_an_accident(chatbot_app, metric_factory):
    conversation = build_conversation(chatbot_app, EMPATHETIC)
    assert_test(conversation, [empathy_rule(metric_factory)])


# ======================================================================================
# 2. Simulated conversations (ConversationSimulator)
# ======================================================================================
def _simulate(chatbot_app, config, metric_factory):
    """Generate conversations from the scenarios in data/datasets/conversations.json."""
    dataset = EvaluationDataset()
    dataset.add_goldens_from_json_file(file_path=str(config.path("paths.datasets_dir") / "conversations.json"))
    goldens = dataset.goldens[: int(config.require("limits.max_simulated_conversations"))]
    assert goldens, "conversations.json produced no conversational goldens"

    def model_callback(input: str, turns: list[Turn]) -> Turn:
        """The simulator calls this once per simulated user message. We answer with the chatbot."""
        history = [{"role": t.role, "content": t.content} for t in turns]
        if history and history[-1]["role"] == "user" and history[-1]["content"] == input:
            history = history[:-1]  # chatbot.respond() appends the new user message itself
        return Turn(role="assistant", content=chatbot_app.respond(input, history))

    simulator = ConversationSimulator(
        model_callback=model_callback,
        simulator_model=metric_factory.judge,
        async_mode=False,
        max_concurrent=int(config.get("evaluation.max_concurrent", 5)),
    )
    cases = simulator.simulate(
        conversational_goldens=goldens,
        max_user_simulations=int(config.require("limits.max_user_simulations")),
    )
    for case in cases:
        case.chatbot_role = CHATBOT_ROLE  # the simulator knows the customer, not our bot's role
    return cases


@pytest.mark.slow
def test_simulated_conversations_keep_the_assistant_in_role(chatbot_app, metric_factory, config):
    conversations = _simulate(chatbot_app, config, metric_factory)
    assert conversations, "the simulator produced no conversations"
    for conversation in conversations:
        assert len(conversation.turns) >= 2, "a simulated conversation needs at least one exchange"
        assert_test(conversation, [metric_factory.build("role_adherence")])


# ======================================================================================
# 3. DESIGNED TO FAIL
# ======================================================================================
@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="demonstrates knowledge attrition: the bot forgets the policy number the customer gave in turn 1",
)
def test_assistant_that_forgets_the_policy_number_is_caught(chatbot_app, metric_factory):
    assert_test(build_conversation(chatbot_app, FORGETFUL), [metric_factory.build("knowledge_retention")])


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="demonstrates an off-topic reply: asked about rental cars, the bot talks about renewals",
)
def test_off_topic_reply_lowers_turn_relevancy(chatbot_app, metric_factory):
    assert_test(build_conversation(chatbot_app, OFF_TOPIC), [metric_factory.build("turn_relevancy")])


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="demonstrates an incomplete conversation: the user's two questions are never answered",
)
def test_vague_reply_leaves_the_request_incomplete(chatbot_app, metric_factory):
    assert_test(build_conversation(chatbot_app, INCOMPLETE), [metric_factory.build("conversation_completeness")])


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="demonstrates a role violation: the bot tells the customer to sue and promises they will win",
)
def test_assistant_that_gives_legal_advice_breaks_its_role(chatbot_app, metric_factory):
    assert_test(build_conversation(chatbot_app, LEGAL_BAD), [metric_factory.build("role_adherence")])


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="demonstrates the empathy rule failing: a customer reports a house fire and gets a one-line instruction",
)
def test_cold_reply_to_a_distressed_customer_fails_the_empathy_rule(chatbot_app, metric_factory):
    assert_test(build_conversation(chatbot_app, COLD), [empathy_rule(metric_factory)])
