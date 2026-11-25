"""Helpers that run the mock app and package its output as DeepEval test cases.

Keeping this here means tests read like business rules ("ask this question, expect that
answer") instead of plumbing.
"""
from __future__ import annotations

from typing import Iterable

from deepeval.dataset import Golden
from deepeval.test_case import ConversationalTestCase, LLMTestCase, ToolCall, Turn

from app.agent import AgentResult, ClaimsAgent
from app.chatbot import CHATBOT_ROLE, ClaimsChatbot
from app.rag import RAGPipeline
from app.tracing_hooks import to_tool_calls


def build_rag_test_case(
    rag: RAGPipeline,
    question: str,
    expected_output: str | None = None,
    context: list[str] | None = None,
) -> LLMTestCase:
    """Run the RAG app and capture input, actual_output and retrieval_context."""
    result = rag.query(question)
    return LLMTestCase(
        input=question,
        actual_output=result.answer,
        expected_output=expected_output,
        retrieval_context=result.retrieval_context,
        context=context,
    )


def tool_calls_from(result: AgentResult) -> list[ToolCall]:
    return to_tool_calls(result.tool_calls)


def build_agent_test_case(
    agent: ClaimsAgent,
    question: str,
    expected_tools: list[ToolCall] | None = None,
    expected_output: str | None = None,
) -> LLMTestCase:
    """Run the agent and capture its answer plus the tools it called (as ToolCall objects)."""
    result = agent.run(question)
    return LLMTestCase(
        input=question,
        actual_output=result.answer,
        expected_output=expected_output,
        tools_called=tool_calls_from(result),
        expected_tools=expected_tools,
    )


def build_chat_test_case(chatbot: ClaimsChatbot, message: str) -> LLMTestCase:
    """Single-turn view of the chatbot (used by the safety metrics)."""
    chatbot.reset()
    return LLMTestCase(input=message, actual_output=chatbot.reply(message))


def build_conversation(
    chatbot: ClaimsChatbot,
    user_messages: Iterable[str],
    scenario: str | None = None,
    expected_outcome: str | None = None,
) -> ConversationalTestCase:
    """Play the user's messages against a fresh chatbot and record every turn."""
    chatbot.reset()
    turns: list[Turn] = []
    for message in user_messages:
        turns.append(Turn(role="user", content=message))
        turns.append(Turn(role="assistant", content=chatbot.reply(message)))
    return ConversationalTestCase(
        turns=turns,
        chatbot_role=CHATBOT_ROLE,
        scenario=scenario,
        expected_outcome=expected_outcome,
    )


def golden_to_test_case(rag: RAGPipeline, golden: Golden) -> LLMTestCase:
    """A Golden holds the question + ideal answer; running the app turns it into a test case."""
    return build_rag_test_case(rag, golden.input, expected_output=golden.expected_output)
