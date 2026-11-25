"""Tool-calling claims agent.

    question --> planner (LLM) --> tool calls --> composer (LLM) --> final answer

Returns the final answer AND every tool call (name + arguments + output), so tests can build
DeepEval `ToolCall` objects and compare them with the tools they expected.

Stub mode: the plan and tool calls come from app/stubs.py (some deliberately wrong), but the
tools themselves really run against the in-memory data in app/tools.py.
Live mode: OpenAI function calling drives the loop.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from deepeval.tracing import observe, update_agent_span, update_current_trace, update_llm_span

from app import stubs
from app.tools import TOOL_REGISTRY, TOOL_SCHEMAS
from app.tracing_hooks import attach, to_tool_calls


@dataclass
class ToolRecord:
    name: str
    arguments: dict[str, Any]
    output: Any = None


@dataclass
class AgentResult:
    question: str
    answer: str
    tool_calls: list[ToolRecord] = field(default_factory=list)
    plan: str = ""


class ClaimsAgent:
    def __init__(self, config):
        self.config = config
        self.mode = config.app_mode
        self.model = config.live_model
        self.max_steps = config.get("app.agent_max_steps", 5)

    # ---- one tool execution (one TOOL span) -------------------------------------
    @observe(type="tool")
    def call_tool(self, name: str, arguments: dict[str, Any]) -> ToolRecord:
        func = TOOL_REGISTRY.get(name)
        output = func(**arguments) if func else {"error": f"Unknown tool {name}"}
        record = ToolRecord(name=name, arguments=arguments, output=output)
        attach("tool", input=f"{name}({arguments})", actual_output=json.dumps(output))
        return record

    # ---- stub planning ----------------------------------------------------------
    @observe(type="llm")
    def _stub_plan(self, question: str) -> dict[str, Any]:
        script = stubs.AGENT_SCRIPTS.get(stubs.normalize(question))
        if script is None:  # unknown question: pick a tool from ids found in the text
            calls: list[tuple[str, dict[str, Any]]] = []
            if (m := re.search(r"CLM-\d{4}", question)):
                calls.append(("check_claim_status", {"claim_id": m.group()}))
            elif (m := re.search(r"POL-\d{6}", question)):
                calls.append(("lookup_policy", {"policy_id": m.group()}))
            answer = stubs.CLAIMS_ASSISTANT_REFUSAL if not calls else "Here is what I found."
            script = {"plan": "Use a tool if an id is mentioned, otherwise explain my limits.",
                      "calls": calls, "answer": answer}
        update_llm_span(model="stub-planner")
        attach("planner", input=question, actual_output=script["plan"])
        return script

    # ---- live loop --------------------------------------------------------------
    @observe(type="llm")
    def _live_step(self, messages: list[dict[str, Any]]):
        from app.llm import chat

        update_llm_span(model=self.model)
        message = chat(messages, model=self.model, tools=TOOL_SCHEMAS, temperature=0)
        attach("planner", input=str(messages[-1].get("content")), actual_output=message.content or "tool call")
        return message

    def _run_live(self, question: str) -> AgentResult:
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": "You are a claims agent. Use the tools to look up policies, "
             "check claim status and estimate payouts. Never promise a payout; payouts are estimates."},
            {"role": "user", "content": question},
        ]
        records: list[ToolRecord] = []
        plan = ""
        for _ in range(self.max_steps):
            message = self._live_step(messages)
            if not message.tool_calls:
                return AgentResult(question, message.content or "", records, plan)
            plan += (message.content or "") + " "
            messages.append(message.model_dump(exclude_none=True))
            for tc in message.tool_calls:
                record = self.call_tool(tc.function.name, json.loads(tc.function.arguments))
                records.append(record)
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": json.dumps(record.output)})
        return AgentResult(question, "I could not finish within the step limit.", records, plan)

    # ---- entry point (the AGENT span) ------------------------------------------
    @observe(type="agent")
    def run(self, question: str) -> AgentResult:
        update_agent_span(available_tools=list(TOOL_REGISTRY))
        if self.mode == "stub":
            script = self._stub_plan(question)
            records = [self.call_tool(name, dict(args)) for name, args in script["calls"]]
            result = AgentResult(question, script["answer"], records, script["plan"])
        else:
            result = self._run_live(question)
        tools_called = to_tool_calls(result.tool_calls)
        attach("agent", input=question, actual_output=result.answer, tools_called=tools_called)
        update_current_trace(input=question, output=result.answer, tools_called=tools_called)
        return result
