from typing import Literal

import pytest

from agentflow import Agent, ClaudeLLM, ScriptedLLM, collect, tool
from examples import multi_agent, support_agent


@tool
def add(a: int, b: int = 0) -> int:
    """Add two numbers.
    a: first number
    b: second number"""
    return a + b


@tool
def boom() -> str:
    """Always fails."""
    raise RuntimeError("database down")


def test_schema_from_type_hints():
    s = add.tool.schema
    assert s["properties"]["a"] == {"type": "integer", "description": "first number"}
    assert s["required"] == ["a"]

    @tool
    def pick(color: Literal["red", "blue"], tags: list[str]) -> str:
        """Pick."""
        return color

    assert pick.tool.schema["properties"]["color"]["enum"] == ["red", "blue"]
    assert pick.tool.schema["properties"]["tags"] == {"type": "array", "items": {"type": "string"}}


def test_loop_feeds_tool_results_back():
    llm = ScriptedLLM([{"tool_calls": [{"name": "add", "input": {"a": 2, "b": 3}}]}, {"text": "It's 5."}])
    r = Agent("calc", "Do maths.", collect(add), llm).run("2+3?")
    assert r.output == "It's 5." and r.stopped_reason == "completed"
    tool_msg = llm.seen[1][-1]
    assert tool_msg["role"] == "tool" and tool_msg["results"][0]["content"] == "5"


def test_tool_errors_and_unknown_tools_dont_crash():
    llm = ScriptedLLM([{"tool_calls": [{"name": "boom", "input": {}}, {"name": "nope", "input": {}}]}, {"text": "Sorry."}])
    r = Agent("x", "", collect(boom), llm).run("go")
    results = [e.data for e in r.trace if e.kind == "tool_result"]
    assert results[0]["is_error"] and "database down" in results[0]["content"]
    assert "unknown tool" in results[1]["content"]
    assert r.output == "Sorry."


def test_step_limit():
    looping = ScriptedLLM([{"tool_calls": [{"name": "add", "input": {"a": 1}}]}] * 10)
    r = Agent("x", "", collect(add), looping, max_steps=3).run("loop")
    assert r.stopped_reason == "max_steps" and r.steps == 3


def test_token_budget():
    looping = ScriptedLLM([{"tool_calls": [{"name": "add", "input": {"a": 1}}]}] * 10)
    r = Agent("x", "", collect(add), looping, max_tokens=250).run("loop")
    assert r.stopped_reason == "budget"


def test_refund_needs_approval():
    support_agent.REFUNDS.clear()
    declined = support_agent.build(llm=ScriptedLLM(support_agent.SCRIPT), approver=lambda n, a: False).run("refund 1042")
    assert support_agent.REFUNDS == []
    assert any(e.kind == "approval" and e.data["approved"] is False for e in declined.trace)

    approved = support_agent.build(llm=ScriptedLLM(support_agent.SCRIPT), approver=lambda n, a: True).run("refund 1042")
    assert support_agent.REFUNDS == [{"order_id": "1042", "amount": 89.0, "reason": "Arrived damaged"}]
    assert approved.tool_calls() == ["lookup_order", "search_policy", "issue_refund"]


def test_supervisor_delegates(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    r = multi_agent.build().run("brief on churn")
    assert r.tool_calls() == ["ask_researcher", "ask_writer"]
    assert "4.1%" in r.output


def test_claude_message_translation():
    m = ClaudeLLM._to_claude({"role": "tool", "results": [{"id": "t1", "content": "5", "is_error": False}]})
    assert m == {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "5", "is_error": False}]}
    a = ClaudeLLM._to_claude({"role": "assistant", "text": "hi", "tool_calls": [{"id": "t1", "name": "add", "input": {"a": 1}}]})
    assert [b["type"] for b in a["content"]] == ["text", "tool_use"]
