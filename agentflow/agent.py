"""The agent loop: think -> call tools -> observe -> repeat, with guard rails.

Guard rails built in:
- max_steps and max_tokens budgets so a confused agent can't run up a bill
- human approval for risky tools (refunds, emails, deletes)
- tool errors are fed back to the model instead of crashing the run
- a full trace of every step for debugging and audit
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Callable

from .llm import LLM
from .tools import Tool

Approver = Callable[[str, dict], bool]


def deny_all(name: str, args: dict) -> bool:
    return False


@dataclass
class TraceEvent:
    step: int
    kind: str  # thought | tool_call | tool_result | approval | final | stop
    data: dict
    ms: int = 0


@dataclass
class RunResult:
    output: str
    stopped_reason: str  # completed | max_steps | budget
    steps: int
    tokens: dict
    trace: list[TraceEvent] = field(default_factory=list)

    def tool_calls(self) -> list[str]:
        return [e.data["name"] for e in self.trace if e.kind == "tool_call"]


class Agent:
    def __init__(self, name: str, instructions: str, tools: dict[str, Tool], llm: LLM,
                 max_steps: int = 8, max_tokens: int = 50_000, approver: Approver = deny_all):
        self.name, self.instructions, self.tools, self.llm = name, instructions, tools, llm
        self.max_steps, self.max_tokens, self.approver = max_steps, max_tokens, approver

    def run(self, task: str) -> RunResult:
        messages: list[dict] = [{"role": "user", "text": task}]
        trace: list[TraceEvent] = []
        tokens = {"input": 0, "output": 0}
        for step in range(1, self.max_steps + 1):
            t0 = time.perf_counter()
            resp = self.llm.step(self.instructions, messages, list(self.tools.values()))
            tokens["input"] += resp["usage"]["input"]
            tokens["output"] += resp["usage"]["output"]
            if resp["text"]:
                trace.append(TraceEvent(step, "thought", {"text": resp["text"]}, int((time.perf_counter() - t0) * 1000)))
            if not resp["tool_calls"]:
                trace.append(TraceEvent(step, "final", {"text": resp["text"]}))
                return RunResult(resp["text"], "completed", step, tokens, trace)

            messages.append({"role": "assistant", "text": resp["text"], "tool_calls": resp["tool_calls"]})
            results = [self._execute(step, call, trace) for call in resp["tool_calls"]]
            messages.append({"role": "tool", "results": results})

            if tokens["input"] + tokens["output"] > self.max_tokens:
                trace.append(TraceEvent(step, "stop", {"reason": "budget"}))
                return RunResult("Stopped: token budget reached.", "budget", step, tokens, trace)

        trace.append(TraceEvent(self.max_steps, "stop", {"reason": "max_steps"}))
        return RunResult("Stopped: step limit reached before finishing.", "max_steps", self.max_steps, tokens, trace)

    def _execute(self, step: int, call: dict, trace: list[TraceEvent]) -> dict:
        name, args = call["name"], call["input"]
        trace.append(TraceEvent(step, "tool_call", {"name": name, "input": args}))
        t = self.tools.get(name)
        if t is None:
            return self._result(step, call, f"Error: unknown tool '{name}'. Available: {', '.join(self.tools)}", True, trace)
        if t.requires_approval:
            ok = self.approver(name, args)
            trace.append(TraceEvent(step, "approval", {"name": name, "approved": ok}))
            if not ok:
                return self._result(step, call, "A human declined this action. Do not retry it; tell the user it needs manual review.", True, trace)
        t0 = time.perf_counter()
        try:
            out = t.run(**args)
            content = out if isinstance(out, str) else json.dumps(out, default=str)
            return self._result(step, call, content, False, trace, int((time.perf_counter() - t0) * 1000))
        except Exception as exc:
            return self._result(step, call, f"Error: {exc}", True, trace)

    @staticmethod
    def _result(step, call, content, is_error, trace, ms=0):
        trace.append(TraceEvent(step, "tool_result", {"name": call["name"], "content": content[:2000], "is_error": is_error}, ms))
        return {"id": call["id"], "content": content, "is_error": is_error}


def as_tool(agent: Agent, description: str) -> Tool:
    """Expose a whole agent as a tool, so a supervisor agent can delegate to specialists."""

    def delegate(task: str) -> str:
        return agent.run(task).output

    return Tool(name=f"ask_{agent.name}", description=description, fn=delegate,
                schema={"type": "object", "properties": {"task": {"type": "string", "description": "What the specialist should do"}}, "required": ["task"]})
