"""Model backends speaking one shared format.

Every backend's `step(system, messages, tools)` returns:
    {"text": str, "tool_calls": [{"id": str, "name": str, "input": dict}], "usage": {"input": int, "output": int}}
Messages use a provider-neutral shape and are translated per provider.
"""
from __future__ import annotations

import json
import os
from typing import Protocol

import requests

from .tools import Tool


class LLM(Protocol):
    def step(self, system: str, messages: list[dict], tools: list[Tool]) -> dict: ...


class ClaudeLLM:
    def __init__(self, model: str | None = None, session=None):
        self.model = model or os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-5")
        self.s = session or requests.Session()

    def step(self, system, messages, tools):
        body = {
            "model": self.model, "max_tokens": 1500, "system": system,
            "tools": [{"name": t.name, "description": t.description, "input_schema": t.schema} for t in tools],
            "messages": [self._to_claude(m) for m in messages],
        }
        r = self.s.post("https://api.anthropic.com/v1/messages", json=body, timeout=120, headers={
            "x-api-key": os.environ["ANTHROPIC_API_KEY"], "anthropic-version": "2023-06-01", "content-type": "application/json"})
        r.raise_for_status()
        d = r.json()
        return {
            "text": "".join(b["text"] for b in d["content"] if b["type"] == "text"),
            "tool_calls": [{"id": b["id"], "name": b["name"], "input": b["input"]} for b in d["content"] if b["type"] == "tool_use"],
            "usage": {"input": d["usage"]["input_tokens"], "output": d["usage"]["output_tokens"]},
        }

    @staticmethod
    def _to_claude(m):
        if m["role"] == "assistant":
            content = ([{"type": "text", "text": m["text"]}] if m.get("text") else []) + [
                {"type": "tool_use", "id": c["id"], "name": c["name"], "input": c["input"]} for c in m.get("tool_calls", [])]
            return {"role": "assistant", "content": content}
        if m["role"] == "tool":
            return {"role": "user", "content": [{"type": "tool_result", "tool_use_id": r["id"], "content": r["content"], "is_error": r.get("is_error", False)} for r in m["results"]]}
        return {"role": "user", "content": m["text"]}


class OpenAILLM:
    def __init__(self, model: str | None = None, session=None):
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.s = session or requests.Session()

    def step(self, system, messages, tools):
        msgs = [{"role": "system", "content": system}]
        for m in messages:
            if m["role"] == "assistant":
                msgs.append({"role": "assistant", "content": m.get("text") or None, "tool_calls": [
                    {"id": c["id"], "type": "function", "function": {"name": c["name"], "arguments": json.dumps(c["input"])}} for c in m.get("tool_calls", [])] or None})
            elif m["role"] == "tool":
                msgs += [{"role": "tool", "tool_call_id": r["id"], "content": r["content"]} for r in m["results"]]
            else:
                msgs.append({"role": "user", "content": m["text"]})
        body = {"model": self.model, "messages": msgs, "tools": [
            {"type": "function", "function": {"name": t.name, "description": t.description, "parameters": t.schema}} for t in tools]}
        r = self.s.post("https://api.openai.com/v1/chat/completions", json=body, timeout=120,
                        headers={"authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"})
        r.raise_for_status()
        d = r.json()
        msg = d["choices"][0]["message"]
        return {
            "text": msg.get("content") or "",
            "tool_calls": [{"id": c["id"], "name": c["function"]["name"], "input": json.loads(c["function"]["arguments"] or "{}")} for c in msg.get("tool_calls") or []],
            "usage": {"input": d["usage"]["prompt_tokens"], "output": d["usage"]["completion_tokens"]},
        }


class ScriptedLLM:
    """Deterministic stand-in for demos and tests: replays a list of pre-written steps."""

    def __init__(self, steps: list[dict]):
        self.steps = list(steps)
        self.seen: list[list[dict]] = []

    def step(self, system, messages, tools):
        self.seen.append(list(messages))
        if not self.steps:
            return {"text": "Done.", "tool_calls": [], "usage": {"input": 0, "output": 0}}
        s = self.steps.pop(0)
        calls = [{"id": f"call_{len(self.seen)}_{i}", **c} for i, c in enumerate(s.get("tool_calls", []))]
        return {"text": s.get("text", ""), "tool_calls": calls, "usage": {"input": 100, "output": 20}}


def default_llm() -> LLM | None:
    if os.getenv("ANTHROPIC_API_KEY"):
        return ClaudeLLM()
    if os.getenv("OPENAI_API_KEY"):
        return OpenAILLM()
    return None
