"""Supervisor + specialists: a manager agent delegates research and writing to two sub-agents.

    python -m examples.multi_agent "Write a 3-bullet brief on our Q3 churn"
"""
from __future__ import annotations

import sys

from agentflow import Agent, ScriptedLLM, as_tool, collect, default_llm, tool

METRICS = {"q3_churn_pct": 4.1, "q2_churn_pct": 3.2, "top_reason": "missing integrations", "accounts_lost": 37}


@tool
def get_metric(name: str) -> str:
    """Read a business metric. Known: q3_churn_pct, q2_churn_pct, top_reason, accounts_lost."""
    return str(METRICS[name])


def build() -> Agent:
    live = default_llm()
    researcher = Agent("researcher", "Find the facts needed using your tools. Reply with facts only.", collect(get_metric),
                       live or ScriptedLLM([
                           {"tool_calls": [{"name": "get_metric", "input": {"name": "q3_churn_pct"}}, {"name": "get_metric", "input": {"name": "q2_churn_pct"}},
                                           {"name": "get_metric", "input": {"name": "top_reason"}}]},
                           {"text": "Q3 churn 4.1% (Q2 3.2%). Top reason: missing integrations."}]))
    writer = Agent("writer", "Turn facts into crisp executive bullet points. No new facts.", {},
                   live or ScriptedLLM([{"text": "• Churn rose to 4.1% in Q3 (from 3.2%).\n• Main driver: missing integrations.\n• Next: prioritise the top 3 integration requests."}]))
    supervisor_tools = {t.name: t for t in (as_tool(researcher, "Gathers facts and numbers"), as_tool(writer, "Writes polished text from facts"))}
    return Agent("supervisor", "Break the task down. Use the researcher for facts, then the writer for the final text. Return the writer's text.",
                 supervisor_tools,
                 live or ScriptedLLM([
                     {"tool_calls": [{"name": "ask_researcher", "input": {"task": "Q3 vs Q2 churn and the top reason"}}]},
                     {"tool_calls": [{"name": "ask_writer", "input": {"task": "3 bullets: Q3 churn 4.1% vs 3.2% Q2, top reason missing integrations"}}]},
                     {"text": "• Churn rose to 4.1% in Q3 (from 3.2%).\n• Main driver: missing integrations.\n• Next: prioritise the top 3 integration requests."}]))


if __name__ == "__main__":
    result = build().run(" ".join(sys.argv[1:]) or "Write a 3-bullet brief on our Q3 churn")
    print("Delegations:", result.tool_calls())
    print(result.output)
