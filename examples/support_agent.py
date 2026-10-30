"""E-commerce support agent: looks up orders, checks policy, and issues refunds only with human approval.

    python -m examples.support_agent "Order 1042 arrived broken, I want my money back"

Uses Claude/OpenAI if a key is set; otherwise replays a scripted run so you can see the loop offline.
"""
from __future__ import annotations

import sys
from typing import Literal

from agentflow import Agent, ScriptedLLM, collect, default_llm, tool

ORDERS = {
    "1042": {"customer": "ana@example.com", "item": "Ceramic vase", "total": 89.0, "status": "delivered", "delivered_days_ago": 2},
    "1043": {"customer": "li@example.com", "item": "Desk lamp", "total": 45.0, "status": "in_transit", "delivered_days_ago": None},
}
POLICY = {
    "refund": "Damaged or defective items can be refunded in full within 30 days of delivery. Photos are recommended but not required under $100.",
    "shipping": "Standard shipping is 3-5 business days. In-transit orders can't be cancelled.",
}
TICKETS: list[dict] = []
REFUNDS: list[dict] = []


@tool
def lookup_order(order_id: str) -> dict:
    """Get order details: item, total, status and delivery date.
    order_id: The order number, digits only"""
    if order_id not in ORDERS:
        raise ValueError(f"No order {order_id}")
    return ORDERS[order_id]


@tool
def search_policy(topic: Literal["refund", "shipping"]) -> str:
    """Read the company policy on a topic before making any decision."""
    return POLICY[topic]


@tool(requires_approval=True)
def issue_refund(order_id: str, amount: float, reason: str) -> dict:
    """Refund an order to the original payment method. Needs human approval.
    order_id: The order number
    amount: Refund amount in USD
    reason: Short reason for the refund"""
    REFUNDS.append({"order_id": order_id, "amount": amount, "reason": reason})
    return {"refunded": amount, "order_id": order_id}


@tool
def create_ticket(summary: str, priority: Literal["low", "normal", "high"] = "normal") -> dict:
    """Escalate to the human support team when you can't resolve something."""
    TICKETS.append({"summary": summary, "priority": priority})
    return {"ticket_id": f"T-{len(TICKETS):04d}"}


INSTRUCTIONS = """You are a customer support agent for an online home-goods store.
Always look up the order and read the relevant policy before deciding anything.
Only refund when the policy allows it. If a refund is declined by a human, create a ticket instead.
Finish with a short, friendly message to the customer."""

SCRIPT = [
    {"text": "Let me check the order first.", "tool_calls": [{"name": "lookup_order", "input": {"order_id": "1042"}}]},
    {"tool_calls": [{"name": "search_policy", "input": {"topic": "refund"}}]},
    {"text": "Delivered 2 days ago, damaged, $89: eligible for a full refund.", "tool_calls": [{"name": "issue_refund", "input": {"order_id": "1042", "amount": 89.0, "reason": "Arrived damaged"}}]},
    {"text": "I'm sorry your vase arrived broken. I've refunded the full $89 to your original payment method; you'll see it in 5-7 business days."},
]


def console_approver(name: str, args: dict) -> bool:
    return input(f"\n  APPROVAL NEEDED → {name}({args}) [y/N] ").strip().lower() == "y"


def build(llm=None, approver=console_approver) -> Agent:
    llm = llm or default_llm() or ScriptedLLM(SCRIPT)
    return Agent("support", INSTRUCTIONS, collect(lookup_order, search_policy, issue_refund, create_ticket), llm, approver=approver)


if __name__ == "__main__":
    task = " ".join(sys.argv[1:]) or "Order 1042 arrived broken, I want my money back"
    result = build().run(task)
    for e in result.trace:
        print(f"  step {e.step} {e.kind:<11} {e.data}")
    print(f"\nFINAL ({result.stopped_reason}, {result.steps} steps): {result.output}")
