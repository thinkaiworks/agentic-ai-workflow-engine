# Agentic AI Workflow Engine

**AI agents that do real work in your systems, with the brakes built in.** A small, readable framework for agents that look things up, make decisions and take actions through your APIs, while a human approves anything risky and hard limits stop runaway costs.

No 50,000-line framework to learn. The core is ~250 lines of Python you can read in ten minutes and hand to any developer.

Built by [ThinkAIWorks](https://thinkaiworks.com).

## What's in the box

| Feature | Why it matters |
|---|---|
| **Any Python function becomes a tool** with `@tool`. The schema comes from your type hints | Connect CRMs, databases, Shopify, internal APIs in minutes |
| **Human approval** on risky tools (`@tool(requires_approval=True)`) | Refunds, emails and deletes never happen without a yes |
| **Step and token budgets** | A confused agent can't loop forever or run up a bill |
| **Errors go back to the model** instead of crashing | The agent can recover, retry differently or escalate |
| **Full trace** of every thought, tool call, result and approval | Debug it, audit it, show it to the client |
| **Multi-agent delegation** with `as_tool(agent)` | A supervisor hands work to specialist agents |
| **Claude and OpenAI** behind one interface | Switch models with an environment variable |

## Example: support agent that can refund, safely

```python
@tool
def lookup_order(order_id: str) -> dict:
    """Get order details: item, total, status and delivery date."""
    ...

@tool(requires_approval=True)
def issue_refund(order_id: str, amount: float, reason: str) -> dict:
    """Refund an order to the original payment method. Needs human approval."""
    ...

agent = Agent("support", INSTRUCTIONS, collect(lookup_order, search_policy, issue_refund, create_ticket), llm)
result = agent.run("Order 1042 arrived broken, I want my money back")
```

Run it:

```bash
python -m examples.support_agent "Order 1042 arrived broken, I want my money back"
```

```
step 1 tool_call   lookup_order {'order_id': '1042'}
step 2 tool_call   search_policy {'topic': 'refund'}
step 3 thought     Delivered 2 days ago, damaged, $89: eligible for a full refund.
step 3 tool_call   issue_refund {'order_id': '1042', 'amount': 89.0, ...}
  APPROVAL NEEDED → issue_refund(...) [y/N] y
step 4 final       I'm sorry your vase arrived broken. I've refunded the full $89...
```

Without an API key the examples replay a scripted run, so you can see the loop offline. Add `ANTHROPIC_API_KEY` or `OPENAI_API_KEY` to run them live.

## Multi-agent

```bash
python -m examples.multi_agent "Write a 3-bullet brief on our Q3 churn"
```

A supervisor delegates fact-finding to a **researcher** agent (which has data tools) and the write-up to a **writer** agent (which has none, so it can't invent numbers).

## Install

```bash
pip install -r requirements.txt
pytest -q
```

## Project layout

```
agentflow/
  tools.py   @tool decorator → JSON schema from type hints
  llm.py     Claude, OpenAI and ScriptedLLM behind one step() interface
  agent.py   the loop: budgets, approvals, error handling, trace, delegation
examples/
  support_agent.py   orders + policy + approval-gated refunds
  multi_agent.py     supervisor → researcher + writer
```

## Stack

Python · Claude tool use · OpenAI function calling · plain HTTP (no SDK lock-in)
