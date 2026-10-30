"""Turn plain Python functions into LLM tools. The JSON schema is built from type hints and the docstring."""
from __future__ import annotations

import inspect
import typing
from dataclasses import dataclass
from typing import Any, Callable

_PY_TO_JSON = {str: "string", int: "integer", float: "number", bool: "boolean", list: "array", dict: "object"}


@dataclass
class Tool:
    name: str
    description: str
    fn: Callable[..., Any]
    schema: dict
    requires_approval: bool = False

    def run(self, **kwargs) -> Any:
        return self.fn(**kwargs)


def _json_type(annotation) -> dict:
    origin = typing.get_origin(annotation)
    if origin is typing.Literal:
        return {"type": "string", "enum": list(typing.get_args(annotation))}
    if origin in (list, typing.List):
        (item,) = typing.get_args(annotation) or (str,)
        return {"type": "array", "items": _json_type(item)}
    if origin is typing.Union:
        args = [a for a in typing.get_args(annotation) if a is not type(None)]
        return _json_type(args[0]) if args else {"type": "string"}
    return {"type": _PY_TO_JSON.get(annotation, "string")}


def tool(_fn=None, *, requires_approval: bool = False, name: str | None = None):
    """Decorator. `requires_approval=True` pauses the agent for a human yes/no before running."""

    def wrap(fn):
        sig = inspect.signature(fn)
        hints = typing.get_type_hints(fn)
        doc = inspect.getdoc(fn) or ""
        summary, _, rest = doc.partition("\n")
        arg_docs = {}
        for line in rest.splitlines():
            line = line.strip()
            if ":" in line and not line.lower().startswith(("args", "returns")):
                k, _, v = line.partition(":")
                arg_docs[k.strip()] = v.strip()
        props, required = {}, []
        for pname, param in sig.parameters.items():
            spec = _json_type(hints.get(pname, str))
            if pname in arg_docs:
                spec["description"] = arg_docs[pname]
            props[pname] = spec
            if param.default is inspect.Parameter.empty:
                required.append(pname)
        schema = {"type": "object", "properties": props, "required": required}
        t = Tool(name or fn.__name__, summary.strip() or fn.__name__, fn, schema, requires_approval)
        fn.tool = t
        return fn

    return wrap(_fn) if _fn else wrap


def collect(*fns) -> dict[str, Tool]:
    return {f.tool.name: f.tool for f in fns}
