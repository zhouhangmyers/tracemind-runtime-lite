"""Small public message protocol; independent of any model provider."""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass
from typing import Literal

type Json = None | bool | int | float | str | list[Json] | dict[str, Json]
type Role = Literal["system", "user", "assistant", "tool"]


def require_text(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be nonblank text")


def validate_json(value: Json) -> None:
    """Reject Python objects, non-string keys and non-finite numbers."""
    pending = [value]
    seen: set[int] = set()
    while pending:
        item = pending.pop()
        if isinstance(item, (dict, list)):
            if id(item) in seen:
                continue
            seen.add(id(item))
        if isinstance(item, dict):
            if any(not isinstance(key, str) for key in item):
                raise ValueError("JSON object keys must be strings")
            pending.extend(item.values())
        elif isinstance(item, list):
            pending.extend(item)
        elif item is not None and type(item) not in (bool, int, float, str):
            raise ValueError("Expected JSON data")
    json.dumps(value, allow_nan=False)


@dataclass(frozen=True, slots=True)
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Json]

    def __post_init__(self) -> None:
        require_text(self.id, "Tool call ID")
        require_text(self.name, "Tool name")
        if not isinstance(self.arguments, dict):
            raise ValueError("Tool arguments must be an object")
        validate_json(self.arguments)
        object.__setattr__(self, "arguments", deepcopy(self.arguments))


@dataclass(frozen=True, slots=True)
class Message:
    role: Role
    content: str
    tool_calls: tuple[ToolCall, ...] = ()
    tool_call_id: str | None = None

    def __post_init__(self) -> None:
        if self.role not in ("system", "user", "assistant", "tool"):
            raise ValueError("Invalid message role")
        if not isinstance(self.content, str):
            raise ValueError("Message content must be text")
        if self.tool_calls and self.role != "assistant":
            raise ValueError("Only assistant messages can request tools")
        if any(not isinstance(call, ToolCall) for call in self.tool_calls):
            raise ValueError("Expected ToolCall objects")
        if self.role == "tool":
            require_text(self.tool_call_id or "", "Tool result call ID")
        elif self.tool_call_id is not None:
            raise ValueError("Only tool messages can reference a tool call")
        object.__setattr__(self, "tool_calls", deepcopy(tuple(self.tool_calls)))


@dataclass(frozen=True, slots=True)
class ToolDefinition:
    name: str
    description: str
    parameters: dict[str, Json]

    def __post_init__(self) -> None:
        require_text(self.name, "Tool name")
        require_text(self.description, "Tool description")
        if not isinstance(self.parameters, dict):
            raise ValueError("Tool parameters must be a schema object")
        validate_json(self.parameters)
        object.__setattr__(self, "parameters", deepcopy(self.parameters))


@dataclass(frozen=True, slots=True)
class ModelResponse:
    content: str = ""
    tool_calls: tuple[ToolCall, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.content, str):
            raise ValueError("Model response content must be text")
        if not self.tool_calls and not self.content.strip():
            raise ValueError("Model response must contain text or tool calls")
        if any(not isinstance(call, ToolCall) for call in self.tool_calls):
            raise ValueError("Expected ToolCall objects")
        object.__setattr__(self, "tool_calls", deepcopy(tuple(self.tool_calls)))
