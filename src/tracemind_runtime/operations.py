"""Registered tools, schema validation and explicit per-call approval."""

import asyncio
import json
from collections.abc import Awaitable, Callable, Iterable
from copy import deepcopy
from dataclasses import dataclass
from typing import Literal

from jsonschema import Draft202012Validator
from referencing import Registry

from .events import EventEmitter
from .protocol import Json, Message, ToolCall, ToolDefinition, validate_json

type ToolHandler = Callable[[dict[str, Json]], Awaitable[Json]]
type Approver = Callable[[ToolCall], Awaitable[bool]]
type OutcomeStatus = Literal[
    "ok",
    "unknown_tool",
    "invalid_arguments",
    "denied",
    "approval_failed",
    "approval_timeout",
    "failed",
    "timeout",
    "output_too_large",
]


@dataclass(frozen=True, slots=True)
class Tool:
    definition: ToolDefinition
    handler: ToolHandler
    requires_approval: bool = True


@dataclass(frozen=True, slots=True)
class ToolOutcome:
    call_id: str
    name: str
    status: OutcomeStatus
    content: str

    def message(self) -> Message:
        return Message("tool", self.content, tool_call_id=self.call_id)


class Operations:
    def __init__(self, tools: Iterable[Tool]) -> None:
        self._tools: dict[str, Tool] = {}
        self._validators: dict[str, Draft202012Validator] = {}
        for tool in tools:
            name = tool.definition.name
            if name in self._tools:
                raise ValueError(f"Duplicate tool name: {name}")
            if type(tool.requires_approval) is not bool:
                raise ValueError("requires_approval must be a boolean")
            definition = deepcopy(tool.definition)
            if definition.parameters.get("type") != "object":
                raise ValueError("Tool parameter schemas must describe objects")
            Draft202012Validator.check_schema(definition.parameters)
            self._validators[name] = Draft202012Validator(
                definition.parameters, registry=Registry()
            )
            self._tools[name] = Tool(definition, tool.handler, tool.requires_approval)

    def definitions(self) -> tuple[ToolDefinition, ...]:
        return deepcopy(tuple(tool.definition for tool in self._tools.values()))

    async def execute(
        self,
        call: ToolCall,
        *,
        approve: Approver | None,
        timeout: float,
        approval_timeout: float,
        result_limit: int,
        events: EventEmitter,
        step: int,
    ) -> ToolOutcome:
        def failure(status: OutcomeStatus, message: str) -> ToolOutcome:
            return ToolOutcome(
                call.id,
                call.name,
                status,
                json.dumps(
                    {
                        "status": status,
                        "message": message,
                    }
                ),
            )

        tool = self._tools.get(call.name)
        if tool is None:
            return failure("unknown_tool", "This tool is not registered.")
        arguments = deepcopy(call.arguments)
        try:
            self._validators[call.name].validate(arguments)
        except Exception:
            return failure("invalid_arguments", "Arguments do not match the tool schema.")

        if tool.requires_approval:
            await events.emit("approval_requested", step, call_id=call.id, name=call.name)
            if approve is None:
                await events.emit("approval_resolved", step, call_id=call.id, approved=False)
                return failure("denied", "This operation requires host approval.")
            try:
                async with asyncio.timeout(approval_timeout):
                    permitted = await approve(deepcopy(call))
            except TimeoutError:
                await events.emit("approval_resolved", step, call_id=call.id, approved=False)
                return failure("approval_timeout", "Approval expired; no tool was executed.")
            except Exception:
                await events.emit("approval_resolved", step, call_id=call.id, approved=False)
                return failure("approval_failed", "Approval failed; no tool was executed.")
            await events.emit(
                "approval_resolved",
                step,
                call_id=call.id,
                approved=permitted is True,
            )
            if permitted is not True:
                return failure("denied", "The host did not approve this operation.")

        await events.emit("tool_started", step, call_id=call.id, name=call.name)
        try:
            async with asyncio.timeout(timeout):
                value = await tool.handler(arguments)
            validate_json(value)
            content = json.dumps({"status": "ok", "result": value}, allow_nan=False)
        except TimeoutError:
            return failure(
                "timeout",
                "The tool timed out. External effects may have occurred; "
                "check external state before retrying.",
            )
        except Exception:
            return failure(
                "failed",
                "The tool failed. External effects may have occurred; "
                "check external state before retrying.",
            )
        if len(content) > result_limit:
            return failure(
                "output_too_large",
                "The tool finished but its output exceeded the "
                "result limit. Do not repeat side effects just to retrieve output.",
            )
        return ToolOutcome(call.id, call.name, "ok", content)
