"""Versioned history snapshots. These are conversation data, not execution checkpoints."""

import json
from typing import Any, cast

from .protocol import Message, Role, ToolCall, require_text
from .turn import RunStatus

_MAX_BYTES = 4 * 1024 * 1024
_MAX_MESSAGES = 10_000


def _validate(messages: tuple[Message, ...]) -> None:
    if not messages or len(messages) > _MAX_MESSAGES or messages[0].role != "system":
        raise ValueError("History must start with one system message and fit the message limit")
    require_text(messages[0].content, "System prompt")
    pending: set[str] = set()
    used: set[str] = set()
    for index, message in enumerate(messages):
        if message.role == "system":
            if index != 0:
                raise ValueError("System messages are allowed only at the start")
        elif message.role == "tool":
            if message.tool_call_id not in pending:
                raise ValueError("Tool results must match an unanswered request")
            pending.remove(message.tool_call_id)
        else:
            if pending:
                raise ValueError("A tool batch is incomplete")
            if message.role == "user" or not message.tool_calls:
                require_text(message.content, "Message content")
            for call in message.tool_calls:
                if call.id in used:
                    raise ValueError("A tool call ID was reused")
                used.add(call.id)
                pending.add(call.id)
    if pending:
        raise ValueError("A tool batch is incomplete")


def encode_history(messages: tuple[Message, ...], status: RunStatus) -> str:
    if status == RunStatus.RUNNING:
        raise ValueError("A running session cannot be exported")
    _validate(messages)
    data = {
        "version": 1,
        "status": status.value,
        "messages": [
            {
                "role": message.role,
                "content": message.content,
                "tool_calls": [
                    {"id": call.id, "name": call.name, "arguments": call.arguments}
                    for call in message.tool_calls
                ],
                "tool_call_id": message.tool_call_id,
            }
            for message in messages
        ],
    }
    text = json.dumps(data, ensure_ascii=False, allow_nan=False)
    if len(text.encode("utf-8")) > _MAX_BYTES:
        raise ValueError("History exceeds the snapshot size limit")
    return text


def _object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON keys are not accepted")
        result[key] = value
    return result


def _constant(_: str) -> None:
    raise ValueError("Non-finite JSON constants are not accepted")


def decode_history(text: str) -> tuple[tuple[Message, ...], RunStatus]:
    if not isinstance(text, str) or len(text.encode("utf-8")) > _MAX_BYTES:
        raise ValueError("History exceeds the snapshot size limit")
    try:
        data = json.loads(text, parse_constant=_constant, object_pairs_hook=_object)
    except (RecursionError, json.JSONDecodeError) as error:
        raise ValueError("Invalid history JSON") from error
    if not isinstance(data, dict) or set(data) != {"version", "status", "messages"}:
        raise ValueError("Unexpected history envelope")
    if type(data["version"]) is not int or data["version"] != 1:
        raise ValueError("Unsupported history format version")
    if not isinstance(data["status"], str):
        raise ValueError("Invalid history status")
    status = RunStatus(data["status"])
    if status == RunStatus.RUNNING:
        raise ValueError("Running state is not recoverable from a history snapshot")
    rows = data["messages"]
    if not isinstance(rows, list) or not rows or len(rows) > _MAX_MESSAGES:
        raise ValueError("Invalid history message list")
    messages = []
    for row in rows:
        if (
            not isinstance(row, dict)
            or set(row)
            != {
                "role",
                "content",
                "tool_calls",
                "tool_call_id",
            }
            or not isinstance(row["tool_calls"], list)
        ):
            raise ValueError("Unexpected history message fields")
        calls = []
        for call in row["tool_calls"]:
            if not isinstance(call, dict) or set(call) != {"id", "name", "arguments"}:
                raise ValueError("Unexpected tool call fields")
            calls.append(ToolCall(call["id"], call["name"], call["arguments"]))
        messages.append(
            Message(
                cast(Role, row["role"]),
                row["content"],
                tuple(calls),
                row["tool_call_id"],
            )
        )
    result = tuple(messages)
    _validate(result)
    return result, status
