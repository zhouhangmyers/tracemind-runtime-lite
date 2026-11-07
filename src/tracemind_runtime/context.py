"""Append-only, in-memory conversation history and tool-message integrity."""

import json
from copy import deepcopy

from .history import decode_history, encode_history
from .protocol import Message, require_text
from .turn import RunStatus, TurnState

_INTERRUPTED_RESULT = json.dumps(
    {
        "status": "interrupted",
        "message": (
            "No result was recorded. Execution may be incomplete; "
            "check external state before repeating side effects."
        ),
    }
)


class Session:
    def __init__(self, *, system_prompt: str = "You are a helpful assistant.") -> None:
        require_text(system_prompt, "System prompt")
        self._history = [Message("system", system_prompt)]
        self._turn = TurnState()

    @property
    def status(self) -> RunStatus:
        return self._turn.status

    def snapshot(self) -> tuple[Message, ...]:
        """Mutable tool arguments never alias the authoritative history."""
        return deepcopy(tuple(self._history))

    def to_json(self) -> str:
        """Export terminal conversation data; no approvals, tasks or run state are restored."""
        return encode_history(self.snapshot(), self.status)

    @classmethod
    def from_json(cls, text: str) -> "Session":
        """Load history from a trusted source. Structural validation is not authentication."""
        messages, status = decode_history(text)
        session = cls(system_prompt=messages[0].content)
        session._history = list(messages)
        session._turn.status = status
        return session

    def _append(self, message: Message) -> None:
        self._history.append(deepcopy(message))

    def _call_ids(self) -> set[str]:
        return {call.id for message in self._history for call in message.tool_calls}

    def _close_pending(self) -> None:
        """An interrupted batch must not leave unmatched calls in the next request."""
        answered: set[str] = set()
        for message in reversed(self._history):
            if message.role == "tool":
                answered.add(message.tool_call_id or "")
                continue
            if message.role == "assistant":
                for call in message.tool_calls:
                    if call.id not in answered:
                        self._append(
                            Message(
                                "tool",
                                _INTERRUPTED_RESULT,
                                tool_call_id=call.id,
                            )
                        )
            break
