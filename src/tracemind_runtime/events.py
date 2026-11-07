"""Per-run execution events, separate from model messages and transport streaming."""

from collections.abc import Awaitable, Callable
from copy import deepcopy
from dataclasses import dataclass
from typing import Literal

from .protocol import Json

type EventKind = Literal[
    "run_started",
    "model_started",
    "model_completed",
    "operation_requested",
    "approval_requested",
    "approval_resolved",
    "tool_started",
    "operation_completed",
    "run_finished",
]


@dataclass(frozen=True, slots=True)
class RunEvent:
    run_id: str
    sequence: int
    kind: EventKind
    step: int
    data: dict[str, Json]


type EventSink = Callable[[RunEvent], Awaitable[None]]


class EventEmitter:
    def __init__(self, run_id: str, sink: EventSink | None) -> None:
        self.run_id = run_id
        self._sink = sink
        self._sequence = 0

    async def emit(self, kind: EventKind, step: int = 0, **data: Json) -> None:
        if self._sink is not None:
            self._sequence += 1
            await self._sink(
                RunEvent(
                    self.run_id,
                    self._sequence,
                    kind,
                    step,
                    deepcopy(data),
                )
            )
