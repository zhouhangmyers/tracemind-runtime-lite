"""In-process turn ownership. No persistence or recovery is implied."""

from dataclasses import dataclass
from enum import StrEnum
from uuid import uuid4


class RunStatus(StrEnum):
    IDLE = "idle"
    RUNNING = "running"
    COMPLETED = "completed"
    LIMIT_REACHED = "limit_reached"
    FAILED = "failed"
    CANCELLED = "cancelled"


class SessionBusyError(RuntimeError):
    """A session accepts only one active turn, including across Runtime objects."""


@dataclass(slots=True)
class TurnState:
    status: RunStatus = RunStatus.IDLE
    run_id: str | None = None

    def enter(self) -> str:
        if self.status == RunStatus.RUNNING:
            raise SessionBusyError("This session already has an active turn")
        self.run_id = str(uuid4())
        self.status = RunStatus.RUNNING
        return self.run_id

    def finish(self, status: RunStatus) -> None:
        if status in (RunStatus.IDLE, RunStatus.RUNNING):
            raise ValueError("Expected a terminal status")
        self.status = status
