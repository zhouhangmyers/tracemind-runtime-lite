"""Sequential model/tool loop. No transport retry or durable execution machinery."""

import asyncio
from dataclasses import dataclass, field
from math import isfinite

from .context import Session
from .events import EventEmitter
from .llm import ChatModel
from .operations import Approver, Operations, ToolOutcome
from .protocol import Message, ModelResponse
from .turn import RunStatus


@dataclass(frozen=True, slots=True)
class RunOptions:
    max_steps: int = 12
    max_tools_per_step: int = 8
    model_timeout: float = 60.0
    tool_timeout: float = 30.0
    approval_timeout: float = 120.0
    result_limit: int = 12_000

    def __post_init__(self) -> None:
        for name in ("max_steps", "max_tools_per_step", "result_limit"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("model_timeout", "tool_timeout", "approval_timeout"):
            value = getattr(self, name)
            if type(value) not in (float, int) or not isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")


class LoopFailure(Exception):
    """Safe diagnostic codes; provider exception strings stay outside results."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


@dataclass(slots=True)
class LoopProgress:
    steps: int = 0
    outcomes: list[ToolOutcome] = field(default_factory=list)


async def run_loop(
    model: ChatModel,
    operations: Operations,
    session: Session,
    progress: LoopProgress,
    options: RunOptions,
    approve: Approver | None,
    events: EventEmitter,
) -> tuple[RunStatus, str]:
    for _ in range(options.max_steps):
        progress.steps += 1
        await events.emit("model_started", progress.steps)
        try:
            async with asyncio.timeout(options.model_timeout):
                reply = await model.complete(session.snapshot(), operations.definitions())
        except TimeoutError:
            raise LoopFailure("model_timeout") from None
        except Exception:
            raise LoopFailure("model_error") from None
        if not isinstance(reply, ModelResponse):
            raise LoopFailure("invalid_response")
        calls = reply.tool_calls
        ids = [call.id for call in calls]
        if (
            len(calls) > options.max_tools_per_step
            or len(set(ids)) != len(ids)
            or set(ids) & session._call_ids()
        ):
            raise LoopFailure("invalid_tool_batch")
        await events.emit(
            "model_completed",
            progress.steps,
            content=reply.content,
            tool_calls=len(calls),
        )
        session._append(Message("assistant", reply.content, calls))
        if not calls:
            return RunStatus.COMPLETED, reply.content
        for call in calls:
            await events.emit(
                "operation_requested",
                progress.steps,
                call_id=call.id,
                name=call.name,
            )
            outcome = await operations.execute(
                call,
                approve=approve,
                timeout=options.tool_timeout,
                approval_timeout=options.approval_timeout,
                result_limit=options.result_limit,
                events=events,
                step=progress.steps,
            )
            session._append(outcome.message())
            progress.outcomes.append(outcome)
            await events.emit(
                "operation_completed",
                progress.steps,
                call_id=call.id,
                name=call.name,
                status=outcome.status,
            )
    return RunStatus.LIMIT_REACHED, ""
