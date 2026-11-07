"""Public entry and composition root for the community edition."""

import asyncio
from collections.abc import AsyncGenerator, Iterable
from contextlib import suppress
from dataclasses import dataclass

from .agent_loop import LoopFailure, LoopProgress, RunOptions, run_loop
from .context import Session
from .events import EventEmitter, EventSink, RunEvent
from .llm import ChatModel
from .operations import Approver, Operations, Tool, ToolOutcome
from .protocol import Message, require_text
from .turn import RunStatus


@dataclass(frozen=True, slots=True)
class RunResult:
    run_id: str
    status: RunStatus
    content: str
    model_steps: int
    operations: tuple[ToolOutcome, ...]
    error: str | None = None


class AgentRuntime:
    def __init__(
        self,
        model: ChatModel,
        *,
        tools: Iterable[Tool] = (),
        options: RunOptions | None = None,
    ) -> None:
        self._model = model
        self._operations = Operations(tools)
        self._options = options if options is not None else RunOptions()

    async def run(
        self,
        session: Session,
        message: str,
        *,
        approve: Approver | None = None,
    ) -> RunResult:
        return await self._run(session, message, approve=approve)

    async def stream(
        self,
        session: Session,
        message: str,
        *,
        approve: Approver | None = None,
    ) -> AsyncGenerator[RunEvent, None]:
        """Yield bounded execution events; close with contextlib.aclosing on early exit.

        This is execution progress, not token streaming. A slow consumer applies
        backpressure. Closing the iterator cancels its producer and waits for cleanup.
        """
        queue: asyncio.Queue[RunEvent] = asyncio.Queue(maxsize=32)
        task = asyncio.create_task(self._run(session, message, approve=approve, sink=queue.put))
        waiter: asyncio.Task[RunEvent] | None = None
        try:
            while True:
                if not queue.empty():
                    yield queue.get_nowait()
                elif task.done():
                    await task
                    break
                else:
                    waiter = asyncio.create_task(queue.get())
                    done, _ = await asyncio.wait(
                        (task, waiter), return_when=asyncio.FIRST_COMPLETED
                    )
                    if waiter in done:
                        event = waiter.result()
                        waiter = None
                        yield event
                    else:
                        waiter.cancel()
                        with suppress(asyncio.CancelledError):
                            await waiter
                        waiter = None
        finally:
            if waiter is not None:
                waiter.cancel()
                with suppress(asyncio.CancelledError):
                    await waiter
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    async def _run(
        self,
        session: Session,
        message: str,
        *,
        approve: Approver | None,
        sink: EventSink | None = None,
    ) -> RunResult:
        require_text(message, "User message")
        run_id = session._turn.enter()
        session._append(Message("user", message))
        progress = LoopProgress()
        events = EventEmitter(run_id, sink)
        try:
            await events.emit("run_started")
            status, content = await run_loop(
                self._model,
                self._operations,
                session,
                progress,
                self._options,
                approve,
                events,
            )
        except asyncio.CancelledError:
            session._close_pending()
            session._turn.finish(RunStatus.CANCELLED)
            raise
        except LoopFailure as failure:
            session._close_pending()
            session._turn.finish(RunStatus.FAILED)
            result = RunResult(
                run_id, RunStatus.FAILED, "", progress.steps, tuple(progress.outcomes), failure.code
            )
        except Exception:
            session._close_pending()
            session._turn.finish(RunStatus.FAILED)
            result = RunResult(
                run_id,
                RunStatus.FAILED,
                "",
                progress.steps,
                tuple(progress.outcomes),
                "runtime_error",
            )
        else:
            session._turn.finish(status)
            result = RunResult(run_id, status, content, progress.steps, tuple(progress.outcomes))
        await events.emit(
            "run_finished",
            progress.steps,
            status=result.status.value,
            content=result.content,
            error=result.error,
        )
        return result
