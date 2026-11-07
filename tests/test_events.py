import asyncio
import unittest
from contextlib import aclosing

from test_runtime import ScriptedModel, assert_paired, call, record_tool

from tracemind_runtime import AgentRuntime, ModelResponse, RunOptions, RunStatus, Session


class EventTests(unittest.IsolatedAsyncioTestCase):
    async def test_events_are_ordered_and_do_not_include_tool_payloads(self):
        effects = []

        async def approve(_):
            return True

        runtime = AgentRuntime(
            ScriptedModel(ModelResponse(tool_calls=(call(),)), ModelResponse("done")),
            tools=[record_tool(effects, approval=True)],
        )
        events = [event async for event in runtime.stream(Session(), "go", approve=approve)]
        self.assertEqual(
            [e.kind for e in events],
            [
                "run_started",
                "model_started",
                "model_completed",
                "operation_requested",
                "approval_requested",
                "approval_resolved",
                "tool_started",
                "operation_completed",
                "model_started",
                "model_completed",
                "run_finished",
            ],
        )
        self.assertEqual([e.sequence for e in events], list(range(1, len(events) + 1)))
        self.assertEqual(len({e.run_id for e in events}), 1)
        self.assertEqual(events[-1].data["status"], "completed")
        self.assertEqual(events[-1].data["content"], "done")
        self.assertFalse(any("arguments" in e.data for e in events))
        self.assertEqual(effects, [7])

    async def test_missing_approval_emits_denial_without_tool_started(self):
        effects = []
        runtime = AgentRuntime(
            ScriptedModel(ModelResponse(tool_calls=(call(),)), ModelResponse("denied")),
            tools=[record_tool(effects, approval=True)],
        )
        events = [event async for event in runtime.stream(Session(), "go")]
        self.assertNotIn("tool_started", [e.kind for e in events])
        resolved = next(e for e in events if e.kind == "approval_resolved")
        self.assertIs(resolved.data["approved"], False)
        self.assertEqual(effects, [])

    async def test_model_error_is_visible_as_a_terminal_event(self):
        runtime = AgentRuntime(ScriptedModel(RuntimeError("private provider detail")))
        events = [event async for event in runtime.stream(Session(), "go")]
        self.assertEqual(events[-1].kind, "run_finished")
        self.assertEqual(events[-1].data["status"], "failed")
        self.assertEqual(events[-1].data["error"], "model_error")
        self.assertNotIn("private provider", repr(events))

    async def test_early_close_cancels_and_joins_the_tool_task(self):
        entered, cancelled = asyncio.Event(), asyncio.Event()

        async def slow(arguments):
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()

        session = Session()
        runtime = AgentRuntime(
            ScriptedModel(ModelResponse(tool_calls=(call(),))),
            tools=[record_tool([], handler=slow)],
        )
        async with aclosing(runtime.stream(session, "go")) as stream:
            async for event in stream:
                if event.kind == "tool_started":
                    await asyncio.wait_for(entered.wait(), 1)
                    break
        self.assertTrue(cancelled.is_set())
        self.assertEqual(session.status, RunStatus.CANCELLED)
        assert_paired(self, session.snapshot())

    async def test_slow_consumer_applies_bounded_backpressure(self):
        class RepeatingModel:
            def __init__(self):
                self.calls = 0

            async def complete(self, messages, tools):
                self.calls += 1
                return ModelResponse(tool_calls=(call(str(self.calls)),))

        model = RepeatingModel()
        session = Session()
        runtime = AgentRuntime(model, tools=[record_tool([])], options=RunOptions(max_steps=100))
        async with aclosing(runtime.stream(session, "go")) as stream:
            self.assertEqual((await anext(stream)).kind, "run_started")
            await asyncio.sleep(0.02)
            self.assertLess(model.calls, 20)
        self.assertEqual(session.status, RunStatus.CANCELLED)
        assert_paired(self, session.snapshot())

    async def test_cancelling_consumer_does_not_leave_running_producer(self):
        entered, cancelled = asyncio.Event(), asyncio.Event()

        class WaitingModel:
            async def complete(self, messages, tools):
                entered.set()
                try:
                    await asyncio.Event().wait()
                finally:
                    cancelled.set()

        session = Session()
        runtime = AgentRuntime(WaitingModel())

        async def consume():
            async with aclosing(runtime.stream(session, "go")) as stream:
                async for _ in stream:
                    pass

        task = asyncio.create_task(consume())
        await asyncio.wait_for(entered.wait(), 1)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertTrue(cancelled.is_set())
        self.assertEqual(session.status, RunStatus.CANCELLED)

    async def test_step_limit_has_its_own_terminal_status(self):
        runtime = AgentRuntime(
            ScriptedModel(ModelResponse(tool_calls=(call(),))),
            tools=[record_tool([])],
            options=RunOptions(max_steps=1),
        )
        events = [event async for event in runtime.stream(Session(), "go")]
        self.assertEqual(events[-1].data["status"], "limit_reached")
