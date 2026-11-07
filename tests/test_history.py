import asyncio
import json
import unittest

from test_runtime import ScriptedModel, assert_paired, call, record_tool

from tracemind_runtime import AgentRuntime, ModelResponse, RunStatus, Session


class HistoryTests(unittest.IsolatedAsyncioTestCase):
    async def test_round_trip_preserves_tool_history_and_allows_a_new_turn(self):
        session = Session(system_prompt="Be concise.")
        await AgentRuntime(
            ScriptedModel(ModelResponse(tool_calls=(call(),)), ModelResponse("one")),
            tools=[record_tool([])],
        ).run(session, "first")
        loaded = Session.from_json(session.to_json())
        self.assertEqual(loaded.snapshot(), session.snapshot())
        self.assertEqual(loaded.status, RunStatus.COMPLETED)
        model = ScriptedModel(ModelResponse("two"))
        self.assertEqual((await AgentRuntime(model).run(loaded, "second")).content, "two")
        self.assertEqual(len(model.requests[0][0]), len(session.snapshot()) + 1)
        assert_paired(self, loaded.snapshot())

    async def test_running_session_cannot_be_saved(self):
        entered = asyncio.Event()

        class WaitingModel:
            async def complete(self, messages, tools):
                entered.set()
                await asyncio.Event().wait()

        session = Session()
        task = asyncio.create_task(AgentRuntime(WaitingModel()).run(session, "go"))
        await asyncio.wait_for(entered.wait(), 1)
        try:
            with self.assertRaises(ValueError):
                session.to_json()
        finally:
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
        self.assertEqual(Session.from_json(session.to_json()).status, RunStatus.CANCELLED)

    async def test_loaded_data_does_not_restore_approvals_or_execute_tools(self):
        effects = []
        session = Session()
        model = ScriptedModel(ModelResponse(tool_calls=(call(),)), ModelResponse("done"))

        async def approve(_):
            return True

        await AgentRuntime(model, tools=[record_tool(effects, approval=True)]).run(
            session,
            "go",
            approve=approve,
        )
        loaded = Session.from_json(session.to_json())
        self.assertEqual(effects, [7])
        second = ScriptedModel(ModelResponse(tool_calls=(call("new"),)), ModelResponse("denied"))
        result = await AgentRuntime(second, tools=[record_tool(effects, approval=True)]).run(
            loaded,
            "new request",
        )
        self.assertEqual(result.operations[0].status, "denied")
        self.assertEqual(effects, [7])


class HistoryValidationTests(unittest.TestCase):
    def data(self):
        return json.loads(Session().to_json())

    def test_unknown_versions_fields_and_running_status_are_rejected(self):
        for change in ({"version": 2}, {"version": True}, {"status": "running"}, {"extra": 1}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                Session.from_json(json.dumps(self.data() | change))

    def test_duplicate_keys_and_nonfinite_values_are_rejected(self):
        for text in ('{"version":1,"version":1}', '{"version":NaN}', '{"version":Infinity}'):
            with self.subTest(text=text), self.assertRaises(ValueError):
                Session.from_json(text)

    def test_history_size_is_bounded(self):
        with self.assertRaises(ValueError):
            Session.from_json(" " * (4 * 1024 * 1024 + 1))

    def test_extra_system_messages_are_not_accepted(self):
        data = self.data()
        data["messages"].append(data["messages"][0].copy())
        with self.assertRaises(ValueError):
            Session.from_json(json.dumps(data))

    def test_unpaired_or_duplicate_tool_messages_are_rejected(self):
        base = self.data()
        assistant = {
            "role": "assistant",
            "content": "",
            "tool_call_id": None,
            "tool_calls": [{"id": "one", "name": "read", "arguments": {}}],
        }
        tool = {"role": "tool", "content": "{}", "tool_calls": [], "tool_call_id": "one"}
        for suffix in (
            [assistant],
            [tool],
            [assistant, tool, tool],
            [assistant, tool, assistant, tool],
        ):
            with self.subTest(suffix=suffix), self.assertRaises(ValueError):
                Session.from_json(json.dumps(base | {"messages": base["messages"] + suffix}))
