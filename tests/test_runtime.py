import asyncio
import json
import unittest

from tracemind_runtime import (
    AgentRuntime,
    ModelResponse,
    RunOptions,
    RunStatus,
    Session,
    SessionBusyError,
    Tool,
    ToolCall,
    ToolDefinition,
)


class ScriptedModel:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.requests = []

    async def complete(self, messages, tools):
        self.requests.append((messages, tools))
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def call(identifier="call-1", value=7, name="record"):
    return ToolCall(identifier, name, {"value": value})


def record_tool(effects, *, approval=False, handler=None):
    async def record(arguments):
        effects.append(arguments["value"])
        return arguments["value"]

    return Tool(
        ToolDefinition(
            "record",
            "Record a number in memory.",
            {
                "type": "object",
                "properties": {"value": {"type": "number"}},
                "required": ["value"],
                "additionalProperties": False,
            },
        ),
        handler or record,
        requires_approval=approval,
    )


def assert_paired(test, messages):
    pending = set()
    for message in messages:
        if message.role == "tool":
            test.assertIn(message.tool_call_id, pending)
            pending.remove(message.tool_call_id)
        else:
            test.assertFalse(pending, "Unmatched calls before a new message")
            pending.update(call.id for call in message.tool_calls)
    test.assertFalse(pending)


class RuntimeTests(unittest.IsolatedAsyncioTestCase):
    async def test_text_turns_preserve_history_and_one_system_message(self):
        session = Session(system_prompt="Be concise.")
        model = ScriptedModel(ModelResponse("one"), ModelResponse("two"))
        runtime = AgentRuntime(model)
        self.assertEqual((await runtime.run(session, "first")).content, "one")
        self.assertEqual((await runtime.run(session, "second")).content, "two")
        self.assertEqual(
            [m.role for m in session.snapshot()],
            ["system", "user", "assistant", "user", "assistant"],
        )
        self.assertEqual(len(model.requests[1][0]), 4)
        self.assertEqual(session.status, RunStatus.COMPLETED)

    async def test_tool_batch_runs_sequentially_and_pairs_all_results(self):
        effects = []
        model = ScriptedModel(
            ModelResponse(tool_calls=(call("a", 1), call("b", 2))), ModelResponse("done")
        )
        session = Session()
        result = await AgentRuntime(model, tools=[record_tool(effects)]).run(session, "record")
        self.assertEqual(effects, [1, 2])
        self.assertEqual([o.status for o in result.operations], ["ok", "ok"])
        self.assertEqual(result.model_steps, 2)
        assert_paired(self, session.snapshot())

    async def test_tools_require_approval_by_default(self):
        effects = []
        base = record_tool(effects)
        tool = Tool(base.definition, base.handler)
        model = ScriptedModel(ModelResponse(tool_calls=(call(),)), ModelResponse("denied"))
        result = await AgentRuntime(model, tools=[tool]).run(Session(), "record")
        self.assertEqual(result.operations[0].status, "denied")
        self.assertEqual(effects, [])

    async def test_approval_cannot_rewrite_the_executed_arguments(self):
        effects = []

        async def approve(request):
            request.arguments["value"] = 999
            return True

        model = ScriptedModel(ModelResponse(tool_calls=(call(value=7),)), ModelResponse("done"))
        result = await AgentRuntime(model, tools=[record_tool(effects, approval=True)]).run(
            Session(),
            "record",
            approve=approve,
        )
        self.assertEqual(effects, [7])
        self.assertEqual(result.operations[0].status, "ok")

    async def test_truthy_nonboolean_approval_is_denied(self):
        effects = []

        async def approve(_):
            return 1

        model = ScriptedModel(ModelResponse(tool_calls=(call(),)), ModelResponse("done"))
        result = await AgentRuntime(model, tools=[record_tool(effects, approval=True)]).run(
            Session(),
            "record",
            approve=approve,
        )
        self.assertEqual(result.operations[0].status, "denied")
        self.assertFalse(effects)

    async def test_approval_error_does_not_execute_or_expose_exception(self):
        effects = []

        async def approve(_):
            raise RuntimeError("private approval detail")

        model = ScriptedModel(ModelResponse(tool_calls=(call(),)), ModelResponse("done"))
        result = await AgentRuntime(model, tools=[record_tool(effects, approval=True)]).run(
            Session(),
            "record",
            approve=approve,
        )
        self.assertEqual(result.operations[0].status, "approval_failed")
        self.assertNotIn("private approval", result.operations[0].content)
        self.assertFalse(effects)

    async def test_approval_timeout_does_not_execute(self):
        effects = []

        async def approve(_):
            await asyncio.Event().wait()

        model = ScriptedModel(ModelResponse(tool_calls=(call(),)), ModelResponse("done"))
        result = await AgentRuntime(
            model,
            tools=[record_tool(effects, approval=True)],
            options=RunOptions(approval_timeout=0.01),
        ).run(
            Session(),
            "record",
            approve=approve,
        )
        self.assertEqual(result.operations[0].status, "approval_timeout")
        self.assertFalse(effects)

    async def test_no_registered_tool_means_no_implicit_shell_access(self):
        model = ScriptedModel(
            ModelResponse(tool_calls=(call(name="exec_command"),)), ModelResponse("unavailable")
        )
        result = await AgentRuntime(model).run(Session(), "run something")
        self.assertEqual(result.operations[0].status, "unknown_tool")

    async def test_invalid_arguments_never_reach_the_handler(self):
        for value in (True, "7", None, {"nested": 7}):
            with self.subTest(value=value):
                effects = []
                model = ScriptedModel(
                    ModelResponse(tool_calls=(call(value=value),)), ModelResponse("invalid")
                )
                result = await AgentRuntime(model, tools=[record_tool(effects)]).run(
                    Session(),
                    "record",
                )
                self.assertEqual(result.operations[0].status, "invalid_arguments")
                self.assertFalse(effects)

    async def test_duplicate_call_ids_in_batch_fail_before_any_effect(self):
        effects = []
        model = ScriptedModel(ModelResponse(tool_calls=(call(), call())))
        result = await AgentRuntime(model, tools=[record_tool(effects)]).run(Session(), "record")
        self.assertEqual(result.error, "invalid_tool_batch")
        self.assertEqual(result.status, RunStatus.FAILED)
        self.assertFalse(effects)

    async def test_reused_call_id_is_not_reexecuted(self):
        effects = []
        model = ScriptedModel(
            ModelResponse(tool_calls=(call(),)), ModelResponse(tool_calls=(call(),))
        )
        session = Session()
        result = await AgentRuntime(model, tools=[record_tool(effects)]).run(session, "record")
        self.assertEqual(result.error, "invalid_tool_batch")
        self.assertEqual(effects, [7])
        assert_paired(self, session.snapshot())

    async def test_step_limit_is_distinct_from_completed(self):
        effects = []
        model = ScriptedModel(ModelResponse(tool_calls=(call(),)))
        session = Session()
        result = await AgentRuntime(
            model, tools=[record_tool(effects)], options=RunOptions(max_steps=1)
        ).run(session, "record")
        self.assertEqual(result.status, RunStatus.LIMIT_REACHED)
        self.assertEqual(effects, [7])
        self.assertEqual(len(model.requests), 1)
        assert_paired(self, session.snapshot())

    async def test_oversized_batch_fails_without_effects(self):
        effects = []
        model = ScriptedModel(ModelResponse(tool_calls=(call("a"), call("b"))))
        result = await AgentRuntime(
            model, tools=[record_tool(effects)], options=RunOptions(max_tools_per_step=1)
        ).run(Session(), "go")
        self.assertEqual(result.error, "invalid_tool_batch")
        self.assertFalse(effects)

    async def test_large_tool_output_is_bounded_and_is_not_reexecuted(self):
        effects = []

        async def large(arguments):
            effects.append(arguments["value"])
            return "large" * 100

        model = ScriptedModel(ModelResponse(tool_calls=(call(),)), ModelResponse("done"))
        result = await AgentRuntime(
            model, tools=[record_tool(effects, handler=large)], options=RunOptions(result_limit=100)
        ).run(Session(), "go")
        self.assertEqual(result.operations[0].status, "output_too_large")
        self.assertEqual(effects, [7])
        self.assertNotIn("large" * 50, model.requests[1][0][-1].content)

    async def test_provider_failure_is_not_retried_and_is_redacted(self):
        model = ScriptedModel(RuntimeError("private provider credential"))
        result = await AgentRuntime(model).run(Session(), "go")
        self.assertEqual(result.error, "model_error")
        self.assertNotIn("private provider", repr(result))
        self.assertEqual(len(model.requests), 1)

    async def test_model_timeout_is_a_failed_turn(self):
        class WaitingModel:
            async def complete(self, messages, tools):
                await asyncio.Event().wait()

        session = Session()
        result = await AgentRuntime(WaitingModel(), options=RunOptions(model_timeout=0.01)).run(
            session,
            "go",
        )
        self.assertEqual(result.error, "model_timeout")
        self.assertEqual(session.status, RunStatus.FAILED)

    async def test_tool_timeout_reports_uncertainty_without_retrying(self):
        effects = []

        async def slow(arguments):
            effects.append(arguments["value"])
            await asyncio.Event().wait()

        model = ScriptedModel(ModelResponse(tool_calls=(call(),)), ModelResponse("check state"))
        result = await AgentRuntime(
            model, tools=[record_tool(effects, handler=slow)], options=RunOptions(tool_timeout=0.01)
        ).run(Session(), "go")
        self.assertEqual(effects, [7])
        self.assertEqual(result.operations[0].status, "timeout")
        self.assertIn("External effects may have occurred", result.operations[0].content)

    async def test_handler_exception_is_redacted_without_retry(self):
        effects = []

        async def broken(arguments):
            effects.append(arguments["value"])
            raise RuntimeError("private handler detail")

        model = ScriptedModel(ModelResponse(tool_calls=(call(),)), ModelResponse("check state"))
        result = await AgentRuntime(model, tools=[record_tool(effects, handler=broken)]).run(
            Session(),
            "go",
        )
        self.assertEqual(effects, [7])
        self.assertEqual(result.operations[0].status, "failed")
        self.assertNotIn("private handler", result.operations[0].content)

    async def test_cancellation_pairs_pending_calls_and_preserves_completed_effects(self):
        effects = []
        entered = asyncio.Event()

        async def record(arguments):
            if arguments["value"] == 2:
                entered.set()
                await asyncio.Event().wait()
            effects.append(arguments["value"])
            return arguments["value"]

        model = ScriptedModel(ModelResponse(tool_calls=(call("a", 1), call("b", 2), call("c", 3))))
        session = Session()
        task = asyncio.create_task(
            AgentRuntime(
                model,
                tools=[
                    record_tool(
                        effects,
                        handler=record,
                    )
                ],
            ).run(session, "record")
        )
        await asyncio.wait_for(entered.wait(), 1)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertEqual(session.status, RunStatus.CANCELLED)
        self.assertEqual(effects, [1])
        outcomes = [json.loads(m.content)["status"] for m in session.snapshot() if m.role == "tool"]
        self.assertEqual(outcomes, ["ok", "interrupted", "interrupted"])
        assert_paired(self, session.snapshot())
        result = await AgentRuntime(ScriptedModel(ModelResponse("resumed chat"))).run(
            session,
            "Check the existing state; do not repeat anything.",
        )
        self.assertEqual(result.status, RunStatus.COMPLETED)
        self.assertEqual(effects, [1])

    async def test_cancellation_during_approval_does_not_execute(self):
        entered = asyncio.Event()
        effects = []

        async def approve(_):
            entered.set()
            await asyncio.Event().wait()

        session = Session()
        runtime = AgentRuntime(
            ScriptedModel(ModelResponse(tool_calls=(call(),))),
            tools=[record_tool(effects, approval=True)],
        )
        task = asyncio.create_task(runtime.run(session, "go", approve=approve))
        await asyncio.wait_for(entered.wait(), 1)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertFalse(effects)
        assert_paired(self, session.snapshot())

    async def test_session_rejects_concurrent_turns_from_another_runtime(self):
        entered, finish = asyncio.Event(), asyncio.Event()

        class WaitingModel:
            async def complete(self, messages, tools):
                entered.set()
                await finish.wait()
                return ModelResponse("finished")

        session = Session()
        task = asyncio.create_task(AgentRuntime(WaitingModel()).run(session, "first"))
        await asyncio.wait_for(entered.wait(), 1)
        try:
            with self.assertRaises(SessionBusyError):
                await AgentRuntime(ScriptedModel()).run(session, "second")
            self.assertEqual(len(session.snapshot()), 2)
        finally:
            finish.set()
            await task

    async def test_independent_sessions_do_not_share_history(self):
        class EchoModel:
            async def complete(self, messages, tools):
                await asyncio.sleep(0)
                return ModelResponse(messages[-1].content)

        runtime = AgentRuntime(EchoModel())
        results = await asyncio.gather(runtime.run(Session(), "a"), runtime.run(Session(), "b"))
        self.assertEqual([r.content for r in results], ["a", "b"])

    async def test_blank_input_leaves_session_untouched(self):
        session = Session()
        with self.assertRaises(ValueError):
            await AgentRuntime(ScriptedModel()).run(session, "   ")
        self.assertEqual(session.status, RunStatus.IDLE)
        self.assertEqual(len(session.snapshot()), 1)

    async def test_request_objects_cannot_mutate_session_history(self):
        effects = []

        class MutatingModel:
            def __init__(self):
                self.step = 0

            async def complete(self, messages, tools):
                self.step += 1
                tools[0].parameters["properties"] = {}
                if self.step == 1:
                    return ModelResponse(tool_calls=(call(),))
                messages[-2].tool_calls[0].arguments["value"] = 999
                return ModelResponse("done")

        session = Session()
        await AgentRuntime(MutatingModel(), tools=[record_tool(effects)]).run(session, "go")
        self.assertEqual(effects, [7])
        self.assertEqual(session.snapshot()[2].tool_calls[0].arguments["value"], 7)

    async def test_remote_schema_reference_is_rejected_without_fetching(self):
        effects = []
        tool = record_tool(effects)
        definition = ToolDefinition(
            "record",
            "Reference not available locally.",
            {
                "type": "object",
                "$ref": "https://invalid.example/private-schema.json",
            },
        )
        model = ScriptedModel(ModelResponse(tool_calls=(call(),)), ModelResponse("invalid"))
        result = await AgentRuntime(model, tools=[Tool(definition, tool.handler, False)]).run(
            Session(),
            "go",
        )
        self.assertEqual(result.operations[0].status, "invalid_arguments")
        self.assertFalse(effects)

    async def test_malformed_model_response_does_not_become_success(self):
        result = await AgentRuntime(ScriptedModel("not a ModelResponse")).run(Session(), "go")
        self.assertEqual(result.error, "invalid_response")


class ProtocolTests(unittest.TestCase):
    def test_options_reject_infinite_negative_and_boolean_values(self):
        for values in (
            {"max_steps": 0},
            {"max_steps": True},
            {"model_timeout": float("inf")},
            {"tool_timeout": -1},
            {"approval_timeout": float("nan")},
        ):
            with self.subTest(values=values), self.assertRaises(ValueError):
                RunOptions(**values)

    def test_json_arguments_are_finite_plain_data(self):
        for value in (float("nan"), float("inf"), object(), {1: "bad"}):
            with self.subTest(value=value), self.assertRaises(ValueError):
                call(value=value)

    def test_duplicate_tool_names_are_rejected(self):
        tool = record_tool([])
        with self.assertRaises(ValueError):
            AgentRuntime(ScriptedModel(), tools=[tool, tool])

    def test_empty_model_reply_is_rejected(self):
        with self.assertRaises(ValueError):
            ModelResponse()


if __name__ == "__main__":
    unittest.main()
