import unittest
from typing import cast

try:
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.messages import AIMessage, ToolMessage

    from tracemind_runtime.integrations.langchain import LangChainModel
except ImportError:
    LangChainModel = None

from tracemind_runtime import AgentRuntime, Message, Session, ToolCall, ToolDefinition


class FakeModel:
    def __init__(self, response):
        self.response = response
        self.messages = None
        self.tools = None

    def bind_tools(self, tools):
        self.tools = tools
        return self

    async def ainvoke(self, messages):
        self.messages = messages
        return self.response


@unittest.skipIf(LangChainModel is None, "Optional langchain-core dependency not installed")
class LangChainTests(unittest.IsolatedAsyncioTestCase):
    async def test_native_calls_and_tool_results_are_translated_without_network(self):
        native = FakeModel(
            AIMessage(
                content="",
                tool_calls=[
                    {"id": "next", "name": "lookup", "args": {"key": "a"}},
                ],
            )
        )
        adapter = LangChainModel(cast("BaseChatModel", native))
        messages = (
            Message("system", "Be concise"),
            Message("user", "lookup"),
            Message("assistant", "", (ToolCall("prior", "lookup", {"key": "b"}),)),
            Message("tool", '{"result":"b"}', tool_call_id="prior"),
        )
        definition = ToolDefinition("lookup", "Lookup a key.", {"type": "object"})
        reply = await adapter.complete(messages, (definition,))
        self.assertEqual(reply.tool_calls[0].id, "next")
        self.assertEqual(native.tools[0]["function"]["name"], "lookup")
        self.assertIsInstance(native.messages[-1], ToolMessage)
        self.assertEqual(native.messages[-1].tool_call_id, "prior")

    async def test_no_tools_does_not_bind_a_tool_schema(self):
        native = FakeModel(AIMessage(content="hello"))
        adapter = LangChainModel(cast("BaseChatModel", native))
        result = await AgentRuntime(adapter).run(Session(), "hello")
        self.assertEqual(result.content, "hello")
        self.assertIsNone(native.tools)

    async def test_invalid_tool_arguments_fail_closed(self):
        native = FakeModel(
            AIMessage(
                content="",
                invalid_tool_calls=[
                    {"id": "bad", "name": "lookup", "args": "{", "error": "Invalid JSON"},
                ],
            )
        )
        adapter = LangChainModel(cast("BaseChatModel", native))
        with self.assertRaises(ValueError):
            await adapter.complete((Message("user", "hello"),), ())

    async def test_unsupported_reasoning_state_is_not_silently_dropped(self):
        native = FakeModel(
            AIMessage(
                content="answer",
                additional_kwargs={
                    "reasoning_content": "provider state requiring replay",
                },
            )
        )
        adapter = LangChainModel(cast("BaseChatModel", native))
        with self.assertRaises(ValueError):
            await adapter.complete((Message("user", "hello"),), ())


if __name__ == "__main__":
    unittest.main()
