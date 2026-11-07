"""A complete two-step tool loop without credentials or network access."""

import asyncio
import json

from tracemind_runtime import (
    AgentRuntime,
    Json,
    Message,
    ModelResponse,
    Session,
    Tool,
    ToolCall,
    ToolDefinition,
)


async def multiply(arguments: dict[str, Json]) -> Json:
    return float(str(arguments["a"])) * float(str(arguments["b"]))


def multiplication_tool() -> Tool:
    return Tool(
        ToolDefinition(
            "multiply",
            "Multiply two finite numbers.",
            {
                "type": "object",
                "properties": {
                    "a": {"type": "number"},
                    "b": {"type": "number"},
                },
                "required": ["a", "b"],
                "additionalProperties": False,
            },
        ),
        multiply,
        requires_approval=False,
    )


class DemoModel:
    async def complete(
        self, messages: tuple[Message, ...], tools: tuple[ToolDefinition, ...]
    ) -> ModelResponse:
        if messages[-1].role == "tool":
            outcome = json.loads(messages[-1].content)
            return ModelResponse(f"6 × 7 = {outcome['result']:g}")
        return ModelResponse(tool_calls=(ToolCall("multiply-once", "multiply", {"a": 6, "b": 7}),))


async def main() -> None:
    runtime = AgentRuntime(DemoModel(), tools=[multiplication_tool()])
    result = await runtime.run(Session(), "What is 6 times 7?")
    print(result.content)
    print(f"status={result.status}, model_steps={result.model_steps}")


if __name__ == "__main__":
    asyncio.run(main())
