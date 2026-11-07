"""The host approves a pure in-memory operation; no files are written."""

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


class DraftModel:
    async def complete(
        self, messages: tuple[Message, ...], tools: tuple[ToolDefinition, ...]
    ) -> ModelResponse:
        if messages[-1].role == "tool":
            return ModelResponse(f"Tool result: {messages[-1].content}")
        return ModelResponse(tool_calls=(ToolCall("draft-once", "save_draft", {"text": "Hello"}),))


async def main() -> None:
    drafts: list[Json] = []

    async def save(arguments: dict[str, Json]) -> Json:
        drafts.append(arguments["text"])
        return {"saved_in_memory": True}

    async def approve(call: ToolCall) -> bool:
        print(f"Requested tool: {call.name}; arguments: {json.dumps(call.arguments)}")
        answer = await asyncio.to_thread(input, "Approve this in-memory draft? [y/N] ")
        return answer.strip().lower() == "y"

    tool = Tool(
        ToolDefinition(
            "save_draft",
            "Add text to the example's in-memory draft list.",
            {
                "type": "object",
                "properties": {"text": {"type": "string", "maxLength": 200}},
                "required": ["text"],
                "additionalProperties": False,
            },
        ),
        save,
    )
    result = await AgentRuntime(DraftModel(), tools=[tool]).run(
        Session(),
        "Create a draft.",
        approve=approve,
    )
    print(result.content)


if __name__ == "__main__":
    asyncio.run(main())
