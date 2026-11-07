"""Text/tool-calling adapter for an explicitly configured LangChain chat model."""

from typing import Any, cast

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage

from ..protocol import Json, Message, ModelResponse, ToolCall, ToolDefinition


class LangChainModel:
    """The caller owns credentials, provider parameters, retries and model lifecycle.

    This adapter supports ordinary text and tool calls, not multimodal content
    or provider-specific reasoning/signature state. Disable thinking modes that
    require replaying that state, or supply your own ChatModel adapter.
    """

    def __init__(self, model: BaseChatModel) -> None:
        self._model = model

    async def complete(
        self, messages: tuple[Message, ...], tools: tuple[ToolDefinition, ...]
    ) -> ModelResponse:
        history: list[BaseMessage] = []
        for message in messages:
            if message.role == "system":
                history.append(SystemMessage(content=message.content))
            elif message.role == "user":
                history.append(HumanMessage(content=message.content))
            elif message.role == "assistant":
                calls: list[Any] = [
                    {"id": call.id, "name": call.name, "args": call.arguments, "type": "tool_call"}
                    for call in message.tool_calls
                ]
                history.append(AIMessage(content=message.content, tool_calls=calls))
            else:
                history.append(
                    ToolMessage(
                        content=message.content,
                        tool_call_id=message.tool_call_id or "",
                    )
                )
        declarations: list[dict[str, Any]] = [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                },
            }
            for tool in tools
        ]
        model = self._model.bind_tools(declarations) if tools else self._model
        reply = await model.ainvoke(history)
        if not isinstance(reply, AIMessage) or reply.invalid_tool_calls:
            raise ValueError("The model did not return a valid assistant message")
        if not isinstance(reply.content, str):
            raise ValueError("This adapter supports text-only responses")
        if reply.additional_kwargs.get("reasoning_content"):
            raise ValueError("Thinking-state replay is outside this adapter's supported scope")
        return ModelResponse(
            reply.content,
            tuple(
                ToolCall(call.get("id") or "", call["name"], cast(dict[str, Json], call["args"]))
                for call in reply.tool_calls
            ),
        )
