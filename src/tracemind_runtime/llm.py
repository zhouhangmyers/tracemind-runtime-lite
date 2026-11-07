"""The host supplies a model. This package does not manage provider accounts."""

from typing import Protocol

from .protocol import Message, ModelResponse, ToolDefinition


class ChatModel(Protocol):
    async def complete(
        self, messages: tuple[Message, ...], tools: tuple[ToolDefinition, ...]
    ) -> ModelResponse: ...
