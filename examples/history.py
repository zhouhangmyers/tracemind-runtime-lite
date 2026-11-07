"""The host saves conversation JSON; no execution or approvals are resumed."""

import asyncio
import tempfile
from pathlib import Path

from tracemind_runtime import AgentRuntime, Message, ModelResponse, Session, ToolDefinition


class CountModel:
    async def complete(
        self,
        messages: tuple[Message, ...],
        tools: tuple[ToolDefinition, ...],
    ) -> ModelResponse:
        return ModelResponse(f"User messages seen: {sum(m.role == 'user' for m in messages)}")


async def main() -> None:
    runtime = AgentRuntime(CountModel())
    original = Session()
    print((await runtime.run(original, "First message")).content)
    with tempfile.TemporaryDirectory(prefix="runtime-history-") as directory:
        file = Path(directory) / "conversation.json"
        file.write_text(original.to_json(), encoding="utf-8")
        loaded = Session.from_json(file.read_text(encoding="utf-8"))
    print((await runtime.run(loaded, "Second message")).content)


if __name__ == "__main__":
    asyncio.run(main())
