"""Read an explicitly selected project, showing execution events. Offline by default."""

import argparse
import asyncio
import json
import os
from contextlib import aclosing
from pathlib import Path

from tracemind_runtime import (
    AgentRuntime,
    Message,
    ModelResponse,
    ReadOnlyWorkspace,
    Session,
    ToolCall,
    ToolDefinition,
)


class DemoProjectModel:
    """A deterministic workflow for demonstrating real file I/O, not an LLM."""

    def __init__(self) -> None:
        self.step = 0
        self.files: list[str] = []
        self.matches = 0

    async def complete(
        self,
        messages: tuple[Message, ...],
        tools: tuple[ToolDefinition, ...],
    ) -> ModelResponse:
        self.step += 1
        if self.step == 1:
            return ModelResponse(tool_calls=(ToolCall("inspect-list", "workspace_list", {}),))
        last = json.loads(messages[-1].content)
        if last["status"] != "ok":
            return ModelResponse("Inspection stopped: a read-only operation was rejected.")
        if self.step == 2:
            self.files = last["result"]["files"]
            if not self.files:
                return ModelResponse("No readable files were found in this workspace.")
            return ModelResponse(
                tool_calls=(
                    ToolCall(
                        "inspect-search",
                        "workspace_search",
                        {"query": "class ", "limit": 5},
                    ),
                )
            )
        if self.step == 3:
            self.matches = len(last["result"]["matches"])
            selected = "README.md" if "README.md" in self.files else self.files[0]
            return ModelResponse(
                tool_calls=(
                    ToolCall(
                        "inspect-read",
                        "workspace_read",
                        {"path": selected, "line_count": 12},
                    ),
                )
            )
        return ModelResponse(
            f"Offline inspection: {len(self.files)} readable files; "
            f"{self.matches} sampled class declarations.\n"
            f"Preview of {last['result']['path']}:\n{last['result']['content']}"
        )


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="Authorized project directory to inspect")
    parser.add_argument("--live", action="store_true", help="Use your paid DeepSeek API")
    args = parser.parse_args()
    workspace = ReadOnlyWorkspace(args.root)
    if args.live:
        from langchain_deepseek import ChatDeepSeek

        from tracemind_runtime.integrations.langchain import LangChainModel

        if not os.environ.get("DEEPSEEK_API_KEY") or not os.environ.get("DEEPSEEK_MODEL"):
            raise SystemExit("Set DEEPSEEK_API_KEY and DEEPSEEK_MODEL first.")
        model = LangChainModel(
            ChatDeepSeek(
                model=os.environ["DEEPSEEK_MODEL"],
                timeout=50,
                max_retries=0,
                extra_body={"thinking": {"type": "disabled"}},
            )
        )
        runtime = AgentRuntime(model, tools=workspace.tools())
    else:
        runtime = AgentRuntime(DemoProjectModel(), tools=workspace.tools())
    session = Session(system_prompt="Inspect the project using only the supplied read-only tools.")
    async with aclosing(
        runtime.stream(session, "List files, inspect code and summarize the project.")
    ) as events:
        async for event in events:
            print(f"[{event.sequence:02}] {event.kind}", end="")
            if event.kind == "run_finished":
                print(f" ({event.data['status']})\n{event.data['content']}")
            else:
                print(f" {event.data.get('name', '')}")


if __name__ == "__main__":
    asyncio.run(main())
