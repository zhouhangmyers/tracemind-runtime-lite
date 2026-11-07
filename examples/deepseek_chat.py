"""Opt-in live example. Importing this file never sends a model request."""

import asyncio
import os

from offline import multiplication_tool

from tracemind_runtime import AgentRuntime, Session


async def main() -> None:
    from langchain_deepseek import ChatDeepSeek

    from tracemind_runtime.integrations.langchain import LangChainModel

    if not os.environ.get("DEEPSEEK_API_KEY") or not os.environ.get("DEEPSEEK_MODEL"):
        raise SystemExit(
            "Set DEEPSEEK_API_KEY and DEEPSEEK_MODEL before running this live example."
        )
    model = ChatDeepSeek(
        model=os.environ["DEEPSEEK_MODEL"],
        max_retries=0,
        timeout=50,
        extra_body={"thinking": {"type": "disabled"}},
    )
    runtime = AgentRuntime(LangChainModel(model), tools=[multiplication_tool()])
    result = await runtime.run(Session(), "请调用 multiply 计算 6 乘以 7，然后回答。")
    print(result.content if result.content else f"status={result.status}, error={result.error}")


if __name__ == "__main__":
    asyncio.run(main())
