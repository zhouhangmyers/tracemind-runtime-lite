# TraceMind Runtime Lite

一个精简、可独立运行的 Python 单智能体运行时。

模型决定下一步，Runtime 组织消息与工具调用，宿主决定哪些工具可以执行。
适合学习运行时边界，以及作为小型 Agent 应用的起点。

这是独立实现的社区精简版，不是 TraceMind 桌面产品的完整 Runtime，
也不是其内部接口的兼容发行包。TraceMind 产品继续独立维护。

## 已包含

- 单智能体多步执行循环与进程内连续对话。
- 独立模型接口，可选 LangChain 文本/工具调用适配器。
- 显式注册工具、JSON Schema 参数校验和默认逐次审批。
- 模型、工具和审批超时；原生 asyncio 取消。
- 调用 ID 去重、工具消息配对、完成/失败/步数耗尽状态。
- 有序执行事件流，包含审批、工具调用和终态，支持背压与关闭清理。
- 可选工作区只读工具：列举文件、按行读取、字面文本检索。
- 版本化 JSON 会话历史，保存后可加载并开始新一轮对话。
- 完全离线的示例和测试。

没有持久执行检查点、任务重启恢复、Code Mode、多智能体调度、自动摘要、
缓存优化策略、浏览器控制、Skills/MCP 管理、长期记忆或桌面服务。

## 安装与离线运行

Python 3.12 或更新版本。在仓库目录运行：

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell 使用 .\.venv\Scripts\Activate.ps1
python -m pip install -e .
python examples/offline.py
```

预期输出：

```text
6 × 7 = 42
status=completed, model_steps=2
```

这个示例不需要模型密钥，不联网，也不产生模型费用。

## 完整项目检索示例

```bash
python examples/project_review.py .
```

这会在明确指定的项目目录内列举文件、检索类声明并读取文件预览，同时打印执行事件。
默认使用离线规则模型，演示的文件读取是真实操作，结论只反映规则计数，不冒充 LLM 分析。
安装 DeepSeek 可选依赖、设置密钥与模型后，加 `--live` 才会调用自己的付费 API：

```bash
python examples/project_review.py ./your-project --live
```

## 执行事件流

```python
from contextlib import aclosing

# 在 async 函数内：
# async with aclosing(runtime.stream(session, "分析项目")) as events:
#     async for event in events:
#         print(event.sequence, event.kind, event.data)
```

每次运行有独立 ID 和连续事件序号。事件描述模型调用、操作请求、审批、工具开始、
操作结果和最终状态；不会默认携带工具参数或完整工具输出。
这是执行进度流，不是逐 Token 流。缓冲最多32条事件，消费变慢会阻止生产者无限堆积。
提前结束迭代时应使用 `aclosing`，以取消生产任务、等待清理并补齐未完成的工具消息。

## 工作区只读工具

```python
from tracemind_runtime import ReadOnlyWorkspace

workspace = ReadOnlyWorkspace("./your-project")
# runtime = AgentRuntime(model, tools=workspace.tools())
```

宿主明确选择根目录后，才会提供 `workspace_list`、`workspace_read` 和 `workspace_search`。
读取使用相对路径，阻止静态路径越界、符号链接和 Windows junction，跳过隐藏文件、
常见依赖目录及部分凭据/数据库后缀；文件、扫描条目和检索输出均有限额。
读取结果带行号与后续位置，扫描不完整时明确返回 `truncated`。

这些便利检查不构成 OS 沙箱，也不能识别所有敏感文件。只选择有权向模型提供的工作区，
不用于目录可能被其他进程恶意替换的场景。

## 会话历史保存

```python
from pathlib import Path
from tracemind_runtime import Session

# 路径、加密、权限和保存时机由宿主管理。
# Path("conversation.json").write_text(session.to_json(), encoding="utf-8")
# loaded = Session.from_json(Path("conversation.json").read_text(encoding="utf-8"))
```

JSON 只保存会话消息和末次状态，不包含活跃任务、审批许可、供应商凭据或执行检查点。
加载会校验格式版本、重复键、工具调用 ID 和消息配对，拒绝恢复 `running` 状态。
历史可能包含用户内容和工具结果，应由宿主保护，且只从可信来源加载。

```bash
python examples/history.py
```

这个离线示例在临时目录保存历史，加载后继续第二轮对话；它不会重放任何旧工具调用。

## 接入自己的模型

实现以下接口即可，核心不要求使用某个框架或服务商：

```python
from tracemind_runtime import AgentRuntime, Message, ModelResponse, Session, ToolDefinition

class MyModel:
    async def complete(
        self,
        messages: tuple[Message, ...],
        tools: tuple[ToolDefinition, ...],
    ) -> ModelResponse:
        # 在这里调用自己的模型，并转换文本和工具请求。
        return ModelResponse("Hello")

runtime = AgentRuntime(MyModel())
# 在 async 函数内：
# result = await runtime.run(Session(), "Hello")
```

可选 LangChain 适配器：

```bash
python -m pip install -e ".[langchain]"
```

```python
from tracemind_runtime.integrations.langchain import LangChainModel

# configured_model 是调用方已配置好的 LangChain BaseChatModel。
# runtime = AgentRuntime(LangChainModel(configured_model))
```

适配器只处理普通文本和工具调用，不处理多模态或供应商专用的推理状态回传。
需要这些能力时，实现自己的 `ChatModel`。密钥、模型名称、SDK 重试和客户端关闭由宿主管理。

DeepSeek 可选示例：

```bash
python -m pip install -e ".[deepseek]"
# 在当前终端设置 DEEPSEEK_API_KEY 和 DEEPSEEK_MODEL；不要写入仓库。
python examples/deepseek_chat.py
```

运行该示例会调用你自己的付费模型服务。模型必须支持普通工具调用；
示例显式关闭 thinking，避免丢失需要回传的推理状态。模型 ID 由调用方明确提供，
不内置会随时间失效的默认型号。

## 工具与审批

`Tool` 由工具声明、异步执行函数和 `requires_approval` 组成，默认需要审批。
未提供审批函数、审批失败或未明确返回 `True` 时，都不会调用工具。
审批回调会收到独立的参数副本，不能修改随后实际执行的参数。

只有宿主明确声明为无需审批的工具，才可直接执行；这种声明应仅用于可信、无副作用的操作。
本项目不附带 shell、文件写入或网络执行工具。

```bash
python examples/approval.py
```

审批示例只修改进程内的草稿列表，不写入文件。

## 会话、限制与取消

复用同一个 `Session` 可保留进程内历史。同一会话一次只允许一轮执行，
不同会话可并发；模型适配器本身需要支持宿主采用的并发方式。
会话仅供单个事件循环使用，不支持跨线程或跨进程共享。

`RunOptions` 可设置步数、每步工具数、超时和单条工具结果长度。
达到步数限制会返回 `limit_reached`，不会伪装成已完成。

使用 asyncio 的原生取消：

```python
import asyncio

# task = asyncio.create_task(runtime.run(session, "执行任务"))
# task.cancel()
# await task  # 抛出 asyncio.CancelledError；session.status 为 cancelled。
```

取消或超时无法保证外部副作用停止或回滚。Runtime 不会自动重试模型请求或工具执行，
也不会把未记录的工具结果当作成功。中断后会补齐缺少的工具结果消息，
提醒后续调用先核对外部状态；这不是持久恢复或幂等执行保证。

详细边界见 [架构说明](docs/architecture.md) 和 [执行边界](docs/execution.md)。

## 开发检查

```bash
python -m pip install -e ".[dev,langchain]"
python -m unittest discover -s tests -v
ruff check src tests examples
mypy src
```

不安装 LangChain 时，适配器测试会明确跳过；核心测试仍可运行。
所有测试使用离线模型或 SDK 消息对象，不请求真实 API，也不读取用户数据。

## 许可证

MIT，见 [LICENSE](LICENSE)。第三方依赖仍遵循各自许可证。
