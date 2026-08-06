# Uni2-Agent：整体架构、Planner Tool 调用与 LIBERO 闭环

> 对应项目：`/home/eren/codes/uni2-agent`。本文中的 “liebro” 指 `LIBERO`。

## 1. 一句话总结

`ReActAgent` 负责决策，`Toolbox` 负责 dispatch，`libero.py` 是 planner-facing adapter，`LiberoRuntime` 负责 episode 状态和统一协议，`PrimitiveBackend` 负责真正执行动作，`environment.py` 连接 LIBERO/MuJoCo，`services.py` 通过 RPC 调 SAM3/VLA；训练时 `Gateway` 负责模型生成、tool-call 解析和 trajectory 记录，但不直接执行 LIBERO tool。

```text
模型生成 tool call
  -> ReAct 写入 assistant message
  -> Toolbox 执行 LIBERO runtime
  -> runtime/backend 操作环境或调用 VLA/SAM3
  -> 返回动作后的 fresh observation
  -> ReAct 写入 role=tool
  -> 下一轮把完整 transcript 发回模型
  -> planner 调 finish
  -> runtime official success
  -> reward + trajectory
```

## 2. 项目整体分层

```text
训练器 / 推理入口
        |
        v
Agent Framework -> Gateway Session -> Agent / Planner
                                      |
                                      v
                         Toolbox / ToolContext
                                      |
                                      v
                               LIBERO Runtime
                                      |
                                      v
                              Primitive Backend
                                |     |      |
                              env   SAM3   VLA
                                      |
                                      v
                              Reward / Trajectory / TQ
```

主要目录：

```text
uni_agent/agents/react/       # ReAct planner loop + chat client
uni_agent/tools/              # Tool / Toolbox / LIBERO wrappers
uni_agent/tasks/libero/       # task / runtime / primitives / services / protocol
uni_agent/gateway/            # HTTP API、Gateway actor、session、codec
uni_agent/framework/          # RL rollout、session 生命周期、TQ 写入
```

LIBERO 子包的依赖方向：

```text
protocol <- environment <- services <- primitives <- runtime <- task
```

## 3. Planner 在哪里

项目没有单独的 `Planner` 类，LIBERO 默认由 `ReActAgent` 充当白盒 planner：

```text
ReActAgent.run()
  -> ReActAgent.step()
     1. 请求模型生成
     2. 记录 assistant message
     3. 分发 tool_calls
     4. 记录 tool observation
     5. 下一轮继续
```

关键文件：

- `uni_agent/agents/react/agent.py`
- `uni_agent/agents/react/model.py`

`ReActAgent` 的终止工具集合是：

```python
_FINISH_TOOLS = {"submit", "finish"}
```

## 4. Planner 看到的 LIBERO Tool

LIBERO 对 planner 暴露三个 Uni-Agent tool：

```text
libero_observe
libero_run_tool
finish
```

### `libero_observe`

读取 task language、当前 step、success/terminated、末端状态摘要、相机和最近 command log。

### `libero_run_tool`

它是 dispatcher，不是每个 primitive 一个 function。调用形状：

```json
{
  "name": "move_to",
  "args": {"xyz": [0.1, 0.2, 0.9], "gripper": "open"}
}
```

`name` 的 enum 和描述来自 `uni_agent/tasks/libero/protocol.py` 的 `LIBERO_TOOLS`，当前 ABI：

| 类别 | Tool |
| --- | --- |
| inspection | `view_driver_state`, `view_camera_meta` |
| perception | `segment`, `back_project` |
| analytic | `move_to`, `move_pose`, `rotate_wrist`, `rotate_pitch`, `set_gripper`, `release` |
| vla | `pi0_pick`, `pi0_doubled` |

设计不变量：planner 不看到 raw 7D action、`env.step`、IK、OSC 或关节目标；mutating tool 必须返回动作后的 fresh observation。

### `finish`

```json
{"status": "success", "summary": "placed the red mug on the plate"}
```

它记录 planner 自己的判断，同时读取 runtime 的官方 success 状态；不直接决定 reward。

## 5. 一次 tool call 如何回到对话框

假设模型返回：

```json
{
  "id": "call_123",
  "type": "function",
  "function": {
    "name": "libero_run_tool",
    "arguments": "{\"name\":\"segment\",\"args\":{\"prompt\":\"red mug\",\"camera\":\"agentview\"}}"
  }
}
```

### 5.1 ReAct 请求模型

`ReActAgent.step()` 调用：

```python
content, tool_calls, gen_info = await model.query(
    transcript,
    sampling_params=sampling_params,
)
```

`OpenAICompatibleChatModel` 发送：

```text
POST {base_url}/chat/completions
```

请求包括当前 `messages`、`Toolbox.schemas()` 生成的 OpenAI tools schema，以及采样参数。

模型返回原生 OpenAI `tool_calls` 时直接使用；若返回 Qwen/OpenRouter 风格 XML，`model.py` 会做 fallback 解析。

### 5.2 写入 assistant transcript

```python
assistant_msg = {"role": "assistant", "content": content}
if tool_calls:
    assistant_msg["tool_calls"] = tool_calls
transcript.append(assistant_msg)
```

### 5.3 Toolbox 分发

```python
tool_result = await toolbox.call(
    name,
    fn.get("arguments"),
    timeout=cfg.action_timeout,
)
```

`Toolbox.call()` 顺序：

```text
查找工具 -> 解析 JSON arguments -> Pydantic 校验 -> tool.run() -> ToolResult
```

### 5.4 进入 LIBERO runtime

`LiberoRunTool.run()` 做：

```python
tool_args = args.get("args") if isinstance(args.get("args"), dict) else {}
payload = await runtime.run_tool(
    str(args.get("name") or ""),
    tool_args,
)
```

例如：

```python
runtime.run_tool("segment", {"prompt": "red mug", "camera": "agentview"})
```

### 5.5 Tool result 回到 transcript

runtime payload 被 `_render()` 转成模型可读文本，例如：

```text
Task: put the mug on the plate
Step: 0/600
Success: false
Terminated: false
Message: segmented 'red mug' ... world_xyz=[...]
```

ReAct 追加：

```python
transcript.append({
    "role": "tool",
    "tool_call_id": tool_call.get("id"),
    "name": name,
    "content": observation,
})
```

下一轮 `model.query(transcript)` 会把 assistant/tool 历史一起发回模型。因此核心链路是：

```text
assistant.tool_calls
  -> Toolbox.call
  -> LiberoRunTool.run
  -> runtime.run_tool
  -> ToolResult / _render
  -> transcript.append(role="tool")
  -> 下一轮 chat/completions
```

## 6. LIBERO Runtime 闭环

`LiberoRuntime` 的统一协议：

```python
health()
reset()
observe(mode="text")
run_tool(name, args)
success()
close()
```

实现：

```text
LocalRuntime: runtime 和 LIBERO 环境同进程
HttpRuntime: 通过 HTTP 访问另一个 simulator 进程/主机
```

HTTP server endpoints：

```text
GET  /health
POST /reset
GET  /observe
POST /run_tool
GET  /success
POST /close
```

统一 payload：

```json
{
  "ok": true,
  "message": "move_to: within 0.0112 m of target",
  "observation": "Task: ...\nStep: 42/600\nSuccess: false...",
  "success": false,
  "terminated": false,
  "step_count": 42,
  "task_language": "put the mug on the plate",
  "info": {
    "runtime": "local_libero",
    "primitive_backend": "scripted_v0",
    "tool_result": {
      "schema_version": "tool_result_v0",
      "name": "move_to",
      "ok": true,
      "backend": "scripted_v0",
      "num_env_steps": 35,
      "failure_reason": "none",
      "diagnostics": {"final_dist_m": 0.0112}
    }
  },
  "protocol_version": "libero-runtime-v0"
}
```

`EpisodeRuntime._tool_payload()` 会记录 `last_tool`、command log，并重新生成 `observation_text()`，保证动作后状态可见。

## 7. Primitive Backend

`PrimitiveBackend` 用 `@primitive("tool_name")` 收集实现：

```python
@primitive("move_to")
def move_to(self, env, args):
    ...
```

执行：

```python
backend.execute(env, tool_name, tool_args)
```

未实现的工具返回结构化 `unsupported_tool`，并列出 backend 支持的 tool，方便 planner 改道。

当前 backend：

```text
scripted_v0
  纯脚本闭环伺服，无模型服务

privileged_v0
  scripted_v0 + oracle object localization，仅调试使用

service_v0
  scripted_v0 + SAM3 segmentation + VLA policy service
```

### `scripted_v0.move_to`

planner 只传：

```json
{"xyz": [0.1, 0.2, 0.9], "gripper": "open", "max_steps": 80}
```

backend 内部执行：

```text
读取末端位置
  -> 计算 xyz delta
  -> 转成 OSC action
  -> 构造内部 7D action
  -> env.step(action)
  -> 重复直到收敛/超步数/terminated
  -> 返回距离、步数、最终状态
```

内部 action 约定为 `action[:3]` 位置、`action[3]` pitch、`action[5]` yaw、`action[6]` gripper；7D action 不暴露给 planner。

## 8. service_v0：SAM3 和 VLA

### 8.1 `segment` 调 SAM3

```text
planner
  -> libero_run_tool(name="segment")
  -> EpisodeRuntime.run_tool()
  -> ServiceBackend.segment()
  -> Sam3Service.segment()
  -> POST {sam3_url}/call
  -> SAM3 返回 mask / score / box
  -> mask + depth + camera calibration
  -> median valid 3D points -> world_xyz
  -> ToolResult + fresh observation
  -> planner 下一轮
```

SAM3 只做视觉分割；world coordinate 由 runtime 本地的 depth/camera geometry 计算。服务不可用时返回 `service_unavailable`，planner 可切换 `back_project`。

### 8.2 `pi0_pick` 调 VLA

```text
planner
  -> libero_run_tool(name="pi0_pick")
  -> ServiceBackend.pi0_pick()
  -> agentview / wrist image + proprio + instruction
  -> VlaService.predict()
  -> POST {vla_url}/call
  -> 返回 action chunk
  -> runtime 内循环 env.step(action)
  -> 检测下降、上升、夹爪闭合
  -> 判断 lift/grasp 成功
  -> fresh observation -> planner
```

VLA 输入 state 是：

```text
[eef_xyz (3), eef_axis_angle (3), gripper_qpos (2)]
```

VLA server 只做：

```text
图像 + proprio + instruction -> action chunk
```

simulator state 和 action chunk 执行仍由 LIBERO runtime 持有。

## 9. `finish` 与 reward

`finish` 调：

```python
payload = await runtime.success()
tool_context.metadata["finish"] = {
    "status": status,
    "summary": summary,
    "runtime_success": bool(payload.get("success")),
}
```

真正 reward：

```text
planner status="success" = planner 的主观记录
runtime.success()["success"] = LIBERO 官方 success predicate
reward = float(runtime.success()["success"])
```

`LiberoTask.run()` 在 agent 返回后再次调用 `runtime.success()`，生成 `TaskResult(reward, accuracy, extra_info)`。

## 10. 训练时 Gateway 如何参与

关键事实：

```text
Gateway 不执行 LIBERO tool。
```

训练时的关系是：

```text
planner -> Gateway 请求模型生成
planner 自己 -> Toolbox 执行 LIBERO tool
```

### 10.1 创建 session

`Agent Framework` 创建 Gateway session：

```python
session = await gateway_manager.create_session(
    session_id,
    sampling_params=dict(sampling_params),
)
```

返回：

```text
http://gateway-host:<port>/sessions/<session_id>/v1
```

`run_task()` 将 `session.base_url` 注入 task 的 `agent.model.base_url`。

### 10.2 Gateway 处理一次生成请求

```text
POST /sessions/{session_id}/v1/chat/completions
  -> openai_to_internal()
  -> GatewaySession.run_generation()
  -> MessageCodec 编码 messages + tools
  -> LLMServerClient.generate()
  -> 返回 token ids
  -> MessageCodec.decode_response()
  -> 解析 tool call
  -> commit 到 Gateway trajectory
  -> 返回 OpenAI assistant response
```

Gateway 负责：

- OpenAI/Anthropic API 适配
- session 路由
- tokenizer/chat template
- tool parser
- token-level trajectory
- response mask/logprobs
- session finalize 和 reward_info

Gateway 不负责：

- `libero_run_tool`
- `env.step`
- SAM3
- VLA
- LIBERO success predicate

## 11. 训练时完整闭环

```text
1. Agent Framework 创建 Gateway session
2. run_task() 将 session.base_url 注入 task config
3. LiberoTask.run() 创建 runtime、reset、ToolContext、ReActAgent
4. ReActAgent 生成 Toolbox schemas
5. ReActAgent 请求 Gateway /chat/completions
6. Gateway 编码对话和 tools，调用 LLM backend
7. Gateway 解析 assistant tool call 并记录 token trajectory
8. ReActAgent 执行 Toolbox.call()
9. LiberoRunTool -> runtime.run_tool()
10. EpisodeRuntime -> PrimitiveBackend.execute()
11. Backend 调 env / SAM3 / VLA
12. 返回 tool_result + fresh observation
13. ReActAgent 追加 role="tool"
14. 下一轮继续请求 Gateway
15. planner 调 finish
16. LiberoTask 调 runtime.success()
17. reward = official LIBERO success
18. reward_info + finalized trajectory -> TQ
```

## 12. 直接推理 vs RL 训练

### 直接推理

```text
ReActAgent -> OpenAICompatibleChatModel -> vLLM/SGLang endpoint
           -> assistant tool_calls -> 本地 Toolbox -> LIBERO runtime
```

### RL 训练

```text
ReActAgent -> Gateway session -> LLMServerClient / rollout model
           -> assistant tool_calls -> 本地 Toolbox -> LIBERO runtime
           -> reward / trajectory / TQ
```

LIBERO tool 执行链两种模式相同，差别只是训练时模型请求经过 Gateway，并被 token-level 记录。

## 13. 抓取放置示例

```text
1. libero_observe({})
2. libero_run_tool(name="segment", args={"prompt": "red mug", "camera": "agentview"})
3. libero_run_tool(name="move_to", args={"xyz": [mx, my, mz + 0.15], "gripper": "open"})
4. libero_run_tool(name="pi0_pick", args={"prompt": "pick up the red mug", "max_chunks": 12})
5. libero_run_tool(name="view_driver_state", args={})
6. libero_run_tool(name="segment", args={"prompt": "plate", "camera": "agentview"})
7. libero_run_tool(name="move_to", args={"xyz": [px, py, pz + 0.15], "gripper": "close"})
8. libero_run_tool(name="move_to", args={"xyz": [px, py, pz], "gripper": "close"})
9. libero_run_tool(name="release", args={})
10. finish(status="success", summary="placed the red mug on the plate")
```

planner 可以重新定位、重试 `pi0_pick`、从 `move_to` 切换到 `move_pose`、改用 `back_project`，或以 `failure/stuck` 结束。

## 14. 关键文件索引

### Planner / Agent

- `uni_agent/agents/react/agent.py`
- `uni_agent/agents/react/model.py`
- `uni_agent/agents/base.py`

### Tool 层

- `uni_agent/tools/base.py`
- `uni_agent/tools/libero.py`

### LIBERO 层

- `uni_agent/tasks/libero/protocol.py`
- `uni_agent/tasks/libero/task.py`
- `uni_agent/tasks/libero/runtime.py`
- `uni_agent/tasks/libero/primitives.py`
- `uni_agent/tasks/libero/services.py`
- `uni_agent/tasks/libero/environment.py`

### Gateway / 训练

- `uni_agent/gateway/gateway.py`
- `uni_agent/gateway/manager.py`
- `uni_agent/gateway/session/session.py`
- `uni_agent/gateway/session/codec.py`
- `uni_agent/framework/framework.py`
- `uni_agent/framework/task_runner.py`

### 设计文档

- `docs/source/concepts/harness-vla-libero-framework.md`
- `docs/source/concepts/libero-primitive-backends.md`
- `docs/source/concepts/libero-migration-design.md`

## 15. 最终心智模型

```text
Planner = 决定做什么
Toolbox = 把函数名映射到 host-side tool
LIBERO Tool = 将 planner 参数转换成 runtime 请求
Runtime = 持有 episode 状态，统一 reset/observe/run/success 协议
Primitive Backend = 将高层 primitive 转为控制循环、VLA 或 perception 调用
Environment = 真正拥有 simulator state
SAM3/VLA Service = 外部模型能力，不拥有 episode
Gateway = 训练时的模型生成和 trajectory 记录层
Task = 负责 episode 创建、评分和清理
```

最短调用链：

```text
assistant.tool_calls
  -> Toolbox.call
  -> LiberoRunTool.run
  -> LiberoRuntime.run_tool
  -> PrimitiveBackend.execute
  -> env / SAM3 / VLA
  -> LiberoPayload
  -> role="tool"
  -> 下一轮模型生成
```
