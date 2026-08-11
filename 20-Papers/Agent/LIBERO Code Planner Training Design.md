# LIBERO Code Planner + Harness Co-Training Design

状态: **大项目设计版 v0.2**  
目标: 一次性定义完整项目，不按简化 demo 设计。

## 0. 一句话定义

这个项目不是给当前 `uni2-agent` 接一个 code planner。

这个项目要做的是:

```text
用 code planner 作为 embodied agent，
用 uni-agent 风格 harness/task/rollout/trajectory 作为训练基础设施，
借 CaP-X 的 code-as-policy 执行形态，
再迁移 Harness-R1 的 failure trajectory -> harness patch -> rerun delta reward 范式，
最终形成一个能训练 code planner、也能训练/修补 harness 的协同训练系统。
```

核心对象有两个模型:

- **Code Planner**: 具身任务执行模型。输入任务、图像/视频、VLM 描述、结构化状态、工具结果、stdout/stderr，输出 subgoal-level Python code block。
- **Harness Engineer**: harness 修补模型。输入 frozen planner 的一批失败轨迹，输出受限 runtime patch，安装到 harness 生命周期 hook 上，通过 same-task rerun 的 delta reward 训练。

这两个模型不是同一个训练目标。

## 1. 已确认决策

这些是目前用户已经明确回答或同意的设计点。

| 决策项 | 已确认方向 |
| --- | --- |
| 项目边界 | 新建项目；不以当前 `uni2-agent` 现有 LIBERO tools 为中心 |
| Agent 形态 | code planner 作为 agent |
| 基础框架思想 | 借鉴 `uni-agent` 的 harness / task / rollout / trajectory 设计 |
| Code 形态 | CaP-X-style Python code block |
| API 边界 | harness-controlled API boundary；模型看似写 Python，但只能调用 allowlisted functions |
| Direct env | 不给模型 direct `env` / simulator / raw `env.step` |
| Multi-turn | 改成 step-wise assistant code turn，不把 CaP-X 的 `REGENERATE` / `FINISH` 作为主协议 |
| Finish | 用 `finish()` API 或特殊 code action，不用自然语言 `FINISH` |
| 执行粒度 | 一个 code block 做一个小 subgoal |
| Subgoal | episode 开始时允许模型输出对 goal 的多个 subgoal |
| Planner 模型 | VLM-code model，基于 Qwen3.5-9B |
| 视觉反馈 | harness 执行后返回 VLM 对图片/视频的描述、工具结果、stdout/stderr、错误信息；planner 可是 VLM-code model |
| Tool/API 来源 | 第一版参考 CaP-X reduced API 的十多种方法，因为符合正规测评 |
| Executor | 第一版做 restricted executor |
| Namespace | 多轮 code block 之间保留有限 session state，要求可序列化、可重放 |
| SFT 数据来源 | 用强 teacher model，例如 GPT-5.6 等，生成成功数据来训练 Qwen3.5-9B |
| SFT label | 训练 `plan comment + code`，不只是纯 code |
| Harness 修补参考 | 迁移 Harness-R1 思路 |

## 2. 非目标

第一版明确不做这些事情:

- 不复用当前 `uni2-agent` 暴露的 LIBERO planner primitive surface 作为硬约束。
- 不把项目目标写成 “把 code planner 接进 `uni2-agent`”。
- 不让 code planner 直接访问 simulator、raw `env.step`、MuJoCo/Libero 内部 state、后端 server client。
- 不照搬 CaP-X 的 unrestricted Python `exec`。
- 不把 CaP-X 的 `REGENERATE` / `FINISH` 作为训练主循环。
- 不一开始做一次性生成完整 episode 程序。
- 不只训练纯 code token；需要训练短 plan comment + code。
- 不让 harness engineer 第一版直接生成 git diff 改仓库源码。
- 不允许 harness patch 修改 reward evaluator、success function、task identity、simulator initial state。
- 不把 harness patch 设计成能直接控制机器人运动的 planner 替代品。

## 3. 总体架构

```text
Teacher Data Generation
  strong VLM/code teacher
  -> rollout with code planner harness
  -> success filtering
  -> SFT traces

Planner Training
  SFT Qwen3.5-9B VLM-code planner
  -> planner RL with environment reward
  -> frozen planner checkpoints

Harness-R1 Migration
  frozen planner baseline rollouts
  -> batch failure packets
  -> harness engineer generates typed runtime patch
  -> patch validation + sandbox compilation
  -> same-task patched rerun
  -> reward = patched metric - baseline metric
  -> harness engineer SFT / GRPO

Joint Evaluation
  planner only
  planner + fixed harness
  planner + trained harness engineer patch
  planner-SFT + harness-R1
  planner-RL + harness-R1
```

项目应该拆成五层:

```text
project/
  planner/
    code_planner_agent
    prompt_builder
    response_parser
    subgoal_state

  harness/
    runtime
    restricted_executor
    api_registry
    visual_feedback
    trace_writer
    reward_adapter
    error_taxonomy

  backends/
    libero_runtime
    capx_reduced_api_adapter
    vla_backend
    sam3_backend
    molmo_backend
    motion_backend

  training/
    sft_data_builder
    planner_rl_rollout
    trajectory_converter
    reward_manager

  harness_r1/
    failure_packet_builder
    patch_schema
    hook_runner
    patch_compiler
    patch_reward
    engineer_sft_builder
    engineer_rl_dataset_builder
    patch_eval
```

## 4. Code Planner 执行协议

### 4.1 主循环

第一版主循环是标准多轮 agent loop:

```text
task instruction
  -> harness builds initial observation
  -> optional raw images / image refs enter prompt
  -> VLM/harness generates visual description
  -> model outputs plan comment + Python code block
  -> harness parses code block
  -> restricted executor validates code
  -> code calls allowlisted harness APIs
  -> APIs call VLA / SAM3 / Molmo / grasp / motion / LIBERO backend
  -> executor captures stdout/stderr/errors
  -> harness computes reward/progress/success
  -> harness builds next observation
  -> repeat until finish/success/failure/budget
  -> trajectory writer emits SFT/RL data
```

### 4.2 Assistant 输出格式

模型每轮输出一个 code block。code block 里允许 plan comment 和思维链。

推荐格式:

```python
# Subgoal: locate the red mug and the wooden tray.
obs = get_observation()
scene = describe_scene(
    question="Locate the red mug and the wooden tray. Report whether both are visible."
)
print(scene.summary)

mug = point_prompt_molmo("red mug")
tray = point_prompt_molmo("wooden tray")
print("mug:", mug)
print("tray:", tray)
```

如果是 episode 第一轮，允许模型先写全局 subgoal plan:

```python
# Plan:
# 1. Locate the target object and destination.
# 2. Pick the target object.
# 3. Move it to the destination.
# 4. Release it.
# 5. Verify the final state.
state["subgoals"] = [
    "locate target object and destination",
    "pick target object",
    "move target to destination",
    "release target",
    "verify success",
]
state["current_subgoal"] = 0

obs = get_observation()
print("Initialized subgoals.")
```

### 4.3 Finish

`finish()` 是显式 API:

```python
finish(reason="The red mug appears to be inside the wooden tray.")
```

不要使用自然语言:

```text
FINISH
```

原因:

- parser 更简单。
- SFT/RL label 更稳定。
- 可以记录 `finish.reason`。
- 可以避免 CaP-X `_parse_multi_turn_decision()` 那种 `REGENERATE` 字符串启发式。

### 4.4 Code block 粒度

一个 code block 做一个小 subgoal，而不是一个 primitive，也不是完整 episode。

例子:

```text
Block 0: initialize subgoals and observe
Block 1: locate target and destination
Block 2: pick target
Block 3: place target
Block 4: verify and finish
```

每个 block 内可以有多个工具调用，暂时不需要预算限制:


## 5. CaP-X 继承关系

### 5.1 CaP-X 具体做法

CaP-X 的 `CodeExecutionEnvBase` 把 Python code string 当成 Gym action:

```text
obs, reward, terminated, truncated, info = code_env.step(python_code_string)
```

它的 trial loop 是:

```text
reset env
  -> build prompt with task + API docs
  -> model generates initial code
  -> _extract_code()
  -> env.step(code)
  -> capture stdout/stderr/reward/visual feedback
  -> build multi-turn decision prompt
  -> model replies FINISH or REGENERATE + code
  -> if REGENERATE, replace remaining code
  -> repeat
```

它的 executor 当前会注入:

```python
{
    "__name__": "__main__",
    "env": low_level_env,
    "APIS": api_name_to_api_object,
    "INPUTS": {},
    "RESULT": None,
    # plus API helper functions
}
```

这说明 CaP-X 能快速让模型写 robot code，但 executor 太开放。

### 5.2 我们要借的部分

- code-as-policy action space。
- Python code block 作为模型输出。
- API docstring 拼进 prompt。
- stdout/stderr 作为下一轮反馈。
- VLM image/video differencing。
- environment reward。
- successful artifacts 转 SFT 数据。
- GRPO/RLOO 训练 code agent 的路线。

### 5.3 我们不照搬的部分

- 不暴露 direct `env`。
- 不允许任意 import。
- 不允许 filesystem/network/process。
- 不使用 `REGENERATE` / `FINISH` 作为主训练协议。
- 不只保存 artifact，要保存训练友好的 raw messages、normalized trace、token trajectory。
- 不把 harness 固定死，后续要能被 Harness-R1-style engineer 修补。

## 6. 第一版 API 边界

用户倾向: 第一版按照 CaP-X reduced API 的十多种方法，因为符合正规测评。

### 6.1 LIBERO reduced API 候选

从 CaP-X 的 `FrankaLiberoApiReduced.functions()` 看，候选 API 包括:

```text
Observation:
  get_observation

Perception:
  segment_sam3_text_prompt
  segment_sam3_point_prompt
  point_prompt_molmo

Grasp:
  plan_grasp
  plan_grasp_from_point_clouds
  get_oriented_bounding_box_from_3d_points

Motion / Control:
  solve_ik
  move_to_joints
  goto_pose
  goto_home_joint_position
  open_gripper
  close_gripper

Point cloud / geometry:
  subsample_point_cloud
  filter_noise
```

如果使用 skill-library variant，还会增加:

```text
Geometry / reusable helpers:
  rotation_matrix_to_quaternion
  decompose_transform
  depth_to_point_cloud
  mask_to_world_points
  pixel_to_world_point
  transform_points
  interpolate_segment
  normalize_vector
  select_top_down_grasp
```

### 6.2 VLA extension

不需要有VLA，VLA并不稳定，不好训练code 

### 6.3 Harness-only API

为了训练稳定，harness 还应该提供一些不是 CaP-X 原生的 wrapper:

```text
describe_scene(question)
describe_transition(question)
finish(reason)
get_last_tool_result()
```

这些 API 不一定直接操作机器人，但对 VLM-code planner 很重要。

### 6.4 API 分层

不要把所有函数平铺给模型。建议分三层:

```text
Planner-visible APIs:
  get_observation
  describe_scene
  point_prompt_molmo
  segment_sam3_text_prompt
  segment_sam3_point_prompt
  plan_grasp
  goto_pose
  open_gripper
  close_gripper
  finish

Expert-visible APIs:
  solve_ik
  move_to_joints
  plan_grasp_from_point_clouds
  geometry helpers

Harness-internal APIs:
  raw simulator
  env.step
  backend clients
  reward evaluator
  task identity
  artifact store
```

第一版可以通过 harness config 控制是否给 planner 看 expert-visible APIs。

## 7. Restricted Executor

### 7.1 目标

模型写 Python，但 Python 世界由 harness 定义。

允许:

- variable assignment。
- `if` / bounded `for`。
- short helper function，可选。
- list/dict/string/number。
- `print()`。
- allowlisted APIs。
- allowlisted imports: 第一版建议只允许 `math` 和 `numpy as np`。

禁止:

- direct `env`。
- raw `env.step`。
- simulator state mutation。
- filesystem。
- network。
- subprocess。
- arbitrary import。
- dynamic eval/exec。
- unbounded loop。
- hidden global object mutation。
- access dunder/private names。

### 7.2 Globals

executor 提供:

```python
safe_globals = {
    "__builtins__": SAFE_BUILTINS,
    "np": numpy,
    "math": math,
    "state": session_state,
    "get_observation": harness.get_observation,
    "describe_scene": harness.describe_scene,
    "describe_transition": harness.describe_transition,
    "point_prompt_molmo": harness.point_prompt_molmo,
    "segment_sam3_text_prompt": harness.segment_sam3_text_prompt,
    "segment_sam3_point_prompt": harness.segment_sam3_point_prompt,
    "plan_grasp": harness.plan_grasp,
    "vla_pick": harness.vla_pick,
    "vla_place": harness.vla_place,
    "goto_pose": harness.goto_pose,
    "open_gripper": harness.open_gripper,
    "close_gripper": harness.close_gripper,
    "finish": harness.finish,
}
```

不提供:

```text
env
sim
robot
os
sys
subprocess
requests
open
socket
__import__
```

### 7.3 AST validation

建议借 Harness-R1 的 `code_runner.py` 思路，做 AST validation:

- 限制 AST 节点数。
- 禁止 `Import` / `ImportFrom`，或只允许 `import numpy as np` / `import math` 的受控 rewrite。
- 禁止 `While`。
- 禁止 `ClassDef`。
- 禁止 `With`。
- 禁止 `Lambda`。
- 禁止 `Global` / `Nonlocal`。
- 禁止访问下划线开头属性。
- 禁止 forbidden builtins。
- 限制 string literal 长度。
- 限制 helper function 数量。

### 7.4 Session state

多轮 block 之间保留:

- `state` dict。
- 可序列化变量。
- object refs / mask refs / pose refs。
- subgoal list。
- current_subgoal。
- last_tool_result summary。

不保留:

- simulator object。
- backend client。
- numpy 巨型 array，除非转成 artifact ref。
- file/socket/thread。
- raw image bytes。

建议所有跨步状态都放在显式 `state`:

```python
state["target_object"] = "red mug"
state["target_ref"] = mug
state["current_subgoal"] = 1
```

不要依赖任意 globals。

## 8. Observation / Visual Feedback 契约

### 8.1 Planner 输入

每一轮 planner prompt 包含:

```text
task_instruction
allowed_api_docs
current_subgoals / state summary
latest visual inputs:
  raw image / image refs, because planner is VLM-code model
  VLM visual description
  image/video differencing text
structured robot state:
  ee_pose
  gripper state
  holding candidate
  step index
  remaining budget
last execution:
  code block summary
  parsed tool calls
  stdout
  stderr
  tool results
  structured errors
reward/progress:
  reward_after_turn
  success flag if available
```

### 8.2 Raw image 与 VLM description 的关系

用户已确认 planner 是 VLM-code model。于是第一版可以给 planner raw image / image artifact。

但仍建议保留 VLM description:

- 它让 trace 更可读。
- 它能服务纯文本 teacher / evaluator。
- 它能给 harness engineer 分析失败原因。
- 它能降低多模态训练不稳定时的风险。

因此第一版 observation 同时存:

```text
image_refs
vlm_scene_description
vlm_transition_description
structured_state
tool_outputs
```

### 8.3 VLM feedback 类型

```text
initial_scene_description:
  第 0 步描述初始场景。

current_scene_description:
  每轮执行后描述当前场景。

transition_description:
  比较执行前后图像/视频，描述机器人做了什么、物体如何变化、任务是否完成。

failure_focused_description:
  当 tool 失败时，问 VLM 为什么可能失败，例如目标不可见、遮挡、抓取偏移。
```

## 9. SFT 设计

### 9.1 模型

目标 planner:

```text
Qwen3.5-9B VLM-code model
```

Teacher:

```text
强 VLM/code teacher，例如 GPT-5.6 等。
```

### 9.2 数据来源

数据来源:

```text
strong teacher successful rollouts
```


### 9.3 SFT 样本格式

每条样本是 multi-turn messages:

```json
{
  "episode_id": "...",
  "task_id": "...",
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "task + API docs + initial observation"},
    {"role": "assistant", "content": "# Plan: ...\nstate['subgoals'] = ...\n..."},
    {"role": "user", "content": "execution result + visual feedback + stdout/stderr"},
    {"role": "assistant", "content": "# Subgoal: pick target\n..."}
  ],
  "metadata": {
    "success": true,
    "final_reward": 1.0,
    "teacher_model": "...",
    "harness_config": "...",
    "api_surface": "capx_libero_reduced_v1"
  }
}
```

### 9.4 Loss mask

训练:

```text
assistant plan comment + code: loss_mask = 1
```

不训练:

```text
system prompt
task instruction
API docs
raw observation text
VLM feedback
tool result
stdout/stderr
error messages
```

重要: 用户明确要求训练 `plan comment思维链 + code`。因此 assistant label 不是纯 code。

推荐 assistant block:

```python
# Subgoal: verify whether the cube is grasped before placing.
obs = get_observation()
desc = describe_scene(question="Is the red cube held by the gripper?")
print(desc.summary)

if "held" not in desc.summary.lower():
    result = vla_pick("red cube")
    print(result)
else:
    state["current_subgoal"] = 2
```

不建议训练长篇自然语言推理。plan comment 应该短、可执行导向、和 code 紧密绑定。

## 10. Planner RL 设计

### 10.1 RL rollout 单位

每个 episode 内有多个 assistant code turns。

```text
prompt_t -> response_t(code block) -> execute_t -> observation_t+1
```

每个 `response_t` 都是模型生成 token，应进入 `response_mask=1`。

环境反馈、VLM feedback、stdout/stderr 进入下一轮 prompt，mask 为 0。

### 10.2 Reward

第一版建议:

```text
main reward = final LIBERO success
```

即:

```text
success: 1.0
failure: 0.0
```

辅助 diagnostics:

- invalid_code_rate。
- invalid_api_rate。
- timeout_rate。
- tool_failure_rate。
- perception_empty_rate。
- vla_failed_rate。
- motion_failed_rate。
- average_code_blocks。
- average_tool_calls。
- average_vlm_calls。
- success_after_n_blocks。

这些先不进主 reward。

### 10.3 Reward 写入 token

第一版有两个可选策略:

```text
Strategy A:
  只把 final reward 写到最后一个 assistant turn 的最后有效 token。

Strategy B:
  episode 内所有 assistant turns 的最后有效 token 都写同一个 final reward。
```

推荐先实现 A，因为简单稳定。后续做 ablation 比较 B。

### 10.4 与 uni-agent / verl 的关系

如果目标是长期 agent+harness co-training，应该优先保留 uni-agent-style token trajectory:

```text
prompt_ids
response_ids
response_mask
response_logprobs
rm_scores
uid / episode_id
finished
extra_info
```

CaP-X / verl GRPO 可以作为快速 baseline，但我们自己的 normalized trace 必须更完整，尤其要支持:

- 多轮 code turns。
- image refs。
- parsed tool calls。
- harness patch metadata。
- same-task rerun identity。

## 11. Trace Schema

必须同时保存三层。

### 11.1 Raw messages

用于 SFT / debug / replay:

```text
messages:
  system
  user task/API/observation
  assistant plan+code
  user execution feedback
  assistant plan+code
  ...
```

### 11.2 Normalized episode trace

用于数据分析、harness engineer failure packet、跨框架转换:

```text
episode_id
task_id
task_suite
task_instruction
seed
planner_model
planner_decoding_config
harness_config
api_surface
initial_artifacts
turns:
  - turn_id
    prompt_messages_ref
    assistant_text
    plan_comment
    code_block
    ast_valid
    parsed_calls
    execution_status
    stdout
    stderr
    error:
      code
      message
      traceback_ref
    tool_results
    visual_feedback:
      image_refs
      scene_description
      transition_description
      video_ref
    structured_state
    reward_after_turn
    done
final_reward
success
failure_type
artifact_refs
```

### 11.3 Token-level RL trajectory

用于 RL:

```text
prompt_ids
response_ids
response_mask
rm_scores
attention_mask
position_ids
uid
finished
non_tensor_batch:
  episode_id
  task_id
  image_refs
  normalized_trace_ref
  reward_info
```

## 12. Harness-R1 迁移设计

### 12.1 Harness-R1 原始范式

Harness-R1 训练的是 harness engineer。

原始流程:

```text
frozen target rollout
  -> batch failure packet
  -> harness engineer
  -> <think>...</think><patch>...</patch>
  -> parse, validate, sandbox hooks
  -> rerun same target on same task identities
  -> reward = patched metric - baseline metric
```

patch 不是 git diff，而是 typed JSON patch。当前 Harness-R1 主 action 是:

```json
{
  "type": "add_code_hook",
  "hook": "on_init | make_pre_hint | on_before_action | on_post_step",
  "code": "def hook(ctx, nb): ..."
}
```

hook 受 sandbox 限制:

- code 必须定义 `def hook(ctx, nb)`。
- 禁止 import。
- 禁止 IO。
- 禁止 eval/exec。
- 禁止 dynamic access。
- 限制 AST 节点数。
- 限制时间。
- hook 出错退化成 no-op。

### 12.2 迁移目标

迁移到本项目后，Harness Engineer 不是修 WebShop/ALFWorld/DBBench harness，而是修 embodied code planner harness:

```text
frozen code planner baseline rollouts
  -> embodied failure packets
  -> harness engineer generates embodied runtime patch
  -> validate patch
  -> install patch into embodied harness hooks
  -> rerun same LIBERO task identities / seeds
  -> reward = patched_success_rate - baseline_success_rate
```

### 12.3 Embodied patch schema

第一版 schema:

```json
{
  "schema_version": "embodied-harness-r1-patch-v1",
  "benchmark": "libero",
  "description": "short reusable description",
  "actions": [
    {
      "type": "add_code_hook",
      "hook": "on_episode_init | make_pre_plan_hint | on_before_tool_call | on_after_code_execute",
      "code": "def hook(ctx, nb):\n    ...\n    return {...}"
    }
  ]
}
```

第一版只允许 `add_code_hook`，不允许直接 git diff。

### 12.4 Embodied lifecycle hooks

推荐四个 hook:

```text
on_episode_init:
  episode reset 后执行。
  用于初始化 nb，注入 task-level skills/tool hints/visual policy。

make_pre_plan_hint:
  planner 生成下一段 code block 前执行。
  用于根据当前 state/error/visual feedback 给短提示。

on_before_tool_call:
  parsed tool call 执行前执行。
  用于 block 明显非法调用，或要求 planner 重新生成。

on_after_code_execute:
  一个 code block 执行后执行。
  用于更新 nb，注入 recovery hint，改变下一轮 visual feedback 策略。
```

后续可加:

```text
on_before_code_execute
on_after_tool_call
on_before_visual_feedback
on_after_visual_feedback
```

但第一版不要过多。

### 12.5 Hook ctx

`ctx` 是只读 evidence，不能让 hook 访问 simulator。

建议字段:

```text
ctx.task:
  task_id
  task_instruction
  suite

ctx.step:
  code_block_index
  remaining_blocks
  remaining_tool_calls

ctx.planner:
  last_plan_comment
  last_code_block
  parsed_calls

ctx.execution:
  stdout
  stderr
  error_code
  ast_valid
  timeout

ctx.state:
  ee_pose
  gripper_open
  holding_candidate
  current_subgoal
  repeated_subgoal_count
  perception_empty_count
  vla_failure_count
  motion_failure_count

ctx.visual:
  scene_description
  transition_description
  image_refs
  video_ref

ctx.tool:
  current_call
  last_results
  allowed_apis

ctx.predicates:
  invalid_api
  syntax_error
  perception_empty
  repeated_observation
  repeated_code_pattern
  no_progress
  object_not_visible
  target_grasped
  target_released
  success_visible
```

`nb` 是 per-episode scratch dict，用于 patch 自己维护状态。

### 12.6 Hook return contracts

第一版允许以下返回。

`on_episode_init`:

```python
return {
    "skills": [{"text": "Use visual differencing after every movement-heavy code block."}],
    "tool_hint": "Prefer VLA pick when segmentation repeatedly fails.",
    "visual_policy": "image_diff"
}
```

`make_pre_plan_hint`:

```python
return {
    "message": "The last two perception calls returned empty masks. Try point_prompt_molmo before SAM3 segmentation."
}
```

`on_before_tool_call`:

```python
return {
    "kind": "block_and_prompt",
    "message": "This call tries to use solve_ik with an invalid pose shape. Regenerate the code with a valid 7D pose."
}
```

`on_after_code_execute`:

```python
return {
    "kind": "inject_hint",
    "message": "The object did not move after the last pick attempt. Verify grasp before proceeding to place."
}
```

可选但需要谨慎:

```python
return {
    "kind": "set_visual_feedback_policy",
    "policy": "video_diff"
}
```

不建议第一版允许:

```text
rewrite_code
force_robot_action
direct_tool_result_override
reward_override
sim_state_patch
```

### 12.7 Patch reward

Harness engineer RL reward:

```text
score = patched_success_rate - baseline_success_rate
```

或:

```text
score = patched_average_reward - baseline_average_reward
```

第一版推荐:

```text
primary = delta_success_rate
diagnostics = delta_average_reward, delta_invalid_code_rate, delta_tool_failure_rate, delta_timeout_rate
```

invalid patch:

```text
score = 0.0 或 small negative
valid_patch = false
```

runtime no-op patch:

```text
score = 0.0
```

patch 造成退化:

```text
score < 0
```

### 12.8 Same-task identity

必须记录并校验:

```text
task_id
task_suite
task_instruction
seed
initial_state_hash
camera_config
planner_model
planner_decoding_config
VLM model / endpoint / decoding config
API surface version
harness base version
backend server versions
reward protocol version
```

baseline 和 patched rerun 必须相同。否则 delta reward 不可信。

### 12.9 Harness engineer SFT

SFT 数据:

```text
failure packet
  -> strong teacher engineer writes <think> + <patch>
  -> validate patch
  -> rerun
  -> keep valid positive or high-quality patches
  -> SFT harness engineer
```

Assistant label:

```text
<think>
分析 recurring failure pattern。
定位应该在哪个 lifecycle hook 修。
解释 patch 为什么不影响已成功轨迹。
</think>
<patch>
{...}
</patch>
```

### 12.10 Harness engineer RL

RL sample:

```json
{
  "prompt": [
    {"role": "system", "content": "You are a harness engineer..."},
    {"role": "user", "content": "failure packet + patch schema"}
  ],
  "label": "",
  "metadata": {
    "benchmark": "libero",
    "batch_tag": "...",
    "baseline_success_rate": 0.2,
    "baseline_rewards": {"task_0": 0.0, "task_1": 1.0},
    "target_planner_model": "qwen3.5-9b-vlm-code-sft",
    "harness_base_version": "...",
    "task_identity_manifest": "..."
  }
}
```

Reward function:

```text
parse patch
  -> validate schema
  -> compile hooks
  -> install overlay
  -> rerun frozen planner on same task identities
  -> compute delta metric
  -> return reward + diagnostics
```

## 13. Failure Packet 设计

Harness engineer 不应该看单条失败，而应该看一批失败。

### 13.1 Packet 内容

```text
packet_id
benchmark: libero
task_suite
target_planner_model
harness_base_version
api_surface_version
baseline_metrics:
  success_rate
  average_reward
  invalid_code_rate
  tool_failure_rate
  timeout_rate
episodes:
  - episode_id
    task_id
    task_instruction
    seed
    success
    final_reward
    failure_type
    compressed_timeline:
      - turn_id
        plan_comment
        code_excerpt
        parsed_calls
        stdout_excerpt
        stderr_excerpt
        visual_feedback_text
        error_code
        progress_summary
    recurring_signals:
      perception_empty_count
      vla_failure_count
      motion_failure_count
      repeated_subgoal_count
      no_progress_count
positive_examples:
  optional successful traces for contrast
patch_schema
rules
```

### 13.2 Failure taxonomy

第一版错误码:

```text
syntax_error
runtime_error
invalid_api
timeout
tool_failed
perception_empty
perception_wrong_object
vla_failed
grasp_failed
motion_failed
no_progress
repeated_subgoal
premature_finish
budget_exceeded
safety_violation
success_but_not_finished
finish_but_not_success
```

### 13.3 Failure packet compression

具身轨迹很长，不能把所有图像和 full stdout/stderr 都塞进 engineer prompt。

需要:

- 只放 image refs，不直接放大图。
- VLM 生成 failure summary。
- 每个 turn 截断 stdout/stderr。
- code 只保留关键片段。
- grouped failures 按 recurring failure 聚类。

## 14. Planner 与 Harness 的协同训练路线

### Phase 0: 项目骨架

目标:

- 建新项目。
- 接 LIBERO runtime。
- 接 CaP-X reduced API adapter。
- 实现 restricted executor。
- 实现 trace writer。
- 实现 baseline evaluation。

产物:

```text
可以让 strong teacher / Qwen planner 输出 code block 并跑 episode。
可以保存 normalized traces。
```

### Phase 1: Teacher 数据生成

目标:

- 用强 teacher model 生成 successful rollouts。
- 成功过滤。
- 转 SFT messages。

产物:

```text
planner_sft_train.jsonl
planner_sft_val.jsonl
```

### Phase 2: Planner SFT

目标:

- SFT Qwen3.5-9B VLM-code planner。
- 学会 API 语法、subgoal plan、错误恢复基本格式。

指标:

- code parse success。
- API validity。
- execution success。
- LIBERO success。
- average code blocks。
- average tool calls。

### Phase 3: Planner RL

目标:

- frozen harness 下训练 planner。
- 主 reward 用 final success。

产物:

```text
planner_rl_checkpoint
planner_rl_rollout_traces
```

### Phase 4: Harness failure packet

目标:

- freeze planner。
- 跑 baseline rollouts。
- 收集失败批次。
- 构造 harness engineer prompts。

产物:

```text
harness_failure_packets.jsonl
```

### Phase 5: Harness engineer SFT

目标:

- 用强 teacher engineer 生成 patches。
- validate + rerun。
- 保留有效正收益 patches。
- SFT harness engineer。

产物:

```text
harness_engineer_sft_checkpoint
```

### Phase 6: Harness engineer RL

目标:

- 用 GRPO/RLOO 训练 harness engineer。
- reward 为 same-task rerun delta success。

产物:

```text
harness_engineer_rl_checkpoint
```

### Phase 7: Joint evaluation

评估矩阵:

```text
Base planner
Planner SFT
Planner RL
Planner SFT + hand-designed harness
Planner SFT + harness engineer SFT patch
Planner SFT + harness engineer RL patch
Planner RL + harness engineer RL patch
```

## 15. 评估协议

### 15.1 Planner metrics

```text
success_rate
average_reward
code_parse_success_rate
valid_api_call_rate
invalid_code_rate
timeout_rate
tool_failure_rate
perception_empty_rate
motion_failure_rate
vla_failure_rate
average_turns
average_tool_calls
average_vlm_calls
success_by_task_suite
```

### 15.2 Harness engineer metrics

```text
patch_valid_rate
hook_compile_success_rate
runtime_noop_rate
patched_success_rate
baseline_success_rate
delta_success_rate
patched_average_reward
baseline_average_reward
delta_average_reward
regression_rate_on_baseline_successes
improvement_rate_on_baseline_failures
infra_failure_rate
```

### 15.3 必须报告 paired comparison

任何 harness patch 结果都必须 paired:

```text
same task ids
same seeds
same initial states
same frozen planner
same planner decoding params
same VLM params
same backend versions
```

不能拿不同运行条件的 success rate 相减。

## 16. 与当前三个仓库的关系

### 16.1 `/home/eren/codes/cap-x`

用作:

- code-as-policy reference。
- reduced API reference。
- visual differencing reference。
- CaP-RL reference。
- successful artifacts conversion reference。

不直接照搬:

- unrestricted executor。
- direct `env` access。
- `REGENERATE` / `FINISH` 主循环。

### 16.2 `/home/eren/codes/uni-agent`

用作:

- Task / Agent / Harness / Sandbox 分层思想。
- Gateway / token-level trajectory capture 思想。
- SFT/RL 数据格式参考。

### 16.3 `/home/eren/codes/Harness-R1`

用作:

- failure packet。
- typed patch schema。
- code hook sandbox。
- same-batch rerun reward。
- harness engineer SFT + online GRPO。

不直接照搬:

- WebShop/ALFWorld/DBBench 的具体状态机。
- textual action rewrite。
- benchmark-specific predicates。

## 17. 关键工程模块

### 17.1 `planner/code_planner_agent.py`

职责:

- 构造 prompt。
- 调模型。
- 解析 assistant code block。
- 管理 multi-turn。
- 接收 harness observation。
- 发出 `finish()`。

### 17.2 `harness/restricted_executor.py`

职责:

- AST validate。
- safe globals。
- session state。
- stdout/stderr capture。
- timeout。
- tool call budget。
- structured exception。

### 17.3 `harness/api_registry.py`

职责:

- 注册 allowlisted APIs。
- 从 CaP-X reduced API adapter 导入函数。
- 包装 VLA/SAM3/Molmo/motion。
- 给 prompt 生成 API docs。
- 记录 parsed tool calls。

### 17.4 `harness/visual_feedback.py`

职责:

- capture image/video refs。
- call VLM for scene description。
- call VLM for transition description。
- manage wrist camera / multiview。
- cache VLM outputs。

### 17.5 `harness/trace_writer.py`

职责:

- 保存 raw messages。
- 保存 normalized episode trace。
- 保存 artifacts。
- 输出 SFT JSONL。
- 输出 RL token trajectories。
- 输出 Harness-R1 failure packets。

### 17.6 `harness_r1/patch_schema.py`

职责:

- 定义 `embodied-harness-r1-patch-v1`。
- parse `<patch>`。
- normalize patch。
- validate action/hook/effect。

### 17.7 `harness_r1/hook_runner.py`

职责:

- 借 Harness-R1 sandbox 思路。
- compile hook。
- run hook。
- hook failure -> no-op。
- normalize hook return。

### 17.8 `harness_r1/patch_reward.py`

职责:

- 读取 RL sample metadata。
- 安装 patch overlay。
- same-task rerun frozen planner。
- 计算 delta success/reward。
- 缓存 reward。
- 区分 infra failure 与 bad patch。

## 18. 一次性问题清单

下面这些是我现在仍需要你回答的全部关键问题。不是让项目停下来，而是为了把 v0.2 文档收敛成 v1.0 规格。

### A. 项目边界

1. 新项目名字叫什么？是否仍放在 `/home/eren/codes/` 下？
2. 是否完全独立 repo，还是作为 `uni2-agent` sibling project？
3. 是否需要保留 `uni-agent` 的命名体系，例如 Task、Agent、Harness、Gateway？

### B. Planner 模型与输入

4. Qwen3.5-9B VLM-code model 的具体 checkpoint/source 是什么？
5. Planner prompt 里 raw image 是直接给模型，还是 image ref + VLM description 双路都给？
6. VLM description 是由同一个 planner 模型生成，还是由单独 VLM backend 生成？
7. Episode 第一轮是否强制输出 global subgoal plan？
8. 后续每轮是否必须写 `# Subgoal: ...` comment？

### C. API surface

9. 第一版到底用 `FrankaLiberoApiReduced`，还是 `FrankaControlApiReduced`，还是二者抽象合并？
10. 是否启用 skill-library variant 的 geometry helper？
11. VLA API 第一版具体叫什么: `vla_pick`、`vla_place`、`vla_act`，还是别的？
12. 是否允许 planner 使用 `solve_ik` / `move_to_joints` 这种 expert-level API？
13. 是否允许 planner 使用 point cloud helper，例如 `mask_to_world_points`？

### D. Executor

14. 是否允许 `import numpy as np`，还是由 executor 预注入 `np`？
15. 是否允许模型定义 helper function？
16. 是否禁止 `while`？我建议禁止。
17. 是否允许 bounded `for`？我建议允许。
18. 单个 code block 最大字符数是多少？
19. 单 episode 最大 code blocks 是 8、10、12 还是更多？
20. 每 block 最大 tool calls 是 3、5 还是更多？

### E. SFT

21. Teacher 数据生成用哪些强模型？
22. Teacher rollout 是直接用同一个 harness，还是先借 CaP-X harness 生成再转换？
23. SFT 数据是否只保留 successful episodes，还是保留 failed-but-recovered episodes？
24. Plan comment 的格式是否强制统一？
25. SFT 是否训练 initial global subgoal plan？

### F. Planner RL

26. Planner RL 优先用 GRPO、RLOO，还是 PPO？
27. Reward 写入策略先用 final turn，还是所有 assistant turns 共享 episode reward？
28. 是否给 invalid code 小负分？
29. 是否引入 KL penalty / reference model？
30. 是否把 shaped reward 只记录，不进训练？

### G. Harness-R1 迁移

31. Harness engineer 是否也用 Qwen3.5-9B，还是另一个模型？
32. Harness engineer 第一版只做 SFT，还是直接 SFT + GRPO？
33. Patch 第一版是否只允许 `add_code_hook`？
34. Patch hooks 是否采用本文推荐的四个: `on_episode_init`、`make_pre_plan_hint`、`on_before_tool_call`、`on_after_code_execute`？
35. 是否允许 patch 返回 `set_visual_feedback_policy`？
36. 是否允许 patch 返回 `retry_with_modified_args`？
37. 是否严格禁止 patch rewrite planner code？
38. 是否严格禁止 patch force robot action？

### H. Evaluation

39. LIBERO 任务集第一版用哪个 split？
40. 每个 patch reward rerun batch size 是 5、10、还是更大？
41. baseline/patched rerun 是否固定同一 seed？
42. VLM backend 是否 deterministic？
43. 是否需要真实机器人迁移评测，还是先只做 simulation？

## 19. 当前推荐默认答案

如果你不想逐个回答，默认按下面走:

```text
项目:
  新建 /home/eren/codes/embodied-code-planner-r1

Planner:
  Qwen3.5-9B VLM-code
  输入 raw image + VLM description
  输出 plan comment + code
  第一轮强制 global subgoal plan
  每轮强制 # Subgoal comment

API:
  LIBERO 第一版参考 FrankaLiberoApiReduced
  启用基础 reduced API
  skill-library helper 可配置开启
  增加 vla_pick / vla_place / finish / describe_scene

Executor:
  restricted executor
  np/math 预注入
  禁止 while/import/direct env
  允许 bounded for 和短 helper function
  max code blocks = 10
  max tool calls per block = 5

SFT:
  strong teacher successful + recovered traces
  label = plan comment + code
  context = observation / VLM feedback / stdout/stderr / tool outputs

Planner RL:
  GRPO first
  main reward = final success
  diagnostics only shaped metrics
  reward first written to final assistant turn

Harness-R1:
  harness engineer separate model
  first SFT, then GRPO
  patch only add_code_hook
  hooks = on_episode_init, make_pre_plan_hint, on_before_tool_call, on_after_code_execute
  no source code diff in v1
  no reward patch
  no sim state patch
  no force robot action in v1

Evaluation:
  same-task same-seed paired rerun
  reward = delta_success_rate
  report regression rate on baseline successes
```

## 20. 最终目标图

```text
            strong teacher
                 |
                 v
       successful planner traces
                 |
                 v
       SFT Qwen3.5-9B planner
                 |
                 v
        planner RL with harness
                 |
                 v
          frozen planner rollout
                 |
                 v
        batch failure packets
                 |
                 v
          harness engineer
                 |
          <think> + <patch>
                 |
                 v
    validate + sandbox + install hooks
                 |
                 v
       same-task patched rerun
                 |
                 v
     delta reward trains harness engineer
                 |
                 v
   planner + trained harness co-evaluation
```

这个设计保留三条可发表/可实验主线:

1. **Code Planner SFT/RL**: VLM-code planner 是否能通过 step-wise code-as-policy 学会 LIBERO manipulation。
2. **Harness-R1 for Embodiment**: failure trajectory -> runtime patch -> rerun delta reward 是否能修补具身 harness。
3. **Planner + Harness Co-Training**: 训练 planner 与训练 harness engineer 是否有互补收益。
