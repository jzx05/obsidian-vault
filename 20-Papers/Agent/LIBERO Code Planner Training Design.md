# LIBERO Code Planner + Harness Co-Training Design

当前状态: **设计问题草案 / Round 1**。

这份文档记录一个 embodied code planner + harness 协同训练项目的设计需求。它不是“把 code planner 接入当前 `uni2-agent` 的现有 LIBERO tools”，而是重新定义一个以 code planner 为 agent、以 uni-agent 风格 harness 为训练基础设施的项目。

## 背景

目标不是在当前 `uni2-agent` 中接入 code planner，也不是复用当前 `uni2-agent` 已经暴露的 LIBERO 工具函数。当前 `uni2-agent` 里的 planner primitive surface 可以先视为不存在。

真正目标是:

- 用 **code planner 作为 agent**。
- 借鉴 `uni-agent` 的整体 harness / task / rollout / trajectory 设计。
- 设计一个既可以训练 code planner，又可以后续训练、修补、改进 harness 的协同训练项目。
- 模型根据任务、VLM 对图像/视频的描述、结构化状态、工具执行结果、stdout/stderr 和错误信息，自由组合底层工具完成 LIBERO 等具身任务。
- harness 负责执行模型生成的 code block、调用底层工具、收集反馈、计算 reward、记录 SFT/RL 轨迹，并在长期目标里成为可被修补或训练的对象。

因此，这个项目的中心不是 “planner 调哪些现有 `uni2-agent` tools”，而是:

```text
code planner agent
  <-> code execution harness
  <-> embodied tool/runtime backends
  <-> task/reward/evaluation
  <-> SFT/RL trajectory capture
  <-> harness repair / harness training loop
```

## 初始设计倾向

推荐方向是 **step-wise CaP-X-style restricted code planner**:

1. 模型每轮生成一个参考 CaP-X 的 Python code block。
2. harness 在环境里执行该 block。
3. 执行后把 VLM 对图片/视频的描述、工具结果、stdout/stderr 和错误信息返回给模型。
4. 模型继续生成下一步代码，直到调用 `finish()`、环境成功、失败或耗尽预算。
5. SFT 和 RL 都只训练模型生成的代码 token；环境观察和工具输出只作为上下文。

这条路线保留 code-as-policy 的表达力，同时比一次性生成完整 episode 程序更适合图像闭环、失败恢复和 RL credit assignment。

## CaP-X 具体怎么做

下面是根据 `/home/eren/codes/cap-x` 当前代码和文档整理的 CaP-X 机制。重点文件包括:

- `capx/envs/tasks/base.py`
- `capx/envs/trial.py`
- `capx/utils/launch_utils.py`
- `env_configs/cube_lifting/franka_robosuite_cube_lifting_multiturn_vdm_reduced_api.yaml`
- `docs/rl-training.md`

### 1. 核心抽象: CodeExecutionEnvBase

CaP-X 把 embodied task 包装成一个 “文本 action space 的 Gym env”。

普通 Gym env 是:

```text
obs, reward, done, info = env.step(action)
```

CaP-X 变成:

```text
obs, reward, terminated, truncated, info = code_env.step(python_code_string)
```

其中 `python_code_string` 是模型生成的代码。也就是说，模型不是输出离散 action、JSON tool call 或 7D robot action，而是输出一段 Python 程序。harness 执行这段程序，程序内部再调用 robot APIs / perception APIs / motion APIs。

### 2. Prompt 如何构造

CaP-X 的 prompt 主要由三部分组成:

- system prompt: 告诉模型它是生成 Python code 的 assistant。
- task prompt: 例如 “pick up the red cube and lift it”。
- API docs: 把 config 中指定 API 的 docstring 拼进 prompt，让模型知道能调用哪些函数。

例如 cube lifting 的 multi-turn config 里写的是:

```text
You are controlling a Franka Emika robot with API described below.
Goal: pick up the red cube and lift it.
You may write python code comments for reasoning but ONLY write the executable Python code and do not write it in code fences.
After this code executes, you will receive the current console stdout and stderr ...
The functions (APIs) below are already imported to the environment.
```

API 由 YAML 里的 `apis` 字段决定，例如:

```yaml
apis:
  - FrankaControlApiReduced
```

`CodeExecutionEnvBase._get_complete_prompt()` 会调用每个 API 的 `combined_doc()`，把 API 文档拼进 prompt。

### 3. 模型初始输出

CaP-X trial 开始后:

1. reset environment。
2. 可选 capture 初始图像。
3. 如果启用 image/video differencing，会调用 VLM 描述初始状态。
4. 把 task prompt、API docs、可选图像/视觉描述发给模型。
5. 模型生成初始 Python code。
6. `_extract_code()` 从模型回复中抽取 Python code。

当前 `_extract_code()` 的逻辑很简单:

- 如果有 ```python fenced block，就取里面内容。
- 如果没有 fenced block，就把整个 content 当 code。
- 实际返回一个 `list[str]`，但通常只有一个 code block。

所以 CaP-X 的 “code block” 不是强结构化 AST，也不是 primitive-level DSL，而是一段普通 Python 字符串。

### 4. 代码执行环境

CaP-X 的 `CodeExecutionEnvBase._exec_user_code()` 使用 Python `exec()` 直接执行模型代码。

执行 namespace 是 persistent globals，每个 episode reset 时初始化，episode 内多轮 code block 之间保留变量。

它注入的关键对象包括:

```python
{
    "__name__": "__main__",
    "env": low_level_env,
    "APIS": api_name_to_api_object,
    "INPUTS": {},
    "RESULT": None,
    # plus each API function directly bound into globals
}
```

每一步执行前还会更新:

- `obs`: 当前 observation。
- `env`: 当前 low-level env。
- `APIS`: 当前 API dict。
- API helper functions: 重新绑定到 globals。

这意味着 CaP-X 里的模型代码可以直接调用 API helper，也能直接访问 `env`。从研究原型角度这很方便；从训练稳定性和安全边界角度，这太开放。

对我们要做的项目来说，CaP-X 的 executor 可以借鉴，但不能原样照搬。第一版应该更像:

```text
model code
  -> restricted executor
  -> allowlisted harness APIs
  -> backend tools / simulator
```

而不是:

```text
model code
  -> unrestricted exec
  -> direct env access
```

### 5. stdout/stderr 和错误反馈

CaP-X 执行代码时会捕获 stdout 和 stderr:

- `print()` 输出进入 stdout。
- Python exception traceback 进入 stderr。
- 如果执行成功，`sandbox_rc=0`。
- 如果执行失败，`sandbox_rc=1`。

然后 `step()` 返回:

```python
info = {
    "sandbox_rc": 0 or 1,
    "stdout": captured_stdout,
    "stderr": captured_stderr,
    "task_prompt": task_prompt,
    "task_completed": task_completed,
}
```

multi-turn prompt 会把刚刚执行的代码、stdout、stderr 写回给模型，让模型决定下一步是继续修代码还是结束。

### 6. Reward 和 termination

CaP-X 的 reward 通常来自 low-level env:

```python
reward = self.low_level_env.compute_reward()
```

默认逻辑里:

- `reward == 1.0` 时认为 `terminated=True`。
- `task_completed()` 如果 low-level env 有这个方法，会写到 info。
- truncated 通常由 simulator step count 是否超过 max_steps 决定。

所以 CaP-X 的 reward 是环境可验证 reward，不是 LLM judge reward。这一点很适合 RL。

### 7. Multi-turn loop 不是标准一步一动作

CaP-X 的 multi-turn 机制不是严格的 “每轮 planner 输出一步动作”。它更像 “执行代码 -> 让模型判断 FINISH 还是 REGENERATE”。

实际流程:

```text
reset env
  -> build initial prompt
  -> model generates initial code
  -> extract code block
  -> execute code block with env.step(code)
  -> capture stdout/stderr/reward/visual feedback
  -> build multi-turn decision prompt
  -> model replies:
       FINISH
       or REGENERATE + new Python fenced code block
  -> if REGENERATE:
       delete remaining unexecuted code blocks
       append regenerated code blocks
       execute new code
  -> repeat until FINISH / success / truncation / limit
```

CaP-X 用 `MULTITURN_LIMIT = 10` 控制最多多轮次数。

关键点:

- 初始 prompt 要求模型直接输出可执行 Python code，不要 fenced block。
- multi-turn prompt 要求模型输出 `FINISH` 或 `REGENERATE` + fenced Python code。
- `_parse_multi_turn_decision()` 只要看到 `REGENERATE` 就认为要 regenerate，否则默认 finish。
- 当前逻辑更偏 “代码修复/再生成”，不是严格 “一步一 primitive action”。

### 8. Visual feedback / visual differencing

CaP-X 支持几类视觉反馈:

- `use_visual_feedback`: 把当前图像直接放进 prompt，前提是主模型是 VLM。
- `use_img_differencing`: 用 VLM 比较前后两帧图像，生成文本描述。
- `use_video_differencing`: 用 VLM 看当前 turn 的执行视频，生成文本描述。
- `use_wrist_camera`: 可选加入腕部相机图像/视频。

对于 text/code planner 来说，最关键的是 image/video differencing:

```text
previous image/video + current image/video + task
  -> VLM describes what changed and whether task appears complete
  -> text feedback inserted into next planner prompt
```

这和你的设想高度一致: planner 不一定直接吃 raw image，而是吃 harness/VLM 压缩后的视觉状态描述。

### 9. Artifact 保存

CaP-X 每个 trial 会保存很多调试产物:

- `code.py`: 最终拼起来的代码。
- `raw_response.sh`: 初始模型原始输出。
- `all_responses.json`: 每轮 code block、decision、prompt、reasoning。
- `summary.txt`: reward、sandbox rc、stdout/stderr、是否完成、regeneration 次数。
- `visual_feedback_XX.png`: 每轮视觉反馈图。
- per-turn video / combined video。
- 可选保存 initial prompt 和 multi-turn prompt。

这些 artifact 对构造 SFT 数据很有价值，但还不是完整的训练 trajectory。要接 SFT/RL，需要额外定义:

- messages schema。
- 哪些 token 训练，哪些 token 只是 context。
- episode-level reward 如何写入 token-level `rm_scores`。
- stdout/stderr/VLM feedback 如何进入下一轮 prompt。
- code block 与 tool calls 如何结构化记录。

### 10. CaP-RL

CaP-X 文档中的 CaP-RL 使用 GRPO 训练 code-as-policy agents。

配置方式:

- `MODEL_PATH`: 训练的 HF 模型。
- `DATA_SOURCE`: code execution env 名字，例如 `franka_lift_code_env`。
- `GROUP_SIZE`: GRPO 每个 prompt 的 rollout 数。
- custom reward function: `verl_agent_reward/capx_franka_reward.py:compute_score`。
- reward manager 把 score 写到 response 的最后有效 token。

这说明 “code-as-policy + verifiable environment reward + GRPO” 是可跑的。

但它不等价于我们最终要做的项目，因为:

- CaP-X 主要训练 code agent，不训练 harness。
- executor 边界比较开放。
- multi-turn 是 `REGENERATE` / `FINISH`，不是天然的 step-wise planner token trajectory。
- visual feedback 更像 prompt engineering 和 artifact 记录，还没有变成严格的 SFT/RL schema。

## 和 CaP-X 的继承关系

我们应该借 CaP-X 的这些部分:

- code-as-policy action space。
- Python code block 作为模型输出。
- code execution env 把 Python code 当 action。
- stdout/stderr 回传给模型。
- VLM image/video differencing 作为视觉反馈压缩器。
- successful trial artifacts 可转 SFT 数据。
- environment reward 可接 GRPO/RLOO。

我们不应该原样照搬这些部分:

- 不直接暴露 `env` 给模型代码。
- 不允许任意 import。
- 不让模型直接读写文件、网络、进程。
- 不把 multi-turn 只做成 `REGENERATE` / `FINISH` 二选一。
- 不只保存 artifact，而要保存训练友好的 token/message/trace schema。
- 不把 harness 当固定脚手架；长期目标是 harness 也能被修补和训练。

## 训练项目的核心问题

下面是我目前还不明白、必须问清楚的设计问题。每个问题下面给出我的推荐答案。

### Q1 - 项目边界

**问题:** 这个项目是继续放在 `/home/eren/codes/uni2-agent` 里演化，还是新建一个独立项目，只借鉴 `uni-agent` 的 harness 设计？

**推荐答案:** 新建独立子项目或独立包，暂时不要和当前 `uni2-agent` 的 LIBERO tools 强耦合。可以复用代码、文档和思想，但项目边界要服务于 “code planner + harness co-training”，否则会被现有 `uni2-agent` 的工具设计牵着走。

**待确认:** 代码落点、包名、是否还叫 `uni2-agent`。

### Q2 - Agent 输出契约

**问题:** 模型每轮输出到底是 CaP-X 原生 Python code，还是带强约束的 Python action block？

**推荐答案:** 第一版用 **CaP-X-style Python code block + harness-controlled API boundary**。也就是模型看起来在写 Python，但 executor 只给它 allowlisted functions，不给 direct `env`。这样保留 code generator 的表达力，同时为 SFT/RL 提供稳定动作空间。

**待确认:** 是否允许 import `numpy`、是否允许定义 helper function、是否允许 class、loop、condition。

### Q3 - Multi-turn 语义

**问题:** 你想保留 CaP-X 的 `REGENERATE` / `FINISH` 格式，还是改成每轮直接输出下一段 code？

**推荐答案:** 改成更 agent 化的 step-wise loop:

```text
observation
  -> assistant code block
  -> execute
  -> observation
  -> assistant code block
```

不要强依赖 `REGENERATE`。`finish()` 应该是 code API 或特殊 action，而不是自然语言 `FINISH`。CaP-X 的 `REGENERATE` 适合评测时修代码，但对 SFT/RL 的 mask 和 credit assignment 不如标准 assistant turn 清楚。

**待确认:** 是否仍保留 `REGENERATE` 作为兼容 CaP-X trace 的导入格式。

### Q4 - 执行粒度

**问题:** 一个 code block 允许做多大？一次完整 episode、一段 subgoal、还是 1-3 个工具调用？

**推荐答案:** 第一版限制为小 block: 最多 1-3 个 tool calls 或一个小子目标。不要一次性生成完整 episode 程序。具身任务状态变化大，尤其依赖图像反馈，长程序失败后很难定位 reward credit。

**待确认:** 初始 budget 是每 episode 8、10、12 还是更多 code blocks。

### Q5 - 视觉反馈契约

**问题:** code planner 第一版是否直接吃 raw image，还是只吃 VLM 对图像/视频的描述？

**推荐答案:** 第一版主要吃 **VLM visual description / visual differencing text**。raw image 作为 artifact 保存，可选给 VLM，不强制给 planner。这样可以先训练 text/code model，也避免多模态 rollout 系统复杂度直接爆炸。

**待确认:** planner 模型是否确定是纯文本 code LLM，还是 VLM-code model。

### Q6 - 底层工具 API

**问题:** harness 暴露给 code planner 的底层方法有哪些？是 VLA/SAM/Molmo2/motion primitive，还是更低层的 robot control API？

**推荐答案:** 暴露两层，但模型默认只看上层:

- planner-visible APIs: `observe()`、`describe_scene()`、`query_object()`、`segment()`、`vla_pick()`、`move_to()`、`open_gripper()`、`close_gripper()`、`finish()`。
- harness-internal APIs: raw simulator、IK、motion backend、VLA server、SAM/Molmo server、reward evaluator。

模型不应该直接碰 harness-internal APIs。

**待确认:** VLA 应该是一个 callable tool，还是作为底层 primitive 由 code planner 通过更抽象的 `pick(object)` 间接调用。

### Q7 - Executor 安全边界

**问题:** 模型代码在什么环境里执行？完全 Python `exec`，还是 restricted executor？

**推荐答案:** 第一版就做 restricted executor，不要照搬 CaP-X 的 direct `env` access。至少需要:

- allowlisted globals。
- 禁止 direct `env`。
- 禁止 filesystem/network/process。
- import allowlist，例如只允许 `math`、`numpy`。
- timeout。
- tool call count budget。
- stdout/stderr 截断。
- exception 结构化。

**待确认:** 是否为了快速原型先允许 full Python，然后在训练前收紧。

### Q8 - Namespace 是否持久

**问题:** 多轮 code block 之间是否保留 Python 变量？

**推荐答案:** 保留有限 session state，但必须可序列化和可重放。允许保存字符串、数字、list/dict、点、mask id、object id、pose id，不允许保存 simulator object、open file、socket、thread、backend client。

**待确认:** 是否需要显式 `state` dict，例如 `state["target"] = ...`，而不是任意 globals。

### Q9 - SFT 数据来源

**问题:** 第一批 SFT 数据从哪里来？

**推荐答案:** 三路并行:

- 手写 canonical traces: 覆盖每个 API 的正确调用方式。
- scripted/oracle demos: 自动转成 step-wise code traces。
- CaP-X successful artifacts: 把 `all_responses.json`、`code.py`、stdout/stderr、visual feedback 转换成 messages。

**待确认:** 你是否已经有 LIBERO oracle/demo，可以转成 code traces。

### Q10 - SFT label

**问题:** SFT 到底训练哪些内容？只训练 code，还是训练 plan comment + code？

**推荐答案:** 只把 assistant 的 code block 作为主要 label。允许代码开头有短注释，但不要训练长自然语言推理。observation、VLM feedback、tool output、stdout/stderr 全部作为 context，loss mask 为 0。

**待确认:** 是否要强制 “只输出代码，不输出解释”。

### Q11 - RL trajectory

**问题:** RL 是按每个 code block 一条 response，还是整个 episode 拼成一条长 response？

**推荐答案:** 按每个 model turn 捕获 response，但 episode 内共享同一个 final reward。也就是多轮 rollout 中每个 code block 都是可训练 response token，环境反馈进入下一轮 prompt，mask 为 0。reward 可以先稀疏写到最终 turn，后续再试 episode reward 分配到所有 assistant turns。

**待确认:** 你希望优先兼容 uni-agent Gateway 的 token-level trajectory，还是优先兼容 CaP-X / verl 当前 GRPO pipeline。

### Q12 - Reward

**问题:** reward 用 LIBERO success sparse reward，还是加 shaped reward？

**推荐答案:** 主训练 reward 先用 final success。shaped signals 作为 diagnostics 记录，不先进入主 reward。因为 harness 也要被修补，如果 shaped reward 设计不稳，模型和 harness 都会被带偏。

**待确认:** LIBERO 每个任务能不能稳定拿到 binary success，以及是否能拿 progress metric。

### Q13 - Harness 也要训练/修补是什么意思

**问题:** 你说 “修补 harness”，具体是让模型修改 harness 代码，还是训练一个 harness policy/router，还是自动调 prompt/API/reward/schema？

**推荐答案:** 先分成两阶段:

1. **Harness repair as code editing**: 离线根据失败 traces，生成 patch 修工具、prompt、错误处理、observation schema。
2. **Harness policy training**: 在线训练一个 harness/router，决定视觉描述策略、tool exposure、error recovery、budget 分配。

不要第一版就让 harness 在 RL 内直接自修改代码。先把 planner rollout trace 打完整，再做离线 harness repair。

**待确认:** 你说的 harness-R1 更接近哪一种: code repair、router policy、prompt/schema optimization，还是三者都有。

### Q14 - Harness action space

**问题:** 如果 harness 也要被训练，它的 action 是什么？

**推荐答案:** 第一版 harness action 不要是任意代码修改，而是结构化配置选择:

- 选择 VLM 描述 prompt。
- 选择是否给 raw image / diff description / video description。
- 选择暴露哪些 APIs。
- 选择错误恢复模板。
- 选择是否 retry 某个 tool。
- 选择 budget。

离线 repair 阶段再允许 code patch。

**待确认:** harness 训练是和 planner 同时在线训练，还是 planner 先训练，harness 后训练。

### Q15 - 失败恢复

**问题:** code 执行报错、tool 失败、VLA 抓取失败、SAM 分割失败时，episode 是否终止？

**推荐答案:** 大部分失败不终止，而是作为 observation 返回。只有越权、安全违规、timeout、超过 budget 才终止。可恢复错误应该消耗 step budget，并让模型下一轮修复。

**待确认:** 需要定义哪些错误码: `syntax_error`、`runtime_error`、`invalid_api`、`timeout`、`tool_failed`、`perception_empty`、`motion_failed`、`vla_failed`、`budget_exceeded`。

### Q16 - Trace schema

**问题:** 训练轨迹保存成什么格式？

**推荐答案:** 同时保存三层:

- raw conversation messages。
- normalized episode trace。
- token-level RL trajectory。

normalized episode trace 至少包含:

```text
episode_id
task_id
task_instruction
turns:
  - prompt_messages
    assistant_code
    parsed_calls
    stdout
    stderr
    visual_feedback_text
    structured_observation
    reward_after_turn
    done
final_reward
success
harness_config
artifact_refs
```

**待确认:** 图像/video artifact 是否保存在本地路径、对象存储，还是 parquet metadata 引用。

### Q17 - 是否蒸馏为 structured planner

**问题:** code planner 是最终形态，还是探索/训练中间态，后面蒸馏成 structured tool calls？

**推荐答案:** 第一版把 code planner 当最终可执行 action space 做通，但 trace schema 同时记录 raw code 和 parsed tool calls，保留蒸馏出口。

**待确认:** 未来是否需要 deployment 时禁用 Python code，只保留 structured tool calls。

## 暂定执行循环

第一版建议执行循环:

```text
task instruction
  -> harness builds initial observation
  -> optional VLM describes image/video state
  -> model generates Python code block
  -> harness parses code block
  -> restricted executor validates code
  -> code calls allowlisted harness APIs
  -> harness APIs call VLA / SAM / Molmo / motion / simulator backends
  -> executor captures stdout/stderr/errors
  -> harness computes reward/progress/success
  -> harness builds next observation as non-trainable context
  -> repeat until finish/success/failure/budget
  -> trajectory writer emits SFT/RL data
```

这里的 `finish()` 最好是一个显式 API:

```python
finish(reason="the cube appears lifted above the target height")
```

而不是自然语言 `FINISH`。这样 parser 更简单，SFT/RL mask 更清晰。

## 暂定 SFT 契约

SFT 样本保存多轮 `messages`。

训练目标:

- assistant 生成的 code block: label，loss mask 为 1。
- system prompt: context，loss mask 为 0。
- task instruction: context，loss mask 为 0。
- VLM visual description: context，loss mask 为 0。
- tool output/stdout/stderr/error: context，loss mask 为 0。
- harness observation summary: context，loss mask 为 0。

建议 assistant 输出格式:

```python
# optional short comment
obs = observe()
desc = describe_scene()
result = vla_pick("red cube")
print(result)
```

不建议训练长自然语言推理。推理如果需要，写成短注释即可。

## 暂定 RL 契约

RL rollout 的核心规则:

- 每个 model turn 产生一段 `response_ids`，对应 assistant code block。
- `response_mask=1` 只覆盖模型生成 token。
- observation、VLM feedback、stdout/stderr、tool result 进入下一轮 prompt，mask 为 0。
- episode final success 写入主 reward。
- 第一版 reward 可先写在最终 assistant turn 的最后有效 token。
- 后续可实验把 episode reward 分配到所有 assistant code turns。

需要重点解决:

- 多轮 response 如何在 verl/uni-agent Gateway 中对齐。
- 一个 episode 内多个 assistant turns 如何共享 reward。
- timeout/invalid code 是否给 0 reward，还是给轻微负分。
- harness repair 数据是否和 planner RL 数据共用一个 trace schema。

## 非目标

第一版不建议做以下事情:

- 不复用当前 `uni2-agent` 的 LIBERO planner tool surface 作为硬约束。
- 不让模型直接调用 simulator、`env.step`、裸 action 或后端私有方法。
- 不完全照搬 CaP-X 的 unrestricted Python exec。
- 不把自然语言 reasoning 作为动作解析的一部分。
- 不把视觉 observation 伪装成普通长文本；由 VLM/harness 生成 compact feedback。
- 不一开始追求一次性生成完整 episode 程序。
- 不在没有 SFT warm start 的情况下直接大规模 RL。
- 不第一版就让 harness 在线自修改代码。

## 需要收敛的确认决策

| 决策项 | 当前推荐 | 用户确认 |
| --- | --- | --- |
| 项目边界 | 独立 code planner + harness co-training 项目，不依赖当前 `uni2-agent` tools | 已修正目标，代码落点待确认 |
| Agent 输出契约 | CaP-X-style Python code block + harness-controlled API boundary | 待确认 |
| Multi-turn 语义 | 标准 step-wise assistant code turn，兼容 CaP-X `REGENERATE` 导入 | 待确认 |
| 执行粒度 | 每步或每小子目标生成一次 code block | 待确认 |
| 视觉输入 | planner 主要吃 VLM visual description，不直接依赖 raw image | 用户倾向已给出，待最终确认 |
| Tool/API 边界 | 只允许 harness allowlisted APIs，不暴露 direct `env` | 待确认 |
| Executor | restricted executor + timeout + import allowlist | 待确认 |
| Namespace | 有限持久 state，要求可序列化/可重放 | 待确认 |
| SFT 数据 | canonical traces + oracle/demo conversion + CaP-X successful artifacts | 待确认 |
| SFT label | 只训练 assistant code token | 用户倾向已给出，待最终确认 |
| RL reward | final success sparse reward，shaped signals 只记录 | 待确认 |
| RL reward 写入 | 先写最终 turn 末 token，后续实验多 turn 分配 | 待确认 |
| Harness repair | 先离线 code repair，再考虑在线 harness policy | 待确认 |
| Harness action | 第一版结构化配置选择，不直接在线自修改 | 待确认 |
| Trace schema | raw messages + normalized episode trace + token trajectory 三层 | 待确认 |

## 我现在还不明白什么

这些问题是下一轮必须问清楚的，不是可以靠读 repo 得到的事实。

1. **代码落点:** 你最终想在 `/home/eren/codes/uni2-agent` 里重构出这个项目，还是新建 repo / 新子包？
2. **planner 模型类型:** 第一版训练纯文本 code LLM，还是多模态 VLM-code model？
3. **视觉反馈:** planner 是否完全不看 raw image，只看 VLM 描述和工具结果？
4. **CaP-X 兼容:** 你想兼容 CaP-X 的 `REGENERATE` / `FINISH` traces，还是只借鉴 code block 思想？
5. **底层工具:** 第一版必须有 VLA、SAM3、Molmo2、motion primitive 全部，还是先用少数 primitive 跑通训练闭环？
6. **VLA 角色:** VLA 是一个 planner 可直接调用的 tool，还是封装在更高层 `pick(object)` / `place(object, target)` API 后面？
7. **executor 开放程度:** 为了快，是否允许原型期直接 `exec` + direct env；还是从第一天就 restricted executor？
8. **SFT 数据:** 你现在手上有什么 demos/oracle/success traces 可以转？LIBERO demos、CaP-X artifacts、还是需要先人工写？
9. **RL 框架:** 优先接 `uni-agent` 的 Gateway/verl 轨迹，还是先复用 CaP-X 的 GRPO 脚本？
10. **harness-R1 目标:** 你说参考 harness-R1 训练 harness，具体想训练的是 harness code repair、harness router/policy、prompt/schema optimizer，还是全部？
11. **harness 与 planner 是否同时训练:** 先 planner 后 harness，还是从一开始 alternating training？
12. **reward:** 第一版只用 LIBERO binary success，还是需要 progress/collision/tool success 进入 reward？
13. **trace 存储:** 轨迹最终要 Parquet、JSONL、还是 uni-agent 当前 trajectory object？
14. **部署形态:** 最终是否允许部署时执行 Python code，还是必须蒸馏成 structured tool calls？

## 下一步

建议下一步不是直接写代码，而是先回答上面的 14 个问题。回答后把本文更新成 “确认决策版”，然后再拆成实现模块:

- code planner agent loop。
- restricted code executor。
- harness API registry。
- visual feedback / VLM differencing module。
- SFT trace converter。
- RL rollout trajectory writer。
- reward / success evaluator。
- harness repair dataset builder。
