# uni2-agent LIBERO 迁移问答与决策记录

日期：2026-08-05

相关项目：

- `/home/eren/codes/uni2-agent`
- `/home/eren/codes/uni-agent`
- `/home/eren/codes/RPent`
- `/home/eren/codes/cap-x`

相关设计文档：

- `/home/eren/codes/uni2-agent/docs/source/concepts/libero-migration-design.md`

## 背景

目标是在 `uni2-agent` 上迁移 LIBERO 具身任务，并保持低耦合，方便后续扩展到多任务、多环境，以及后续接入 Harness Runtime / Harness-R1 风格的 runtime patch 机制。

初步推荐方案是：

```text
方案 C：普通 Sandbox 里启动 LIBERO Runtime Server
```

也就是：

```text
Uni-Agent Task
  + task-owned runtime server/client
  + 普通 Uni-Agent tools
  + 普通 ReAct 或 HarnessReAct agent
  + official success -> TaskResult
```

## 关键理解

LIBERO 本身主要是仿真环境和 benchmark，不是 Uni-Agent 里的 agent。

迁移时真正要接入的是：

- `Task`：suite、task id、seed、episode 生命周期、reward。
- `Runtime`：LIBERO 仿真器启动、reset、step、render、close。
- `Tools`：模型可见的具身动作接口。
- `Observation`：每一步动作后返回给模型的文本、图像或状态反馈。
- `Dataset/config`：suite/task/seed 样本行和 prompt 构造方式。
- `Reward`：用 official success / `libero_terminated` 形成 `TaskResult.reward`。

## 推荐默认选择

如果没有额外约束，建议第一版按下面的默认选择推进：

```text
1. 先做 mock runtime，再接真 LIBERO。
2. 真 LIBERO 先 local 调通，再考虑 docker 固化。
3. 第一批任务用 libero_spatial task 0 seed 0。
4. 第一版 observation 用 privileged text。
5. 第一版 action surface 用 observe + run_skill + submit。
6. 预留 embodied 通用层，方便后续扩展到 RoboCasa / BEHAVIOR / Robosuite 等。
7. cap-x 主要参考 env 初始化、suite/task、task_language、official success。
8. RPent 主要参考 runtime server/client、robot tools、prompt/memory/skill 经验。
```

## 我需要用户拍板的问题

### 1. 第一版允许先做 mock runtime 吗？

推荐答复：

```text
允许。先用 mock runtime 跑通 Uni-Agent Task/Tool/Agent/Reward 闭环，再接真 LIBERO。
```

理由：

- 可以先验证低耦合架构是否跑通。
- 不会一开始被 MuJoCo、robosuite、LIBERO assets、显示环境、版本冲突卡住。
- 后续把 mock server 替换成真 LIBERO server 即可。

### 2. 真 LIBERO 运行环境优先 local 还是 docker？

推荐答复：

```text
local 先调通，docker 后固化。
```

理由：

- local 更容易调试。
- cap-x 文档说明 LIBERO 需要独立 venv，因为 robosuite 版本和普通 robosuite 可能冲突。
- docker 更适合复现和隔离，但第一阶段构镜像会增加复杂度。

### 3. 第一批任务是否用 `libero_spatial task 0 seed 0`？

推荐答复：

```text
是。
```

理由：

- 单任务 smoke path 最容易定位问题。
- `libero_spatial` 比 `libero_10` 更适合先跑通。
- 之后再扩展到 `libero_object`、`libero_goal`、`libero_10` 和 LIBERO-PRO perturbation suites。

### 4. 第一版 observation 是否用 privileged text？

推荐答复：

```text
是。先用 privileged text observation。
```

理由：

- 最容易验证 Task/Tool/Runtime/Reward。
- 不需要一开始接图像、VLM、SAM。
- 后续可以逐步降级到视觉 observation 或 hybrid observation。

第一版 observation 可以类似：

```text
Observation:
Task: put the mug on the plate
Step: 4
Success: false
Robot: eef_pos=[...], gripper=open
Objects:
- mug: pos=[...]
- plate: pos=[...]
Last action: pick(mug)
Result: grasp failed, object not in gripper
```

### 5. 第一版 action 是否用 `observe + run_skill + submit`？

推荐答复：

```text
是。第一版用少量高层 skill，不直接暴露很多低层控制。
```

推荐第一批工具：

```text
libero_observe
libero_run_skill
libero_submit
```

`libero_run_skill` 可支持：

```text
move_to
pick
place
open
close
wait
```

理由：

- 工具数量少，小模型更容易稳定调用。
- Runtime 可以校验 skill name 和 args。
- 每个环境可以把同一 skill 映射到自己的 controller。
- 后续 Harness Patch 可以在 skill 层做 guard/recovery，不需要理解低层控制器。

### 6. 是否为后续多任务预留 embodied 通用层？

推荐答复：

```text
是。
```

建议目录：

```text
uni_agent/tasks/embodied/
  base.py
  runtime_client.py
  types.py

uni_agent/tasks/libero/
  task.py
  client.py
  server.py
  preprocess.py

uni_agent/tools/embodied/
  base.py

uni_agent/tools/libero.py
```

判断低耦合是否健康的标准：

```text
新增第二个 embodied task family 时，不需要改 ReActAgent、Toolbox 或 base Sandbox。
```

## RPent 与 cap-x 的参考分工

### cap-x 主要参考

重点文件：

```text
/home/eren/codes/cap-x/docs/libero-tasks.md
/home/eren/codes/cap-x/env_configs/libero/*.yaml
/home/eren/codes/cap-x/capx/envs/simulators/libero.py
/home/eren/codes/cap-x/capx/envs/adapters/libero_wrapper.py
/home/eren/codes/cap-x/capx/envs/scripts/run_libero_batch.py
```

参考内容：

- LIBERO env 怎么初始化。
- suite/task/seed 怎么配置。
- task language 怎么获取。
- official success 怎么判断。
- LIBERO-PRO assets/config 怎么组织。
- batch task 怎么跑。

### RPent 主要参考

重点文件：

```text
/home/eren/codes/RPent/robots/libero/env_server.py
/home/eren/codes/RPent/robots/libero/env_client.py
/home/eren/codes/RPent/robots/libero/tools.py
/home/eren/codes/RPent/robots/libero/prompts/system.py
/home/eren/codes/RPent/robots/libero/prompt_bundle.py
```

参考内容：

- runtime server/client 进程架构。
- robot primitive tools 如何抽象。
- VLA server / SAM server 如何拆。
- prompt、memory、skill、recovery 经验。

注意：

```text
RPent prompt 不建议整坨搬。
应该把其中稳定经验拆成 static skills、guard rules、recovery hints 或 Harness Patch。
```

## 默认实现路线

### Phase 1：Mock Smoke Path

新增：

```text
uni_agent/tasks/embodied/
uni_agent/tasks/libero/
uni_agent/tools/libero.py
examples/quickstart/inference/task_config_libero_local.yaml
```

先用 mock runtime server 支持：

```text
GET /health
POST /reset
GET /observe
POST /run_skill
GET /success
POST /close
```

目标：

```text
启动一个 mock LIBERO task，模型能 observe，能 submit，TaskResult 能返回 reward。
```

### Phase 2：真 LIBERO Local Runtime

参考 cap-x 接入真实 LIBERO：

```text
suite_name
task_id
seed
task_language
official success / libero_terminated
```

目标：

```text
不依赖模型，手写 sequence 能通过 Uni-Agent tools 控制一个真 LIBERO task。
```

### Phase 3：Agent Inference

构造 tiny dataset：

```text
suite_name
task_id
seed
task_language
prompt
```

复用：

```text
examples/inference/parallel_infer_api.py
```

目标：

```text
模型可以 observe、run_skill、submit。
```

### Phase 4：Harness Runtime

后续切到 `harness_react`：

```text
on_init
make_pre_hint
on_before_action
on_post_step
```

具身场景可先实现：

```text
阻止还没 observe 就 submit
阻止重复失败 skill
检测 grasp failed
注入 recovery hint
记录 visited objects / attempted skills / last_success
```

### Phase 5：多任务泛化

接第二个 embodied task family，例如：

```text
RoboCasa
BEHAVIOR
Robosuite
```

如果新增第二个任务时不需要修改：

```text
ReActAgent
Toolbox
base Sandbox
```

说明抽象边界是健康的。

## 后续对话记录

后续关于 LIBERO 迁移方案、用户选择、实现决策和问题定位，都继续追加到这里。

### Q：primitive 那套也就是 tool，现在也接入吗？

用户问题：

```text
primitve 那套也就是tool 你准备现在也接入吗
```

建议答复：

```text
现在先接 tool interface 和 mock primitive，不立刻整套接入真实 RPent primitives。
```

原因：

- 当前第一目标是验证 Uni-Agent 低耦合具身链路：
  `LiberoTask -> runtime server/client -> Uni-Agent tools -> ReActAgent -> TaskResult`。
- 如果一开始直接接完整 RPent primitive/VLA/SAM，会同时引入 LIBERO、MuJoCo、VLA、
  perception、controller、tool-calling 多类不确定性，调试成本太高。
- 先定义稳定 tool surface，后面把 mock backend 换成 RPent/cap-x backend 即可。

第一版应该接入：

```text
libero_observe
libero_run_skill
libero_submit
```

其中 `libero_run_skill` 先支持少量高层 skill name：

```text
move_to
pick
place
open
close
wait
```

Phase 1：

```text
tool schema + mock runtime
```

Phase 2：

```text
同一套 tool schema 接真实 LIBERO runtime
```

Phase 3：

```text
把 RPent 的真实 primitives / VLA / SAM 能力挂到 runtime backend 后面
```

关键原则：

```text
模型看到的是稳定 tools。
tools 调的是 EmbodiedRuntimeClient。
RuntimeClient 背后可以是 mock、cap-x、RPent 或真 LIBERO。
不要让 ReActAgent 或 Task 直接依赖 RPent primitive 实现。
```

### 决策：先实现 Mock LIBERO，采用方案 C

用户拍板：

```text
需要。先跑一个 mock libero。严格模仿 uni2-agent 风格，写在 uni2-agent 项目中。
采用方案 C，按照推荐默认选择。
```

实现范围：

```text
1. 新增 embodied 通用 runtime client/context 小抽象。
2. 新增 libero task family。
3. 新增 mock LIBERO runtime server/client。
4. 新增普通 Uni-Agent tools：
   - libero_observe
   - libero_run_skill
   - libero_submit
5. 新增 example task config。
6. 新增 tiny mock dataset builder，方便 parallel_infer_api.py 跑通。
7. 不接真实 LIBERO/MuJoCo/RPent primitives。
8. 不修改 ReActAgent 的 LIBERO 专用逻辑。
```

成功标准：

```text
parallel_infer_api.py
  -> TaskConfigResolver
  -> LiberoTask.run()
  -> start mock runtime server
  -> ReActAgent.run()
  -> tools call mock runtime
  -> libero_submit
  -> TaskResult(reward/extra_info)
```
