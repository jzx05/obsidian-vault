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

### 实现状态：Mock LIBERO 方案 C 已落到 uni2-agent

本轮已经在 `/home/eren/codes/uni2-agent` 中实现第一版 mock LIBERO 闭环。

新增/修改的核心内容：

```text
uni_agent/tasks/embodied/
  __init__.py
  runtime_client.py

uni_agent/tasks/libero/
  __init__.py
  client.py
  server.py
  task.py

uni_agent/tools/libero.py
uni_agent/tasks/registry.py
uni_agent/tools/__init__.py
uni_agent/agents/react/agent.py

examples/quickstart/inference/task_config_libero_mock.yaml
examples/quickstart/inference/make_libero_mock_dataset.py
examples/quickstart/inference/mock_libero_policy_server.py
examples/quickstart/inference/run_infer_libero_mock.sh
tests/uni_agent/tasks/test_libero_mock_task.py
```

实际架构：

```text
parallel_infer_api.py
  -> TaskConfigResolver
  -> get_task({"name": "libero", ...})
  -> LiberoTask.run()
  -> 普通 local sandbox
  -> task-owned mock_libero_runtime_server
  -> ReActAgent.run()
  -> Toolbox
  -> libero_observe / libero_run_skill / submit
  -> LiberoRuntimeClient HTTP 调 runtime
  -> TaskResult(reward=1.0 when success)
```

关键实现决策：

```text
1. Runtime server/client 放在 uni_agent/tasks/libero 内部。
2. 工具是普通 Uni-Agent tools，不改 base Sandbox。
3. ReActAgent 不加入 LIBERO 专用逻辑。
4. libero_submit 的 registry key 是 libero_submit，但模型看到的函数名是 submit。
5. ReAct 的 finish-tool 检测已改成按注册工具解析 model-facing name，因此 libero_submit 也会被识别为结束工具。
6. 第一版 runtime_backend 只支持 mock。
```

验证结果：

```text
py_compile: 通过
mock runtime + Toolbox 直接调用: 通过
dataset builder: 通过，生成 /tmp/libero_mock.parquet
TaskConfigResolver -> LiberoTask -> ReActConfig runtime 注入: 通过
一键 smoke 脚本: 通过
```

一键 smoke 命令：

```bash
cd /home/eren/codes/uni2-agent
examples/quickstart/inference/run_infer_libero_mock.sh
```

成功输出摘要：

```text
resolved       1   (100.0%)
wrong-ans      0
timeout/err    0
total          1
```

注意事项：

```text
当前 /home/eren/miniconda3/envs/uni-agent 环境没有 pytest，所以 pytest 单测没有正式跑。
已用 py_compile 和端到端 smoke 代替验证。
脚本默认优先使用 /home/eren/miniconda3/envs/uni-agent/bin/python。
直接用 python examples/inference/parallel_infer_api.py 时建议设置 PYTHONPATH=/home/eren/codes/uni2-agent，
否则可能导入不到当前工作区源码。
```

下一阶段接真 LIBERO 时优先替换的位置：

```text
uni_agent/tasks/libero/server.py
```

也就是把 mock server 后面的状态机换成真实 LIBERO env：

```text
reset -> env.reset()
observe -> env/render/state/task_language
run_skill -> primitive/controller/VLA/SAM/backend
success -> official success / libero_terminated
close -> env.close()
```

保持不变的部分：

```text
LiberoTask
LiberoRuntimeClient protocol
libero_observe
libero_run_skill
libero_submit/submit
ReActAgent
Toolbox
base Sandbox
```

### Q：Runtime Server 协议为什么要改成正式 v0 契约？

用户问题：

```text
Runtime Server 的协议要不要现在就冻结成第一版契约？
这个我没懂，讲解下，改成正式的。
```

解释：

```text
所谓正式协议，就是 mock LIBERO server 和未来真实 LIBERO server 都讲同一种 JSON 语言。
tools 和 task 只读稳定字段，不猜 backend 私有字段。
```

如果不固定协议，后面很容易出现：

```text
mock 返回 success/libero_terminated/text
真 LIBERO 返回 is_success/done/obs
RPent backend 返回 terminal/finished/description
```

这样会导致：

```text
1. tools 要写很多 if else 猜字段。
2. Task reward 容易读错。
3. trajectory schema 不稳定，后面 RL 和 Harness patch 很难消费。
4. held-out regression 很难比较不同 runtime backend。
```

正式 v0 协议已经落到：

```text
uni_agent/tasks/libero/protocol.py
```

除 `/health` 和 `/close` 外，episode endpoint 都返回：

```json
{
  "ok": true,
  "message": "placed mug on plate",
  "observation": "Task: ...",
  "success": true,
  "terminated": true,
  "step_count": 3,
  "task_language": "Place the mug on the plate.",
  "info": {},
  "protocol_version": "libero-runtime-v0"
}
```

字段语义：

```text
ok              这次 runtime 请求是否被接受并执行
message         短原因，给模型和日志读
observation     模型可见 observation 文本
success         任务是否成功，reward 只读它
terminated      环境是否结束，结束后不再允许 run_skill 改状态
step_count      环境动作步数
task_language   当前任务语言
info            backend 私有诊断信息
protocol_version 协议版本
```

三个结束概念必须分开：

```text
success     -> reward 是否为 1
terminated  -> runtime 是否还允许改变环境
submit      -> ReAct interaction 是否结束
```

因此：

```text
submit 不决定 reward。
runtime /success 决定 reward。
success=True 后 runtime 设置 terminated=True，后续 run_skill 返回 ok=false。
模型仍可 observe，然后 submit 结束 ReAct。
```

### 实现状态：按 grill 推荐补齐 mock 回归与 runtime contract

本轮继续按推荐路线补齐：

```text
1. mock backend 增加 seen + held-out 场景。
2. mock success 条件改成严格匹配 goal_object 和 goal_target。
3. success=True 后 terminated=True，后续 run_skill 返回 ok=false。
4. health endpoint 增加 protocol_version。
5. LiberoTask.extra_info 增加 observation_mode、runtime_health、runtime_protocol_version、runtime_info。
6. mock dataset builder 支持 --include-heldout。
7. mock policy server 从 prompt 解析 Place X on/in Y，支持 held-out smoke。
8. run_infer_libero_mock.sh 支持 INCLUDE_HELDOUT=1，并检测 policy server 启动失败。
```

mock 回归场景：

```text
seen:
  task_id=0  Place the mug on the plate.

heldout:
  task_id=1  Place the bowl on the tray.
  task_id=2  Place the cube in the basket.
```

验证命令：

```bash
cd /home/eren/codes/uni2-agent
examples/quickstart/inference/run_infer_libero_mock.sh
INCLUDE_HELDOUT=1 examples/quickstart/inference/run_infer_libero_mock.sh
```

验证结果：

```text
默认 seen smoke: resolved=1/1
held-out smoke: resolved=3/3
```

仍未完成：

```text
当前 conda 环境没有 pytest / ruff，因此正式 pytest 和 ruff 没跑。
本轮使用 py_compile、runtime smoke、dataset smoke、parallel_infer_api smoke 代替。
```

### Review 后修复

code review 后补齐：

```text
1. LIBERO tools 从 uni_agent.tools.__init__ eager import 改为 get_tool 按 registry key lazy-load。
2. ReAct 结束工具检测不再读取 TOOL_REGISTRY，而是使用 Toolbox.names() 的模型可见函数名。
3. XML/Qwen reasoning-only tool-call fallback 补了边界测试。
4. LiberoTask.run() 在 finally 中调用 runtime_client.close()，mock server 记录 closed=True。
5. LIBERO 设计文档加入 mkdocs.yml nav。
6. mock runtime in-process 明确标为 Phase 0，只用于协议和工具闭环；真 LIBERO 阶段要换成 task-owned local process/docker/remote service。
```

### 实现状态：新增 local LIBERO lifecycle backend

本轮按“方案 C：普通 Sandbox + task-owned Runtime Server”继续推进，但命名和配置保持简单：

```text
runtime_backend: mock | local
```

新增文件：

```text
uni_agent/tasks/libero/local_server.py
examples/quickstart/libero/probe_libero.py
examples/quickstart/inference/task_config_libero_local.yaml
tests/uni_agent/tasks/test_libero_local_server.py
```

代码边界：

```text
LiberoTask 只负责选择 backend。
ReActAgent 不加 LIBERO 特判。
Sandbox 不加 LIBERO 方法。
真实 LIBERO import、OffScreenRenderEnv、bddl/init state/success 检查都收在 local_server.py。
```

当前 local backend 只实现真实 LIBERO 生命周期：

```text
/health
/reset
/observe
/success
/close
```

`/run_skill` 在 local backend 中暂时返回：

```text
ok=false
message="local run_skill is not implemented yet"
```

这是刻意保守的第一步：先确认真实 LIBERO 能 reset、observe、读 official success、close，
再接 RPent/CaP-X primitive 或 skill controller。这样不会把环境配置、primitive、VLA、
SAM/GraspNet 和 agent loop 混在一起调。

验证结果：

```text
py_compile: passed
fake LIBERO lifecycle smoke: passed
real local probe in 当前 uni-agent env: 返回标准 protocol payload，提示 LIBERO 未安装
mock inference smoke: resolved=1/1
```

当前阻塞：

```text
conda env /home/eren/miniconda3/envs/uni-agent 里没有 pytest，因此 pytest 未跑。
当前 env 也不能 import libero，因此真实 LIBERO reset 还没跑通。
```

下一步：

```text
1. 单独配 LIBERO runtime env，让 probe_libero.py 能 reset 成功。
2. 从 RPent/CaP-X 迁移第一版 primitive/skill controller。
3. local backend 的 /run_skill 从 placeholder 改为真实 step。
4. 再把 task_config_libero_local.yaml 的 tools 加回 libero_run_skill。
```

### 实现状态：最小 LIBERO 启动脚本

新增：

```text
examples/quickstart/libero/setup_libero_local.sh
examples/quickstart/libero/run_libero_local.sh
```

作用：

```text
1. 在 uni2-agent 项目内使用 .venv-libero 和 third_party/LIBERO-PRO。
2. 自动设置 MUJOCO_GL=egl 和 PYOPENGL_PLATFORM=egl。
3. 检查本项目 third_party/LIBERO-PRO 的 LIBERO assets 路径。
4. 如果 ~/.libero/config.yaml 不存在，则自动写入。
5. 调用 examples/quickstart/libero/probe_libero.py 启动 local runtime probe。
```

运行：

```bash
cd /home/eren/codes/uni2-agent
examples/quickstart/libero/setup_libero_local.sh
examples/quickstart/libero/run_libero_local.sh
```

如果 GitHub clone 失败，但机器上已有 LIBERO-PRO，可以显式复制到本项目：

```bash
LIBERO_SOURCE=/home/eren/codes/cap-x/capx/third_party/LIBERO-PRO examples/quickstart/libero/setup_libero_local.sh
```

切换任务：

```bash
SUITE=libero_goal TASK_ID=2 SEED=0 examples/quickstart/libero/run_libero_local.sh
```

当前验证：

```text
脚本能检查到 /home/eren/codes/uni2-agent/.venv-libero/bin/python 尚不存在，并给出创建命令。
已经支持 LIBERO_SOURCE=/path/to/LIBERO-PRO，把已有 LIBERO-PRO 复制到 uni2-agent/third_party/LIBERO-PRO。
当前项目内 third_party/LIBERO-PRO 已经从本机已有 checkout 准备好。
临时指定 LIBERO_PYTHON=/home/eren/miniconda3/envs/uni-agent/bin/python 时，脚本能写入 ~/.libero/config.yaml 并进入 probe；
probe 返回标准 libero-runtime-v0 payload，提示当前 Python 不能 import libero。
```

当前阻塞：

```text
系统里没有 uv。
python3.12 存在，但缺 ensurepip / python3.12-venv，因此 python3.12 -m venv 创建 .venv-libero 失败。
```

下一步仍是创建 uni2-agent 项目内的 LIBERO venv。两种方式任选一种：

```bash
python3 -m pip install --user uv
cd /home/eren/codes/uni2-agent
examples/quickstart/libero/setup_libero_local.sh
```

或者：

```bash
sudo apt install python3.12-venv
cd /home/eren/codes/uni2-agent
examples/quickstart/libero/setup_libero_local.sh
```
