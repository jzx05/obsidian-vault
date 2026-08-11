# LIBERO Code Planner Training Design

这份文档记录在 `uni2-agent` 中接入 embodied code planner 的训练设计需求。

当前状态: **Round 1 设计问题草案**。本文先保存需要确认的问题、推荐答案和初始约束；等问题回答后，再把确认项收敛成正式的 SFT / RL 训练契约。

## 背景
首先说明，目标完全不是在uni2-agent中接入code planner，而是用 code planner作为agent，结合uni-agent项目的整体harnesss 设计一个既可以训练code planner，又可以修补harness的协同训练项目。模型根据根据任务、图像观察、结构化状态和工具执行结果，自由组合底层工具完成 LIBERO 等具身任务。目前的uni2-agent里面的可以不需要管！目前uni2-agent暴露的工具函数可以不用管，当作没有！

## 初始设计倾向

推荐方向是 **step-wise restricted code planner**:

1. 模型每轮生成依据cap-x中的 code block。
2. harness 在环境里执行该 block。
3. 执行后把vlm对图片的描述、工具结果、stdout/stderr 和错误信息返回给模型。
4. 模型继续生成下一步代码，直到调用 `finish()`、成功、失败或耗尽预算。
5. SFT 和 RL 都只训练模型生成的代码 token；环境观察和工具输出只作为上下文。

这条路线保留 code-as-policy 的表达力，同时比一次性生成完整 episode 程序更适合图像闭环、失败恢复和 RL credit assignment。

## Round 1 设计问题

以下问题需要用户确认。每个问题下面给出当前推荐答案，后续回答会写入本文的“确认决策”部分。

### Q1 - 模型输出契约

**问题:** 训练的 code planner 最终输出应该是什么形态？

**推荐答案:** 使用受限的 step-wise Python action block。模型每一轮输出一个小 Python 代码块，只允许调用 harness 暴露的工具函数，例如 `observe()`、`vla_pick()`、`segment()`、`goto_pose()`、`finish()`。不要一开始就让它输出完全自由 Python，也不要直接输出 JSON DSL。自由 Python 太难控，JSON DSL 又损失 code-as-policy 的表达力。

**待确认:** 用户是否接受 restricted Python 作为第一版 action space。

### Q2 - 执行粒度

**问题:** 模型一次生成完整 episode 程序，还是每观察一步生成一步代码？

**推荐答案:** 一步或一个小子目标生成一次代码。每个 block 最多 1-3 个工具调用，执行后把图像、工具结果、stdout/stderr、状态摘要返回给模型，再让它继续生成下一段。不要把 CaP-X 式“一次性生成完整代码再执行到底”作为主路线，因为具身任务里图像状态变化大，失败恢复和 RL credit assignment 都会更难。

**待确认:** code block 是按 primitive step、subgoal，还是二者混合。

### Q3 - 工具边界

**问题:** code planner 能调用哪些东西？只能调用 VLA/SAM/Molmo/motion primitive，还是可以直接碰 simulator/runtime/env？

**推荐答案:** 只允许调用 `uni2-agent` 暴露出来的 planner-visible tool wrapper。禁止直接 `env.step`、禁止直接改 simulator state、禁止裸发 7D action、禁止任意 import、禁止读写文件和网络。底层可以有 VLA、SAM3、Molmo2、grasp、motion，但模型只通过稳定 API 组合它们。

**待确认:** 是否需要暴露任何额外的 code-only helper，例如 geometry helper、mask filtering helper、retry helper。

### Q4 - 观察输入

**问题:** 每一步执行后返回给模型的 observation 包含什么？只有图片？还是图片引用、结构化状态、工具结果、stdout/stderr 都包含？

**推荐答案:** 全部包含，但分层序列化。给模型的是: 图片 artifact/ref、相机名、EE pose、gripper 状态、joint/state 摘要、最近工具调用结果、stdout/stderr、episode step 信息。图片不要硬塞成文本；训练轨迹里保存 image refs/artifacts，文本上下文里放 compact summary。

**待确认:** 多模态模型是否直接吃 image artifact；如果训练框架暂时只支持文本，是否先用视觉工具输出和 compact caption 作为替代。

### Q5 - 训练路线

**问题:** 先 SFT 再 RL，还是直接 RL？

**推荐答案:** 先 SFT，再 RL。先用 scripted policy、人工写的成功代码、CaP-X 风格 successful traces、已有 planner traces 生成 SFT 数据，让模型学会基本工具语法和常见任务分解；然后再用 GRPO/RLOO 之类的 RL 做成功率优化。直接 RL 会把大量 budget 浪费在语法错误、非法调用、无效探索上。

**待确认:** 第一阶段 SFT 数据能否从 scripted oracle、现有 demos 或手写 traces 启动。

### Q6 - 奖励设计

**问题:** RL reward 用最终 sparse success，还是要加 shaped reward？

**推荐答案:** 主 reward 用最终 success，辅助记录 shaped signals。也就是 `TaskResult.reward = success ? 1.0 : 0.0` 作为训练主信号，同时在 `extra_info` 里记录 progress、tool success、collision、grasp success、distance-to-goal、invalid-code 等指标。先不要让 shaped reward 进入主优化，除非 sparse reward 完全跑不动。

**待确认:** 哪些 shaped signals 在 LIBERO runtime 中已经能稳定拿到，哪些只是调试指标。

### Q7 - 轨迹 mask 规则

**问题:** SFT/RL 训练时，哪些 token 是 loss/reward 目标？

**推荐答案:** 只训练模型生成的 code/planner token。环境观察、图片描述、tool output、stdout/stderr、system prompt、user task 都是 context，loss mask / response mask 为 0；模型输出的代码块、`finish()`、必要的 planner text 才 mask 为 1。RL reward 最好稀疏写到每轮或最终 assistant/code block 的末尾 token。

**待确认:** 是否允许多个 assistant/code block 共享一个 episode-level reward，或只在最终 block 上写 reward。

### Q8 - 是否保留自然语言 reasoning

**问题:** 模型输出里是否允许夹带自然语言思考，例如“我先识别红色方块，然后抓取”？

**推荐答案:** 训练时可以保留短 plan comment，但执行契约只认代码块。例如允许代码里有少量注释，但不要让自然语言成为必须解析的动作。最终 harness 只执行 fenced code block 或 action block，其他文本最多作为日志，不参与工具调用。

**待确认:** 数据集中是否保留注释、短计划，还是训练纯代码输出。

### Q9 - 代码执行状态是否持久

**问题:** 每一步 code block 执行时，Python namespace 要不要跨步保留变量？

**推荐答案:** 短期保留有限 session state，但强制可序列化和可重放。例如允许 `target_mask_id`、`last_pose`、`candidate_points` 这类变量跨步存在，但每步执行后 harness 要把关键状态写进 trace，保证 SFT/RL replay 时能复现。不要让模型依赖不可见的复杂 Python object 生命周期。

**待确认:** 是否需要跨步变量；如果需要，哪些类型允许进入 session state。

### Q10 - 集成方式

**问题:** 在 `uni2-agent` 里改现有 LIBERO planner，还是新增一个 code planner agent/harness？

**推荐答案:** 新增 `code_planner` agent/harness 路线。现有结构化 planner 和 VLA harness 保留，code planner 作为新的 agent type 接入 `LiberoTask`、Gateway 和 trajectory capture。这样不会污染原来的 tool-call planner，也方便对比 structured planner vs code planner。

**待确认:** 是否接受新 agent type；是否需要与现有 ReAct/Libero agent 共用 prompt、toolbox 和 runtime。

### Q11 - SFT 数据来源

**问题:** 第一批 SFT 数据准备从哪里来？

**推荐答案:** 三类混合: 第一，手写少量 canonical code traces 覆盖每个 primitive；第二，用 scripted oracle / benchmark demos 自动转成 step-wise code traces；第三，从 RL 或 rollout 中筛 successful traces 做 rejection sampling。不要一开始依赖人手写大量长 episode 代码。

**待确认:** 优先使用哪个数据源启动，以及是否要把 CaP-X traces 转成 `uni2-agent` action block 格式。

### Q12 - 失败恢复策略

**问题:** 当 code 执行报错、tool 调用失败、VLA 没抓到、SAM 没分出 mask 时，模型应该怎么收到反馈并恢复？

**推荐答案:** 失败作为 observation 返回，不直接终止 episode。语法错、非法 API、timeout 可以消耗 step budget 并返回 structured error；感知/抓取失败返回 tool result，让模型下一步可以重新观察、换 prompt、换点、调用 VLA 或 motion primitive。只有严重越权、安全违规、超过 budget 才终止。

**待确认:** 哪些错误可恢复，哪些错误必须立即终止并给负向 reward 或 zero reward。

### Q13 - RL rollout 的 episode budget

**问题:** 每个任务允许多少轮 code generation、多少 tool call、多少真实/sim env step？

**推荐答案:** 先设一个保守预算，例如最多 8-12 个 code blocks，每个 block 最多 3 个 tool calls，总 env step 上限跟 LIBERO 原任务一致或略低。预算必须写死进 harness，否则 RL 会学会拖延、刷观察、或者无限修代码。

**待确认:** 第一版 budget 数值，以及 budget 超限时的 `finished`、`reward`、`extra_info` 记录方式。

### Q14 - 最终长期形态

**问题:** 最终模型永远输出 code，还是 code 只是训练/探索中间态，后面蒸馏成 structured planner/tool calls？

**推荐答案:** 先把 code planner 当最终可执行 action space 做通，但保留蒸馏出口。也就是 trace schema 里同时记录 raw code、解析后的 tool calls、工具结果。这样以后可以把 successful code traces 蒸馏成更稳定的 structured planner，也可以继续走 code-as-policy。

**待确认:** trace schema 是否必须从第一版就支持 raw code 与 parsed tool calls 双记录。

### Q15 - 文档落点

**问题:** 这个设计需求文档要放在哪里？

**推荐答案:** 放在 `uni2-agent/docs/source/concepts/libero-code-planner-training-design.md`。内容结构包括: 目标、非目标、输出契约、执行循环、tool API 边界、observation schema、SFT 数据格式、RL 轨迹格式、reward/mask 规则、failure handling、open questions。

**待确认:** 当前已按该路径创建。后续如果要变成正式 RFC，可以移到 `docs/source/designs/` 或 `docs/source/rfcs/`。

## 需要收敛的确认决策

这一节在用户回答后更新。目前先保留为空。

| 决策项 | 当前推荐 | 用户确认 |
| --- | --- | --- |
| 输出契约 | restricted step-wise Python action block | 待确认 |
| 执行粒度 | 每步或每小子目标生成一次代码 | 待确认 |
| 工具边界 | 只允许 planner-visible wrappers | 待确认 |
| observation schema | image refs + compact structured state + tool/stdout/stderr | 待确认 |
| 训练路线 | SFT first, then RL | 待确认 |
| 主 reward | sparse final success | 待确认 |
| mask 规则 | 只训练模型生成 token | 待确认 |
| reasoning 输出 | 可有短注释，执行只认代码 | 待确认 |
| namespace | 有限持久、可序列化、可重放 | 待确认 |
| 集成方式 | 新增 `code_planner` agent/harness | 待确认 |
| SFT 数据 | 手写 canonical + scripted/demo conversion + successful rollouts | 待确认 |
| 失败恢复 | 可恢复错误作为 observation 返回 | 待确认 |
| rollout budget | 8-12 code blocks 起步，每 block 最多 3 tool calls | 待确认 |
| 长期形态 | code planner 做通，同时保留 structured distillation 出口 | 待确认 |

## 暂定训练契约

### SFT

SFT 样本应保存多轮 `messages`。其中:

- system/user/task prompt 是上下文。
- observation/tool output/stdout/stderr 是上下文。
- assistant 输出的 code block 是 label。
- loss mask 只覆盖 assistant 生成的 code/planner token。
- 图像以 artifact/ref 方式记录；文本上下文只写 compact description 或 tool result。

### RL

RL rollout 应通过 Gateway 捕获 token-level trajectory。原则上:

- `prompt_ids` 包含任务、历史 observation、工具输出和历史模型输出上下文。
- `response_ids` 对应本轮模型生成的代码 token。
- `response_mask=1` 只覆盖模型生成 token。
- observation、tool result、stdout/stderr、图像引用进入下一轮 prompt，mask 为 0。
- `TaskResult.reward` 或 `reward_info` 写主 reward。
- `rm_scores` 可稀疏写在最终 code block 的末尾 token，或按 episode-level reward 分配到被训练的 response token，具体规则待确认。

## 暂定执行循环

```text
task instruction
  -> model generates restricted code block
  -> harness parses and validates code
  -> restricted executor runs allowed tool calls
  -> runtime executes VLA / SAM / Molmo / grasp / motion backends
  -> harness collects observation, stdout/stderr, tool results, errors
  -> append observation as non-trainable context
  -> continue until finish/success/failure/budget
  -> task computes reward and records trajectory
```

## 非目标

第一版不建议做以下事情:

- 不做完全自由 Python 代码执行。
- 不让模型直接调用 simulator、`env.step`、裸 action 或后端私有方法。
- 不把自然语言 reasoning 作为动作解析的一部分。
- 不把视觉 observation 伪装成普通长文本；优先保留 artifact/ref。
- 不一开始追求一次性生成完整 episode 程序。
- 不在没有 SFT warm start 的情况下直接大规模 RL。

## 后续 Round 2 可能问题

Round 1 回答后，还需要继续收敛:

- action block 的精确语法和 fenced code 解析规则。
- allowlisted helper API 的完整列表。
- session namespace 的持久化和清理规则。
- SFT trace JSON/Parquet schema。
- RL reward 写入 `rm_scores` 的具体分配策略。
- 多图像、多相机 artifact 的序列化方式。
- timeout、非法调用、语法错误、工具失败的错误码闭集。
- 从 CaP-X traces 到 `uni2-agent` traces 的转换规则。
- evaluation split、baseline 和 ablation 设计。
- 是否需要从 successful code traces 蒸馏出 structured planner traces。
