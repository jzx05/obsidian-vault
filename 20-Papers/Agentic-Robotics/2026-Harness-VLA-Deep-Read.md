---
title: "Harness VLA 深度解读"
authors:
  - Yixian Zhang
  - Huanming Zhang
  - Feng Gao
  - Xiao Li
  - Zhihao Liu
  - Chunyang Zhu
  - Jiaxing Qiu
  - Yuchen Yan
  - Jiyuan Liu
  - Wenhao Tang
  - Zhengru Fang
  - Yi Nie
  - Changxu Wei
  - Yu Wang
  - Wenbo Ding
  - Chao Yu
year: 2026
venue: arXiv preprint
arxiv: 2607.08448v3
tags:
  - paper-deep-read
  - VLA
  - agentic-robotics
  - hierarchical-policy
  - memory
  - robot-manipulation
status: read
rating: 8/10
date-read: 2026-07-30
url: https://arxiv.org/abs/2607.08448
pdf: 'C:\Users\huawei\Zotero\storage\ESDD9J8D\Zhang 等 - 2026 - Harness VLA Steering Frozen VLAs into Reliable Manipulation Primitives via Memory-Guided Agents.pdf'
related:
  - "[[2026-Hi-VLA-Deep-Read]]"
  - "[[2026-Hi-VLA-Orchestration]]"
  - "[[2026-Agentic-VLA-Deep-Read]]"
---
pipline: [[harness-vla]]
NEED TO KOWNE BEFORE READING:
1. primitive: 机器人的“动作 API”或“基础积木”。把一段相对完整、可重复调用的机器人行为，封装成具有明确接口的动作单元。
2. 
# Harness VLA：把冻结 VLA 变成可重试的接触原语

> [!abstract] 一句话总结
> 不让一个 VLA 从语言理解一直负责到整条动作轨迹，而是把**冻结 VLA 降格为局部、接触密集型技能**，再由带记忆的多模态代码代理负责目标重绑定、解析运动、失败检查和重试。论文最有价值的结论不是“VLA 又变强了”，而是：**部署时重新划分职责，也能显著扩大已有 VLA 的有效工作区间。**

## 0. 先给结论

这是一篇典型的 **system / harness paper**。它不提出新的 VLA 架构，也不更新 VLA 权重；核心贡献是围绕冻结策略搭建一个可审计的执行系统：

1. 用 `VLA_ACT` 封装冻结 VLA，只处理抓取、受限放置、按钮、抽屉、龙头等接触密集阶段；
2. 用 `MOVE_TO`、`ROTATE`、`RELEASE` 等解析原语处理定位、预对位、自由空间运输、姿态调整与释放；
3. 用 LLM/VLM 代理做语义重绑定、原语编排、结果观察和失败恢复；
4. 用任务记忆保存“这类任务怎么做”，用全局记忆保存“哪些做法容易成功或失败”。

它的直觉非常扎实：**VLA 最擅长局部接触控制，却不一定适合承担全局语义、长程规划和分布外重定位；解析控制器恰好相反。** Harness VLA 把两者按能力边界组合，而不是要求其中一方包办全部任务。

实验支持“harness 对分布偏移有用”，但不能直接推出“这是一个可部署的通用机器人系统”。绝大多数结果来自仿真，主结果还是每个任务先在 seed 0 上探索并写入任务记忆的 few-shot 协议；系统还依赖 RGB-D、世界坐标、环境内置求解器和 benchmark 成功判据。

## 1. 论文要解决什么问题

端到端 VLA 把下面几件事揉进同一个策略：

- 理解自然语言指令；
- 从当前场景中找出目标对象和目标位置；
- 规划长程步骤；
- 执行自由空间运动；
- 完成抓取、插入、开关门等精细接触动作；
- 判断失败并恢复。

这在训练分布内可行，但一旦部署条件改变，就容易出现结构性失败：

- **语义重定向**：指令改成另一个物体，VLA 仍重复训练时的习惯动作；
- **目标重绑定**：目标容器或谓词改变，动作仍指向旧目标；
- **空间布局变化**：物体互换位置后，VLA 仍向训练时的位置运动；
- **长程误差累积**：一次抓空或接触不稳导致后续整条轨迹失效；
- **接触与非接触职责混杂**：VLA 被迫同时学习精细接触和本可由几何控制稳定完成的运输。

纯代码代理也不是完整答案。它擅长语义和组合，但手写 IK / operational-space controller 很难稳定解决不规则抓取、狭窄放置、铰接物体交互。

因此本文的核心问题是：

> 能否不微调 VLA，也不在部署时持续增加技能库，仅通过一个带记忆、反馈和固定工具接口的代理，把冻结 VLA 可靠地复用到分布外任务？

## 2. 核心设计：不训练更强的 VLA，而是给它加 harness

### 2.1 总体控制回路

每个决策轮次中，代理接收：

$$o_t = (I_t^{rgb}, I_t^d, q_t)$$

其中包含 RGB、对齐的度量深度图，以及末端位姿和夹爪状态等本体信息。代理还读取任务语言 $\ell$、Task Specific Memory 和 Global Memory，然后只输出一个结构化原语调用：

$$c_t = \Pi(o_t, \ell, M_{task}, M_{global}), \qquad c_t \in \mathcal{P}$$

环境执行该原语直到内部终止条件满足，再返回新观察。循环持续到 benchmark 目标谓词成功或预算耗尽。

```text
任务语言 + RGB-D + 机器人状态 + 两类记忆
                    |
                    v
              Agentic Planner
                    |
           选择一个固定原语调用
          /                       \
  解析原语                    VLA_ACT
  定位/运输/旋转/释放          抓取/插入/开关等接触动作
          \                       /
                    v
             新观察 + 执行日志
                    |
          验证、恢复、重对位、重试
```

这里不是传统意义上“高层每隔若干步输出子任务，低层 VLA 连续执行”的简单层级策略。论文把 VLA 和解析控制器统一包装成同一层工具，代理是唯一的认知调度器。

### 2.2 固定原语库

| 原语 | 类型 | 职责 |
|---|---|---|
| `MOVE_TO` | 复合解析原语 | 把末端移动到世界坐标目标 |
| `MOVE_POSE` | 复合解析原语 | 同时调整位置和姿态 |
| `ROTATE_WRIST` | 原子解析原语 | 调整腕部 yaw |
| `ROTATE_PITCH` | 原子解析原语 | 调整腕部 pitch |
| `SET_GRIPPER` | 原子解析原语 | 打开或闭合夹爪 |
| `RELEASE` | 原子解析原语 | 按释放终止条件打开夹爪 |
| `VLA_ACT` | 学习型原语 | 短时运行冻结 VLA，处理局部接触 |
| `NAVIGATE_TO` | RoboCasa 复合原语 | 移动底盘到世界坐标位置 |
| `MOVE_BASE` | RoboCasa 原子原语 | 局部底盘速度控制 |

部署前词表固定，代理不能临时创造新技能。真正的“学习”发生在**如何组合已有原语，以及每个原语在哪些状态下可靠**。

### 2.3 `VLA_ACT` 为什么是关键抽象

`VLA_ACT(prompt, max_chunks, stop)` 将冻结 VLA 封装为一个短时、可中断、可重复调用的接触尝试。代理负责：

- 从当前指令中确定真正的接触目标；
- 用解析运动把机器人重新放到 VLA 熟悉的局部视角和预接触姿态；
- 给 VLA 一条局部、明确的 prompt；
- 在 lift/contact/任务谓词/chunk budget 等条件触发时收回控制权；
- 观察抓空、接触不稳或未完成状态，再重新定位和调用。

这相当于不让冻结策略学习整个新分布，而是**主动把当前状态搬回它擅长的局部分布**。一次 VLA 失败也只污染当前子任务，不再自然传播到整条长程轨迹。

## 3. 两类记忆到底存什么

### 3.1 Task Specific Memory：任务级程序骨架

探索阶段允许代理在某个任务的参考 seed（seed 0）上使用 `RESET` 和更宽松预算，直到找到成功解。成功后保存两部分：

- **procedural memory**：JSONL 原语调用序列；
- **semantic memory**：为什么成功、如何恢复、哪些行为应避免。

坐标不会被当作固定轨迹机械回放，而会被替换为符号化感知查询。部署到 held-out seeds 时，代理保留原语顺序这个“程序骨架”，但根据当前 RGB-D 重新识别对象、支撑面和目标位置。

因此它更像：

```text
先用 VLA 抓住黑碗 -> 重新定位木托盘 -> 解析运输 -> 释放 -> 检查成功
```

而不是：

```text
机械重放 seed 0 中每一个 xyz 坐标和低层动作
```

==For Example:==
```json
//semantic memory
{
  "task": "put the black bowl on the wooden tray",
  "success": true,
  "trace_file": "task_specific_memory_put_black_bowl_on_tray_s0.jsonl",
  "strategy": "use VLA for grasping, then analytic transport and release",
  "avoid": [
    "do not reuse reference xyz values",
    "verify placement with the benchmark success signal"
  ]
}

// procedural memory
{"action":"vla_act","prompt":"grasp the black bowl","max_chunks":2}
{"action":"move_to","xyz":[0.12,-0.08,0.92],"gripper":null}
{"action":"release"}
```
### 3.2 Global Memory：跨任务的成功规则与失败模型

Global Memory 保存与具体任务无关的操作知识，例如：

- 不规则抓取和铰接交互优先用 VLA；
- 稳定抓住后，长距离运输和释放优先用解析原语；
- 夹爪闭合但物体没有随末端移动，应判为空抓；
- 不要因为视觉上“看起来接近”就宣布成功；
- 失败后先重新定位、重对位，再重试 VLA。

作者的定位很准确：记忆不是增加新的技能，而是在逐步标定固定技能库的**适用域与失败边界**。
==For Example:==
```text
Success rule:
Use VLA primitives for contact-rich phases such as irregular grasping
or fixture interaction. After a stable grasp, prefer analytic motion
for long transport and precise placement.

Failure model:
If the gripper closes but the object does not move with the end effector,
treat the attempt as an empty grasp. Re-localize the object and re-stage
before retrying.


```
## 4. 训练与部署协议

这部分决定了应该怎样解读实验数字。

### Few-shot：主要结果采用的协议

1. 每个任务用 seed 0 做探索；
2. 探索期允许 `RESET`，且 wall-clock 预算更宽；
3. 找到成功解后写入该任务的 Task Specific Memory，并积累 Global Memory；
4. seed 0 不计入结果；
5. 在 held-out seeds 上禁用 `RESET`、缩短步数预算，检索并重新落地这条任务记忆。

所以这里的 few-shot 是“**每个任务看过一个参考实例**”，不是只给整个 benchmark 一个示范，也不是对全新任务零样本泛化。

### Zero-shot：论文用了两种不同含义

- LIBERO-Pro Goal：不读取目标设置对应的 Task Specific Memory 和 Global Memory，测试在线推理；
- RoboTwin C2R：从 clean setting 得到任务记忆，再直接迁移到 randomized setting，不在 randomized setting 重新探索。

后者是“对随机化设置零样本”，但仍然有同任务 clean trace，不能理解成完全无记忆的零样本任务解决。

## 5. 实验结果怎么读

### 5.1 标准 LIBERO：harness 没有损伤原能力，但也没有明显增益

| 方法 | Spatial | Object | Goal | LIBERO-10 | 总体 |
|---|---:|---:|---:|---:|---:|
| 冻结 $\pi_{RLinf}$ | 99.0 | 96.0 | 97.0 | 89.0 | 95.3 |
| Harness VLA (CC) | 97.0 | 100.0 | 94.0 | 93.0 | **96.0** |
| AtomVLA | 96.4 | 99.6 | 97.6 | 94.4 | **97.0** |

Harness VLA 相比同一个冻结 VLA 仅增加 0.7pp，而且不是表中最高。正确结论是：**在分布内任务上，它大致保持了原 VLA 的性能。** 论文真正的价值不在这张表。

### 5.2 LIBERO-Pro：对语义和布局偏移的提升非常显著

| 方法 | 指令/目标重定向与位置交换总体成功率 |
|---|---:|
| $\pi_{0.5}$ | 11.0 |
| Cap-X（只报告 6 个非 L10 cell） | 18.2 |
| RATS（只报告 6 个非 L10 cell） | 43.8 |
| 直接冻结 $\pi_{RLinf}$ | 50.0 |
| Harness VLA (Codex) | 72.1 |
| Harness VLA (CC) | **82.4** |

同骨干比较从 50.0% 到 82.4%，是最干净、最有说服力的主结果。它说明将语义重绑定和空间重定位显式交给 planner，确实能修复冻结 VLA“照着训练习惯做”的问题。

论文 headline 的“+38.6pp”来自 82.4 - 43.8，但 RATS 只报告 6 个 cell，Harness 的 overall 含 8 个 cell，二者聚合范围并不完全一致。更稳妥的表述应优先使用同协议、同骨干的 **+32.4pp**。

### 5.3 RoboCasa365：长程厨房任务明显受益，但 unseen 仍然困难

| 方法 | Atomic-Seen | Composite-Seen | Composite-Unseen |
|---|---:|---:|---:|
| RLDX-1 | 60.0 | 21.3 | 5.0 |
| Harness VLA (Codex) | **91.6** | **56.3** | 13.8 |
| Harness VLA (CC) | 79.4 | 47.5 | **15.0** |

Codex 版本 task-weighted overall 为 55.4%，直接 RLDX-1 为 30.0%，提升 25.4pp。导航、对位、重试和长程组合确实适合由 harness 处理。

但 `Composite-Unseen` 只有 13.8%-15.0%，说明系统并没有解决新组合泛化；它主要改善已具备局部能力时的任务组织和恢复。

### 5.4 记忆贡献：语义改写主要靠 planner，空间变化更依赖任务 trace

LIBERO-Pro Goal 的无目标记忆实验：

| 设置 | 无记忆 zero-shot | 有任务记忆 few-shot |
|---|---:|---:|
| Goal-T：指令重定向 | 79.0 | 87.0 |
| Goal-S：位置交换 | 31.0 | 87.0 |

这组对照非常有信息量：

- 对“指令换了目标”这种语义变化，强 planner 本身已经能解决大部分问题；
- 对“物体位置换了”这种几何变化，任务级原语组织和参考经验至关重要；
- Task Specific Memory 的主要价值不是记住语义，而是记住**在什么阶段用哪种控制、何时切换和如何重对位**。

### 5.5 RoboTwin C2R：有增益，但幅度比 LIBERO-Pro 小

| 方法 | C2R 成功率 |
|---|---:|
| 直接 LingBot-VLA | 50.4 |
| Harness VLA (Codex) | 58.0 |
| Harness VLA (CC) | **58.4** |

同一冻结骨干提升约 8pp，证明框架能迁移到双臂环境，但也说明其优势会随 embodiment 和任务结构变化，并非所有场景都有 30-40pp 的提升。

## 6. 三个机制性发现

### 6.1 显式语义重绑定

直接 VLA 在指令改变但画面相近时，容易重复训练时的旧行为；planner 显式解析当前语言并重新寻找目标，再把局部 prompt 交给 VLA。这是在系统层修补 VLA 的 weak language conditioning。

### 6.2 自适应调用与局部重试

作者限制每个 episode 可调用 VLA 的最大次数，成功率在前几次调用后快速上升，随后饱和。结论不是“多调用就好”，而是：

- VLA 调用应当稀疏；
- 每次调用前的状态 staging 很关键；
- 接触失败后局部重试，比让端到端策略继续滚动更稳健。

### 6.3 解析原语包围接触阶段

解析原语并非替代 VLA，而是处理 VLA 前后的非接触结构。在 LIBERO-Pro 中，许多成功 rollout 的最终完成谓词由运输或释放等解析原语触发；在 RoboCasa 和 RoboTwin 中，最终步骤更常是接触操作，因此由 VLA 触发完成。

这是一个清晰的能力分工：

> 解析控制负责“把问题整理成 VLA 会做的局部问题”；VLA 负责“完成解析控制最难写稳的物理接触”。

## 7. 批判性解读：哪些结论可靠，哪些需要保留

### 7.1 强结论

1. **同一冻结骨干加 harness 后，在 LIBERO-Pro 上大幅提升。** 50.0% 到 82.4% 的同协议比较有说服力。
2. **冻结 VLA 可以被重新定位为局部接触 specialist。** 跨 LIBERO、RoboCasa、RoboTwin 的结果共同支持这一系统抽象。
3. **语义偏移和空间偏移需要不同资源。** 无记忆实验揭示 planner 与任务 trace 的分工，这是论文最有研究价值的分析之一。
4. **失败隔离和重试是长程可靠性的核心。** 即使不更新模型参数，改变控制回路也能避免单次接触失败毁掉整条轨迹。

### 7.2 不能过度解读的地方

#### “Few-shot”有较强的任务内先验

主实验对每个任务单独在 seed 0 上探索到成功，再评估 seed 1-10。论文没有把探索成功所需的 reset 次数、LLM 调用量、token、wall-clock 或人工介入成本作为主指标报告。因此 82.4% 代表的是**任务级 bootstrapping 后的初始状态泛化**，不是拿到任意新指令就能直接达到 82.4%。

#### 缺少完整的因果消融

论文展示了：

- 有/无目标设置记忆的局部对照；
- 限制 VLA 调用次数的曲线；
- 最终由哪类原语触发成功的归因。

但没有完整比较 `planner only`、`+ analytic primitives`、`+ Task Memory`、`+ Global Memory`、`+ retry` 等逐项或 factorial ablation。终止原语归因也只说明“最后一步是谁”，不等于该组件对成功的因果贡献。因而目前还不能精确回答 32.4pp 中每个组件分别贡献多少。

#### 使用 benchmark 成功谓词形成了现实差距

状态文件向代理/运行时提供 benchmark success signal，系统还用它做最终验证；`VLA_ACT` 的停止条件也可包含 benchmark predicate。在仿真评估中这保证判分一致，但现实机器人通常没有完美任务完成 oracle。论文自己也承认高低层之间仍是 open feedback loop。若换成视觉 success detector，误判如何影响重试与终止仍未验证。

#### 解析控制依赖强环境接口

系统使用度量深度、世界坐标、机器人本体状态、环境内置 IK/operational-space solver，有些环境还提供 world-map 文件。作者强调 planner 不读取 privileged object pose，这一点是好的；但这些接口在真实杂乱环境中的标定误差、遮挡和动力学偏差并未测试。

#### 结果依赖 planner backbone

LIBERO-Pro 上 CC 为 82.4%，Codex 为 72.1%；RoboCasa 上反而 Codex 为 55.4%，CC 为 48.6%。约 7-10pp 的变化说明 harness 并未完全消除上层模型差异。论文没有报告多次 planner sampling 的方差，也没有系统研究模型能力、成本和延迟的关系。

#### 主要是仿真可靠性，不是现实安全性

四套 benchmark 都是仿真评估，未给真实机器人实验。固定原语和可审计日志确实比端到端黑箱更容易调试，但这不自动等于现实安全：碰撞约束、错误深度、执行延迟、人类进入工作区等问题都不在实验范围内。

## 8. 与相关笔记的关系

### 对比 [[2026-Hi-VLA-Deep-Read|Hi-VLA]]

| 维度 | Hi-VLA | Harness VLA |
|---|---|---|
| 核心问题 | 层级 VLM-VLA 应怎样设计 | 冻结 VLA 应怎样被工具化和调度 |
| 高层输出 | 自然语言子指令 | 结构化原语 JSON 调用 |
| 低层组织 | VLA 是明确的低层 policy | VLA 与解析控制器都属于工具库 |
| 记忆重点 | 历史观察与跨 episode 摘要 | 成功程序 trace + 全局失败模型 |
| 关键收益 | 长程组合与 failure recovery | 分布外重绑定、staging 与 retry |
| 实证特点 | 系统化设计空间消融 | 多 benchmark 强结果，但组件消融较弱 |

两者的共同观点是：**不要让 VLA 独自承担语义推理、长程组织和局部控制。** Harness VLA 比 Hi-VLA 更接近 coding-agent runtime：它强调固定 API、日志、REPL、记忆文件和审计能力。

### 对比 [[2026-Agentic-VLA-Deep-Read|Agentic-VLA]]

Agentic-VLA 选择“继续学习”：用奖励合成、语言探索和经验 warm-start 对 VLA 做在线 RL。Harness VLA 选择“改变使用方式”：冻结权重，把能力边界外的问题交给 planner 与解析控制。

两条路线并不冲突：

- Harness VLA 先用系统分工解决可由语义重绑定和解析运动解决的问题；
- 对依然失败的接触原语，再用 Agentic-VLA 式 RL 做局部适应；
- 这样可以避免用昂贵 RL 去学习本来用几何控制就能稳定完成的自由空间运输。

## 9. 对研究和工程的启示

### 可以直接复用的设计原则

1. **把基础策略包装成可终止、可重试的工具。** 工具接口至少应包含 prompt、预算和停止条件。
2. **先缩小策略的职责，再考虑微调。** 分布外失败可能来自错误的任务分配，而非模型局部控制能力不足。
3. **记忆应保存程序结构和失败约束，而不是绝对坐标。** 这比轨迹回放更容易跨布局复用。
4. **接触前 staging 是策略适配的一种形式。** 不改模型，也可以通过改变输入状态分布提高成功率。
5. **把每次物理执行变成可观察的事务。** command、state、log、done flag 和 trace 让机器人 agent 更容易审计和复现。

### 值得继续做的实验

- 报告 task bootstrapping 的 reset、时间、token 和美元成本分布；
- 做完整组件消融，并用相同 planner seed 重复评估；
- 用学习到的视觉 success detector 替换 benchmark oracle；
- 在真实机器人上加入深度噪声、标定漂移和碰撞安全约束；
- 将 full trace 压缩为参数化 behavior tree / option graph，测试跨任务组合；
- 当固定词表反复出现相同长组合时，再与 ASPIRE 式 skill discovery 结合；
- 只对高失败率的 `VLA_ACT` 子区域做 LoRA/RL，而不是端到端微调整个策略。

## 10. 最终评价

| 维度 | 评价 |
|---|---|
| 核心思想 | 8.5/10：简单、正确，切中了 VLA 部署的职责错配 |
| 方法新颖性 | 7/10：原语编排和记忆并非全新，但“冻结 VLA 作为唯一学习型接触原语”的系统化落地很清晰 |
| 实验广度 | 8.5/10：覆盖桌面、厨房、双臂和多类分布偏移 |
| 因果分析 | 6/10：缺少完整组件消融和成本分析 |
| 现实可部署性 | 5.5/10：仍依赖仿真接口、成功 oracle 和强 planner，未做真机验证 |
| 综合 | **8/10**：值得精读，也值得作为 agentic robotics 系统基线复现 |

> [!summary] 只记住三件事
> 1. 冻结 VLA 不必包办全任务；把它限制在接触密集的局部阶段，反而更可靠。
> 2. Harness 的真正能力来自“语义重绑定 + 解析 staging + 局部 retry + 程序记忆”，不是单纯多调用几次 VLA。
> 3. 论文证明了仿真中的系统级收益，但 82.4% 是每任务 seed-0 bootstrapping 后的 few-shot 结果，不能等同于真实世界的任意任务零样本成功率。

## 资料

- 论文：<https://arxiv.org/abs/2607.08448>
- 项目主页：<https://harnessvla.github.io/>
- 本地 PDF：`C:\Users\huawei\Zotero\storage\ESDD9J8D\Zhang 等 - 2026 - Harness VLA Steering Frozen VLAs into Reliable Manipulation Primitives via Memory-Guided Agents.pdf`
