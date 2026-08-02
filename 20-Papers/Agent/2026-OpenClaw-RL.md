---
title: "OpenClaw-RL: Train Any Agent Simply by Talking"
authors:
  - Yinjie Wang
  - Xuyang Chen
  - Xiaolong Jin
  - Mengdi Wang
  - Ling Yang
year: 2026
venue: arXiv preprint
arxiv: 2603.10165v2
tags:
  - paper
  - agent
  - agentic-rl
  - online-learning
  - personalization
  - process-reward-model
  - on-policy-distillation
  - next-state-signal
status: read
rating: 8/10
date-read: 2026-08-02
url: https://arxiv.org/abs/2603.10165
code: https://github.com/Gen-Verse/OpenClaw-RL
pdf: "C:/Users/huawei/Zotero/storage/7EIYWEL4/2603.pdf"
---

# OpenClaw-RL：把每次 Agent 交互的“下一状态”变成在线训练信号

## TL;DR

OpenClaw-RL 的核心洞察是：**Agent 执行动作后出现的下一状态，本身就是低成本监督。** 下一状态可以是用户的追问或纠正、工具返回值、终端报错、测试结果、GUI 变化。它通常同时回答两个不同问题：

1. 上一步做得好不好？这是 **evaluative signal**，可压缩成标量奖励；
2. 上一步应该怎样改？这是 **directive signal**，可提炼成 hint，再通过 hint-conditioned teacher 提供 token 级监督。

论文将二者组合为 GRPO/PPO 式标量奖励目标与 on-policy distillation（OPD）目标的加权和。为防止 hint 条件下的 teacher 分布与原 student 分布相差过大，作者用 **top-k 支持集重叠**选择 hint，并截断逐 token 的 teacher/student 对数概率差。

最值得记住的不是“聊天就能训练”这个口号，而是下面这条更准确的链路：

> action 后的 next state -> 判断成败 + 提炼纠错方向 -> 异步转成 policy gradient -> 权重更新后继续服务。

论文的贡献同时落在两个层面：

- **系统层**：把服务、环境交互、PRM 判断、训练拆成异步循环，让真实终端持续上传轨迹而不阻塞推理；
- **算法层**：把“频繁但信息少”的评价信号与“稀疏但信息密”的指令信号统一使用，并处理 hint distillation 的分布失配。

我的总体判断：这是一个概念清晰、工程链路完整、消融方向正确的在线 Agent RL 框架；但“真实个人 Agent 会在长期使用中安全地持续进化”仍是愿景。当前个人化证据来自模拟用户、GSM8K 工作流和手工可判定偏好，尚未覆盖真人反馈噪声、偏好漂移、遗忘、跨用户冲突、隐私和数据投毒。

## 1. 问题：部署后的交互数据为什么没有直接变成学习信号？

传统 Agent RL 往往按批次收集数据：训练端启动 rollout，环境执行，得到 reward，再更新模型。真实个人 Agent 的使用模式却不同：

- 环境在用户自己的电脑或不断变化的云服务中；
- 交互稀疏、跨 session、异构且带有私人状态；
- 用户不能等待 PRM 和训练完成后才拿到响应；
- 大量监督并不是显式的 thumbs-up/down，而藏在后续事件里。

例如：

| Agent 动作后的 next state | 评价信息 | 指令信息 |
|---|---|---|
| 用户说“很好” | 正向 | 通常没有 |
| 用户换一种方式重复提问 | 隐含负向 | 可能暴露遗漏点 |
| 用户说“你应该先读文件” | 负向 | 应先检查文件 |
| 测试全部通过 | 正向 | 通常没有 |
| `stderr` 指向某行类型错误 | 负向 | 指向具体修正方向 |
| GUI 状态按预期变化 | 正向 | 可能提供进度信息 |

因此，论文的关键重新定义是：监督单位不是“一个人工标签”，而是二元组 $(a_t, s_{t+1})$。其中 $a_t$ 是 Agent 的动作，$s_{t+1}$ 是动作后实际出现的用户或环境状态。

## 2. 系统架构：让在线训练不打断在线服务

OpenClaw-RL 将系统拆成四个异步组件（论文 Figure 1，pp. 1, 5-6）：

```text
Personal device / cloud environment
        |  request                         | interaction trajectory
        v                                  v
Policy server (SGLang)              Environment server
        ^                                  |
        | weight refresh                   v
Policy trainer (Megatron) <--------- PRM / Judge server
```

- **Policy server**：始终提供推理 API；
- **Environment server**：接收个人设备或云端环境回传的交互流；
- **PRM/Judge server**：异步读取 $(a_t,s_{t+1})$，生成评价票与 directive hint；
- **Training engine**：消费已标注样本并更新权重，随后平滑刷新 serving 权重。

底层建立在 `slime` 的异步 RL 基础设施之上。与普通 RL 训练“训练器拥有环境”不同，这里环境可以是用户设备，客户端只需通过 HTTP 调用策略并回传轨迹。论文声称 serving、rollout、PRM judging、training 四个 loop 完全解耦，训练和信号提取不会造成 serving interruption。

### 支持的环境

| 场景 | 环境 | next-state signal | horizon |
|---|---|---|---|
| OpenClaw | 个人设备 | 用户回复、工具结果 | long |
| Terminal | shell sandbox | stdout/stderr、exit code | long |
| GUI | 屏幕与 accessibility tree | 视觉变化、任务进度 | long |
| SWE | repo 与测试套件 | test verdict、diff、lint | long |
| Tool-call | API/function runtime | 返回值、error trace | medium |

统一性的来源不是环境表示统一，而是这些异构状态最终都可以被 PRM 解释为“前一 action 的证据”。

## 3. 两种互补监督

### 3.1 Evaluative signal：把下一状态压成一个分数

给定 $(a_t,s_{t+1})$，PRM 查询 $m$ 次，每次投票 $\{+1,-1,0\}$，多数票 $r_t$ 作为该步的标量奖励。它的优势是覆盖率高：即使 next state 不包含明确修改意见，成功返回、失败报错、用户继续追问也可以被评价。

代价是信息瓶颈：一个 turn 的全部信息被压成一个数。它告诉模型方向，却不说明哪些 token 或决策需要改变。

### 3.2 Directive signal：把下一状态提炼成纠错 hint

PRM 还判断 $s_{t+1}$ 是否包含有意义、可执行的纠错信息。如果有，就把它提炼为短 hint $h$；如果只有“谢谢”或测试通过，就不强行生成 hint。

对同一个策略模型做两次条件化：

$$
\pi_{old}(\cdot\mid x,y_{<i})
\quad\text{vs.}\quad
\pi_T(\cdot\mid x\oplus h,y_{<i})
$$

左边是原 prompt 下的 student/rollout 分布；右边是加入 hindsight hint 后的 teacher 分布。这里的 teacher 本质上是 **同一个模型看到更多纠错上下文后的分布**，不是必须额外调用一个更强的固定教师模型。

Directive signal 的优点是每个 token 位置都能获得一个分布差，因此信息密度远高于标量 reward；缺点是它只存在于部分 next states，而且错误或含糊的 hint 会把 teacher 拉到 student 几乎没有概率质量的区域，使训练不稳定。

### 3.3 Hybrid objective

二者共享同一批轨迹和一次策略更新：

$$
\mathcal{L}^{hybrid}_i
=w_{RL}\mathcal{L}^{GRPO}_i
+w_{OPD}\mathcal{L}^{OPD}_i,
$$

默认 $w_{RL}=w_{OPD}=1$。

直觉上：

- GRPO 项回答“这整段输出应该整体增大还是减小概率”；
- OPD 项回答“在当前位置，概率质量应该更具体地移向哪里”；
- 没有 directive hint 的样本仍能通过 GRPO 学习；
- 有高质量纠错的样本则同时获得序列级与 token 级信号。

这也是论文最重要的方法论判断：**反馈的覆盖率与反馈的信息密度是两个独立维度，好的学习系统不应只选其中一个。**

## 4. 核心难点：hint-conditioned distillation 为什么会不稳定？

假设 hint 很强或写偏了，teacher 可能偏好一组 student 原本几乎不会生成的 token。此时 teacher/student 分布失配，importance ratio 或 log-probability gap 会出现极端值，少数 token 主导梯度，最终表现为训练震荡、输出越来越长甚至大量截断。

OpenClaw-RL 用两步缓解这个问题。

### 4.1 Overlap-guided hint selection

先为同一个 next state 生成 $M$ 个候选 hint。在 response 的第 $i$ 个 token 位置，取：

$$
S_i^q=\operatorname{topk}\{\pi_{old}(\cdot\mid x,y_{<i})\},
$$

$$
S_{i,h}^p=\operatorname{topk}\{\pi_T(\cdot\mid x\oplus h,y_{<i})\}.
$$

候选 hint 的局部重叠分数为：

$$
O[h,i]=|S_i^q\cap S_{i,h}^p|.
$$

有两种选择方式：

- **sequence-optimal**：最大化 $\sum_i O[h,i]$，整条轨迹只选一个 hint；
- **token-optimal**：每个位置分别最大化 $O[h,i]$，不同 token 可使用不同 hint。

作者主实验偏向 sequence-optimal：它和 token-optimal 的个人化效率相近，但在大批量 Agentic RL 中曲线更稳定，也更容易解释和实现。

一个容易误读的点是：重叠最大并不代表 hint 在语义上“最正确”，而代表 hint 引导出的 teacher 仍停留在 student 的高密度支持区域。因此它首先是一种 **可学习性/信赖域启发式**：选择既能提供方向、又不会让更新跨度过大的老师。

### 4.2 Top-k OPD 与 log-probability-difference clipping

选定 $h^*$ 后，默认只在 student top-k 集合 $S_i=S_i^q$ 上蒸馏。对词表 token $v\in S_i$，定义：

$$
\Delta_v=operatorname{clip}\left(
\log\pi_T(v\mid x\oplus h^*,y_{<i})
-\log\pi_{old}(v\mid x,y_{<i}),-C,C
\right).
$$

再用 student 在 top-k 内归一化后的概率 $w_v$ 加权，得到 $A_v=\Delta_vw_v$，并结合 current/old policy ratio 写成 PPO clipped surrogate。完整公式见论文 Eq. (1), p. 9。

两种约束分工不同：

- overlap selection 在 hint 层面避免选到过远的 teacher；
- $\Delta$ clipping 在 token 层面限制单次蒸馏优势的幅度；
- PPO ratio clipping 再限制当前策略相对 rollout policy 的更新幅度。

因此它实际上有三层 trust-region 风格的稳定器，而不只是“加一个 KL”。

## 5. 长时程 Agent：把 process reward 与 outcome reward 合并

终局奖励只能告诉模型任务最终成败，无法定位长轨迹中的哪一步造成失败。OpenClaw-RL 用 next state 对每一步计算 PRM reward，再与 outcome reward 相加。论文采用的步奖励形式可概括为：

$$
R_t=R_{outcome}+\frac{1}{T}\sum_{j=1}^{T}r_j^{PRM}.
$$

由于真实 terminal/GUI 状态难以按“相似状态”聚类，作者用相同 step index 的 action 组成组并标准化 advantage。这是务实但较粗的近似：第 3 步的 action 未必处于语义相似状态，尤其当不同轨迹进度差异很大时。

需要区分两件事：

- terminal/GUI/SWE/tool-call 的通用框架实验主要证明异构环境可接入及 process reward 有益；
- hybrid OPD 的重点验证则在 personal OpenClaw、ReTool 和 RLVR 设置中完成，并非所有 Figure 5 环境都完整验证了同一套 directive objective。

## 6. 实验设计

### 6.1 个人 Agent 个性化

基础 policy 与 reward model 为 `Qwen3-4B-Thinking-2507`；用户由 `Qwen3-32B` 模拟。每个 session 从 GSM8K 取一个任务，第一条用户消息固定且不暴露偏好。若模型连续三个 session 的首次响应都满足偏好，就认为“优化完成”，指标是达到该条件所需的最少 session 数，5 次独立实验取均值。

三种模拟用户：

| 角色 | 想学到的偏好 | 自动判据 |
|---|---|---|
| Student | 作业回答不显得 AI-like | 避免粗体、编号、boxed answer、过度格式化 |
| TA | 评分具体、详细 | 输出至少 100 tokens 等规则 |
| Teacher | 评论友好、耐心 | 包含 warm phrases 等规则 |

训练设置包括 $k=4$、每样本 3 个候选 hints、每收集 16 个样本触发异步训练；个人设置的 log-probability gap clip 为 $C=1$。系统合计使用 8 张 GPU：policy actor 4、policy server 2、PRM actor 1、PRM server 1（Appendix A.1, p. 21）。

### 6.2 通用 Agent

| 场景 | Policy | 数据/评测 | 并行环境 |
|---|---|---|---:|
| Terminal | Qwen3-8B | SETA RL data | 128 |
| GUI | Qwen3VL-8B-Thinking | OSWorld-Verified 的训练集子集 | 64 |
| SWE | Qwen3-4B | SWE-Bench-Verified | 64 |
| Tool-call | Qwen3-4B-SFT | DAPO RL data / AIME 2024 | 32 |

每个任务采样 8 条 rollout；最大交互步数分别为 GUI 30、SWE 20、Terminal 10。需要注意 GUI 报告的是训练集表现，并排除了 Chrome 与 multi-app tasks，不能当作严格的 held-out 泛化结果。

### 6.3 Hybrid RL 扩展

- 多轮 tool-call：ReTool-4B policy，Qwen3-8B PRM；
- RLVR：DeepSeek-R1-Distill-Qwen-1.5B policy，Qwen3-4B PRM，在 DAPO 训练、AIME 2024 评测；
- 每步 32 tasks，每 task 8 rollouts；$k=4$、每样本 3 hints、$C=2$；
- AIME 结果报告 20 次独立评测平均值。

## 7. 主要结果怎么读

### 7.1 个人化样本效率

达到偏好的平均 session 数，越低越好（Table 3, p. 11）：

| Joint optimization | Hybrid RL | GRPO | OPD | Mem0 | Cognee |
|---|---:|---:|---:|---:|---:|
| Student | **11.6** | 15.4 | 30.8 | 13.6 | 14.6 |
| TA | **8.2** | 12.0 | 34.0 | 15.8 | 15.4 |
| Teacher | **11.4** | 14.8 | 24.4 | 14.2 | 14.8 |
| Average | **10.3** | 14.1 | 29.7 | 14.5 | 14.9 |

| Separate optimization | Hybrid RL | GRPO | OPD | Mem0 | Cognee |
|---|---:|---:|---:|---:|---:|
| Average | **15.0** | 21.1 | 29.4 | 15.1 | 15.1 |

可得三个结论：

1. OPD 单独使用很慢，说明稀疏 directive signal 不足以覆盖全部学习事件；
2. hybrid 在 joint 场景优势最大，表明共享权重可能复用了三种相互关联的行为偏好；
3. separate 场景里 hybrid、Mem0、Cognee 几乎打平，因此“权重学习显著胜过 memory”不能脱离 joint 协议泛化。

此外，Mem0/Cognee 增加推理上下文开销，RL 更新权重后不增加 prompt 长度；但 RL 的训练 GPU、持续更新风险和回滚成本也明显更高，不能只比较 inference context。

### 7.2 Process reward 的收益

在 tool-call 和 GUI 中，outcome + process reward 分别达到 0.25 和 0.33，而 outcome-only 为 0.19 和 0.31（p. 12）。Tool-call 的绝对提升明显，GUI 提升较小。作者也承认 PRM server 带来额外资源成本。

Figure 5 中 Terminal、GUI、SWE、Tool-call 的训练曲线总体上升，说明统一基础设施能够驱动这些环境训练；但多条曲线只展示单一方法、没有误差带或同图强基线，因此更适合作为“系统覆盖面展示”，而非四环境 SOTA 证据。

### 7.3 Hybrid objective 的泛化

Figure 6 显示：

- ReTool 中 Hybrid 最终约 22.5 accuracy，优于 PRM+Outcome 约 19 与 Outcome 约 17；
- RLVR 中 Hybrid 最终约 29.3，Outcome 约 27.2。

这支持 directive token supervision 在个性化以外也有效。尤其 ReTool 中，单纯加入 process scalar reward 仍弱于 hybrid，说明“知道哪一步错”与“知道应该朝哪个 token 分布修正”确实不是同一种监督。

### 7.4 Hint selection 消融

在 Qwen3-32B 的 joint personal 设置中（Table 4, p. 12）：

| 方法 | 达标平均 sessions |
|---|---:|
| Hybrid, token-optimal | **12.4** |
| Hybrid, sequence-optimal | 12.5 |
| Hybrid, random hint | 16.1 |
| GRPO | 15.8 |
| OPD | 29.9 |

随机 hint 不仅差于 overlap selection，甚至略差于 GRPO，说明低质量 directive supervision 会抵消其信息密度优势。Figure 7 进一步显示 sequence-optimal 在 ReTool/RLVR 中比 random 更稳定。

### 7.5 $k$ 与支持集

Table 5 表明 $k$ 从 2 增至 4 时，平均 sessions 从 20.2 降到 10.3；继续增至 8/20 只有微小收益。作者因此取 $k=4$。仅监督实际 top-k overlap 集合可省计算，但 $k=4$ 时平均值从 10.3 变差到 13.5。

论文称“token-level OPD”退化明显，表中平均为 31.0。这里的 token-level OPD 指只使用 rollout 中实际生成 token 的 log-prob gap，**不要与 token-optimal hint selection 混淆**：后者仍在 top-k 词表支持集上训练，只是每个位置可选择不同 hint。

### 7.6 Clipping 消融

无 log-probability-difference clipping 时，ReTool 平均响应长度不断增长，最终截断率 0.5；有 clipping 时为 0.2（Figure 8, p. 32）。这说明纯蒸馏更新可能形成 length inflation，而 outcome reward 又没有充分惩罚冗长输出。

不过 clipping 后 20% 的截断率依然不低，说明机制是缓解而非解决长度偏移。

## 8. 论文真正证明了什么？

### 证据支持较强的主张

- next state 可以统一承载用户、工具、terminal、GUI、SWE 等多种环境反馈；
- evaluative 与 directive signal 在覆盖率和信息密度上互补；
- 在论文设定中，hybrid objective 比单独 GRPO 或 OPD 更省 session；
- 随机选择 hint 会降低效率与稳定性，top-k overlap 是有效的选择启发式；
- 对数概率差截断能抑制极端更新、长度膨胀和截断率；
- serving、signal extraction 与 training 可以通过 server-client 异步架构解耦。

### 证据尚不足的主张

- **真实用户长期在线个性化**：个人实验全部由 LLM 模拟用户完成；
- **自然偏好学习**：偏好多是容易被字符串、长度和格式规则判断的表面风格；
- **持续学习不会遗忘**：没有长期 retention、旧能力回归或灾难性遗忘测试；
- **多用户安全共享**：joint optimization 提升效率，但没有测用户偏好冲突、串扰和隐私泄漏；
- **对恶意反馈鲁棒**：没有 feedback poisoning、prompt injection 或 sybil user 实验；
- **跨环境统一算法全面胜出**：Figure 5 更偏基础设施覆盖，完整 hybrid 对照并未覆盖每个环境；
- **训练成本优于 memory**：论文强调 RL 没有推理上下文开销，却没有给端到端 GPU 成本、延迟与能耗对比。

## 9. 关键局限与风险

### 9.1 PRM 同时是信号提取器也是单点故障

PRM 决定 reward 正负、是否存在 hint、hint 内容以及候选 hint 的生成质量。若 PRM 误读讽刺、礼貌性同意、工具噪声或用户的新需求，错误会直接进入权重。Table 7 只说明 4B 与 8B PRM 在简单个人化任务中表现接近，不能证明复杂开放环境中的 judge robustness。

### 9.2 next state 不总是由前一步 action 单独导致

真实系统可能存在延迟工具返回、外部并发修改、用户突然换任务、网络错误。把 $s_{t+1}$ 全部归因于 $a_t$ 会制造 credit assignment error。需要 causal attribution、事件 ID、工具调用 lineage 和超时边界，而不只是相邻 turn 配对。

### 9.3 “最接近 student 的 teacher”可能过于保守

top-k overlap 提高稳定性，但也偏好与当前策略相近的纠正。若 student 的高概率区域整体错误，最有价值的 hint 可能恰好带来较大分布移动。该准则在 exploitation 上合理，却可能牺牲必要的 conceptual jump。

更完整的选择目标可同时考虑：

$$
\text{utility}(h)
=\text{correctiveness}(h)
-\lambda\,\text{distribution shift}(h)
-\mu\,\text{risk}(h).
$$

当前论文主要优化中间一项，没有显式衡量 hint 的语义正确性和安全性。

### 9.4 在线权重更新比 memory 更难撤销和审计

Memory 条目可以定位、删除、设 TTL 或按用户隔离；权重更新会把大量样本压进不可解释参数，删除某个用户数据并不等于消除其影响。长期个人 Agent 需要：

- 原始样本与梯度更新的 provenance；
- 用户级 adapter / LoRA 隔离；
- shadow evaluation 和 canary deployment；
- 自动 rollback 与冻结窗口；
- machine unlearning 或可删除个性化层。

### 9.5 隐私与投毒是核心问题，不是附录问题

论文结论也明确承认两类风险（p. 16）：负面或恶意纠正会污染在线更新；个性化权重会编码用户偏好与私人信息，成为攻击目标。真实部署还需考虑 API key、终端输出和文件内容进入 PRM/训练链路后的访问控制与保留策略。

## 10. 工程上如何更稳妥地落地

不应把每条用户回复直接送进基础模型更新。更合理的是分层策略：

```text
raw interaction stream
  -> causal/event attribution
  -> PII and secret filtering
  -> adversarial / low-confidence rejection
  -> evaluative + directive extraction
  -> user-scoped replay buffer
  -> offline or shadow update
  -> regression + safety gate
  -> canary serving
  -> promote or rollback
```

具体建议：

1. **先更新用户级 adapter，而非共享 base model**：降低跨用户串扰和删除难度；
2. **显式区分 preference、skill 与 factual correction**：风格偏好可个性化，事实不应因单个用户断言写入权重；
3. **给反馈加置信度与来源权重**：可验证测试结果应高于含糊自然语言反应；
4. **维护正负 replay 和 anchor tasks**：每次更新同时检查旧能力、安全策略和通用指令遵循；
5. **对 directive hint 做独立验证**：生成 hint 的 PRM 与验证 hint 的 judge 最好有模型、规则或数据来源上的独立性；
6. **采用更新预算**：限制每用户、每时间窗的样本数、梯度范数和 KL 漂移；
7. **保留 memory fallback**：低置信偏好先进入可撤销 memory，重复稳定出现后再考虑权重化。

## 11. 可以迁移出的研究启发

### 启发 1：next-state supervision 是 Agent 数据飞轮的统一接口

Agent 的各领域差异很大，但 action 后的状态变化几乎处处存在。可以把训练数据 schema 统一成：

```text
(context, action, next_state, source, causal_link,
 evaluative_label, directive_hint, confidence, privacy_scope)
```

这比为 terminal、browser、robot、SWE 各自设计独立 reward pipeline 更容易扩展。

### 启发 2：反馈应按“覆盖率 × 信息密度”组织

标量 reward 与 token hint 不是互斥路线。进一步还能加入：

- structured patch：代码 diff、tool argument correction；
- state goal：期望 GUI/机器人状态；
- constraint：不应触发的动作或安全边界；
- preference pair：两个可接受响应之间的相对偏好。

它们可以形成从低带宽、广覆盖到高带宽、低覆盖的监督谱系。

### 启发 3：teacher selection 本质是 update geometry control

论文不是按 hint 文本质量打分，而是观察 hint 对输出分布造成的几何变化。这提示可以用更通用的 selection criterion：KL、Jensen-Shannon、Fisher distance、梯度余弦相似度，或预测更新后的 validation improvement。

### 启发 4：个性化更适合“权重 + memory”的混合层级

- 短期、易变、用户显式事实放 memory；
- 高频、稳定、跨任务复现的风格偏好放 adapter；
- 通用可验证技能再汇总到共享 policy；
- 敏感信息永远不应通过共享权重吸收。

这比论文中把 RL 与 Mem0/Cognee 当作竞争基线更符合真实产品结构。

### 启发 5：在机器人中，next state 更丰富也更危险

如果迁移到 embodied agent，next state 可包括 RGB/depth、物体位姿、接触力、控制器 return code、collision、任务 predicate。它们能提供比文本更可靠的 process signal；但错误动作可能不可逆，不能像 terminal 一样廉价重试。因而必须先做安全过滤、counterfactual simulation 和低层控制约束，再允许在线权重更新。

## 12. 值得追踪的后续实验

1. 真人用户、数周时间尺度下的 preference drift 与 retention；
2. 训练后在通用 benchmark、旧用户偏好和安全集上的回归；
3. 冲突用户共同更新一个 policy 时的 interference matrix；
4. 1%、5%、10% 恶意或错误反馈下的 poisoning curve；
5. overlap selection 与语义正确性 judge、KL trust region 的正交消融；
6. adapter、memory、base-weight update 的同等总成本比较；
7. 当 next state 由多个并发事件产生时的 causal credit assignment；
8. 用户删除数据后，个性化行为和隐私泄漏是否真正消失；
9. process reward 按 step index 分组与按 learned state embedding 分组的比较；
10. clipping 后仍有 20% truncation 时，加入显式 length reward 或 adaptive KL 是否更稳。

## 13. 一句话评价

> OpenClaw-RL 最重要的贡献，是把 Agent 使用过程中原本被当作日志的 next state，系统化地升级为“评价 + 纠错方向”两种在线监督；它已经证明这条训练闭环在受控环境中有效，但距离可信的真实个人化部署，还差因果归因、长期评测、用户隔离、可撤销更新与抗投毒机制。

## Source

- Wang, Y., Chen, X., Jin, X., Wang, M., & Yang, L. (2026). *OpenClaw-RL: Train Any Agent Simply by Talking*. arXiv:2603.10165v2.
- Local PDF: `C:/Users/huawei/Zotero/storage/7EIYWEL4/2603.pdf`
- Code: https://github.com/Gen-Verse/OpenClaw-RL

