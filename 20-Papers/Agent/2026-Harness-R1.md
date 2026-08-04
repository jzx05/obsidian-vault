---
title: "Harness-R1: Learning to Edit Executable Runtime Harnesses from Agent Failure Trajectories"
authors:
  - Shuai Shao
  - Kangning Zhang
  - Qingyao Li
  - Shijian Wang
  - Hao Wang
  - Wenxiang Jiao
  - Yuan Lu
  - Yi Guo
  - Weiwen Liu
  - Weinan Zhang
year: 2026
venue: arXiv preprint
arxiv: 2608.02276v1
tags:
  - paper
  - agent
  - agent-harness
  - self-evolving-agent
  - reinforcement-learning
  - GRPO
  - runtime-adaptation
  - executable-patch
status: read
rating: 8.5/10
date-read: 2026-08-04
url: https://arxiv.org/abs/2608.02276
code: https://github.com/DeepExperience/Harness-R1
model: https://huggingface.co/ShaoShuai0605/Harness-R1
pdf: "C:/Users/huawei/Zotero/storage/NB3K6GI8/Shao et al. - 2026 - Harness-R1 Learning to Edit Executable Runtime Harnesses from Agent Failure Trajectories.pdf"
related:
  - "[[2026-TTHE-Test-Time-Harness-Evolution]]"
  - "[[2026-SkillClaw]]"
---

# Harness-R1：从 Agent 失败轨迹中学习编辑可执行运行时

## TL;DR

Harness-R1 不直接训练执行任务的 Agent，而是训练一个独立的 **harness engineer**：它读取目标 Agent 的一批失败轨迹，生成能安装到现有 runtime 中的 Python hook；冻结的目标 Agent 在同一批任务上重新运行，补丁造成的真实 reward 增量再通过 GRPO 更新 engineer。

```text
目标 Agent 的失败轨迹
  -> 压缩为 failure packet
  -> 9B harness engineer 生成可执行 patch
  -> 语法/接口/完整性验证
  -> 安装到冻结目标 Agent 的 runtime
  -> 同批任务全部重跑（包括原来成功的任务）
  -> 新旧 reward 差作为 engineer reward
  -> 只更新 engineer，不更新目标 Agent
```

一句话抓住论文：

> 它把“根据失败修改 Agent 外围代码”从一次性的 LLM 提示或程序搜索，变成了一个由下游执行结果监督、可以被强化学习训练的独立策略。

在 Qwen3.5-9B 目标 Agent 上，三项 benchmark 的平均成功率由 **44.3% 提升至 53.6%（+9.3 个百分点）**；目标 Agent 自身先做 SFT 后，再训练一个 target-specific engineer，可由 **59.2% 提升至 64.2%（+5.0）**。最有说服力的证据不是单一主表，而是：SFT-only engineer 只有 46.4%，固定的大模型编辑器最好为 48.8%，说明收益确实与 outcome-grounded training 有关，而不只是“换一个更强模型来写代码”。

我的判断：这是一篇概念清晰、工程闭环完整的工作，真正推进了 learned harness engineering。但它学到的是**受限 hook API 内的单轮、按 benchmark 生成的 transductive patch**，还不是可持续维护任意真实代码库的通用 harness engineer；“co-evolution”也只展示了一次 Agent SFT 后重新训练 engineer，并未进行多轮交替进化。

## 1. 论文在解决什么问题？

Agent 能力不只来自模型权重，还来自包围模型的运行时：

- 初始 context、memory、skills 如何装载；
- 决策前给模型什么状态和接口约束；
- 模型提出 action 后是否校验、重写或阻止；
- 环境返回错误、循环或停滞时如何恢复。

这些部分合称 **agent harness**。相同模型权重搭配不同 harness，可能表现得像两个能力明显不同的 Agent。

已有 harness 优化通常存在一个关键断点：编辑器本身是固定的。强模型可以根据轨迹提出补丁，搜索系统也可以筛选补丁，但“哪些编辑真正改善下游执行”并没有反过来更新编辑器参数。结果是补丁在文字上合理、语法上正确，却可能降低任务成功率。论文 Figure 1 中，固定 Self-Refine 在三个环境均退化；若用 frontier model 直接充当 harness editor，平均增益也不稳定。

Harness-R1 改变的是学习对象：

| 路线 | 被优化的对象 | Harness 编辑器是否从结果学习 |
|---|---|---|
| Agent SFT / RL | 任务 Agent 权重 | 否 |
| Prompt/skill 搜索 | 单个外部 artifact | 通常否 |
| 固定大模型写 harness | 最终 patch | 否，编辑器参数固定 |
| Harness-R1 | **生成 patch 的 engineer policy** | **是，使用重跑后的任务收益** |

## 2. 可编辑面：四个 lifecycle hook

Engineer 不能任意修改整个 Agent repo，而是在宿主 runtime 暴露的四个位置安装受约束的 Python hook：

| Hook | 调用时机 | 允许产生的效果 | 典型用途 |
|---|---|---|---|
| `on_init` | 第一次决策前 | 添加通用 task guidance / tool hint | 初始化流程、技能和约束 |
| `make_pre_hint` | 每次决策前 | 注入依赖当前状态的消息 | 提醒接口约束、补充检索信息 |
| `on_before_action` | Agent 提议 action 后、环境执行前 | block、reprompt，部分环境可 rewrite/force | 阻止超预算购买、非法或重复动作 |
| `on_post_step` | 环境反馈后 | 注入恢复提示，部分环境可安排下一动作 | 处理错误、循环、停滞和 fallback |

Engineer 输出严格的 JSON patch，动作类型为 `ADD_CODE_HOOK`，代码必须定义 `hook(ctx, nb)`，不能 import、不能依赖 global state，也不能硬编码 task ID、产品 ID、对象编号或答案。宿主只解释预定义的 structured effect。

这个设计很关键：它把高风险的“任意代码编辑”收缩成**有类型的可执行 intervention DSL**。论文因此能够稳定验证和批量执行 patch，但也意味着“lifecycle-wide editing”只是在这四个预埋扩展点内成立。

## 3. 从失败到 reward 的形式化闭环

设冻结目标 Agent 为 $A$，任务批次为 $B=\{x_i\}_{i=1}^{n}$。先运行原始 harness，得到基线 reward $R_i^0$；仅将失败 episode 压缩为 failure packet $s_B$，其中保留任务约束、关键 action-observation 片段、结果和必要环境状态。

Engineer $H_\theta$ 读取一次 $s_B$，生成 batch-conditioned patch $P$。安装后，目标 Agent 在完整任务批次上重跑，得到 $R_i^P$。补丁收益为：

$$
\Delta_B(P)=\frac{1}{n}\sum_{i=1}^{n}(R_i^P-R_i^0).
$$

Engineer reward 定义为：

$$
r(B,P)=
\begin{cases}
\Delta_B(P), & P\text{ 有效且评测完整},\\
0, & \text{无效、空操作或评测未完成}.
\end{cases}
$$

几个容易忽略的细节：

1. 输入只呈现失败轨迹，但重跑覆盖**完整 batch**，包括原先成功的任务，因此同批次内的 regression 会进入 reward。
2. Engineer 在 rollout 前一次性生成 patch，不参与后续交互，也不能自己调用环境工具或提交答案。
3. 无效 patch 得 0 分，而有效但造成退化的 patch 得负分；语法正确不是奖励。
4. 基线轨迹和 failure packet 预先缓存，在线阶段的主要成本来自每个候选 patch 对冻结 Agent 的完整重跑。

## 4. 两阶段训练

### 4.1 Cold-start SFT：先学会写合法补丁

GPT-5.5 teacher 根据与 RL 不相交的 failure packet 生成候选。系统只保留：

- 可执行；
- 能完成同批重跑；
- reward change 非负；
- 每个 packet 至多一个。

最终不是正文概述的“约 1K”，而是附录给出的 **877 条**：WebShop 381、ALFWorld 248、DBBench 248。Qwen3.5-9B engineer 做 2 epochs full-parameter SFT，context length 32,768。

SFT 的作用主要是建立可执行 patch 的语法和结构先验，但它模仿的是经过 teacher + 非退化过滤后的单个候选，尚未直接优化不同补丁之间的相对效用。

### 4.2 Online GRPO：再学习什么补丁真正有用

对每个 failure packet，从当前 engineer 采样 $K=8$ 个候选 patch。每个候选独立安装，并让冻结目标 Agent 重跑完整 batch。组内 reward 标准化为：

$$
\hat A_k=\frac{r_k-\mu_B}{\sigma_B},
$$

再使用 clipped GRPO objective 更新 engineer。sequence-level advantage 共享给该回答的所有 token；论文没有 format-validity bonus，也没有显式 KL loss。在线 RL 约使用 **1,500 个 failure packet**，SFT 与 RL 的 task split 分离。

这一步的本质不是让模型学“补丁看起来像什么”，而是学习一个排序偏好：面对同一组失败证据，哪类 intervention 相比同组候选能带来更高的真实执行收益。

## 5. 实验怎么读？

### 5.1 设置

| Benchmark | 测试任务数 | 能力侧重 | reward |
|---|---:|---|---|
| WebShop | 500 | 搜索、属性约束、购买动作 | shaped reward + success |
| ALFWorld | 500 | 长程具身操作、状态跟踪、恢复 | binary success |
| DBBench | 300 | schema 探索、SQL、结果验证 | binary success |

主目标 Agent 为 Qwen3.5-9B，目标 temperature 为 0。所有非 Reflection 行都是 success@1；Reflection 是累计两次 episode 的 success@2，不能直接排名。

### 5.2 主结果

| 方法 | ALFWorld | WebShop | DBBench | 三项平均 |
|---|---:|---:|---:|---:|
| Default harness | 40.6 | 31.2 | 61.0 | 44.3 |
| ReAct | 43.4 | 37.4 | 61.7 | 47.5 |
| Self-Refine | 39.0 | 29.0 | 57.3 | 41.8 |
| 最佳 fixed frontier editor（GLM-5.2） | 45.0 | 36.0 | 65.3 | 48.8 |
| Supervised-only engineer | 39.4 | 38.6 | 61.3 | 46.4 |
| **Harness-R1** | **53.2** | **42.2** | **65.3** | **53.6** |
| Agent SFT | 71.2 | 42.6 | 63.7 | 59.2 |
| **Agent SFT + target-specific Harness-R1** | **84.0** | **43.0** | **65.7** | **64.2** |

最关键的比较有三组：

- `44.3 -> 53.6`：harness editing 对冻结 Agent 有效；
- `46.4 -> 53.6`：在线 outcome training 比只模仿 teacher 高 **7.2 个表中百分点**（正文写 7.1，来自未四舍五入值）；
- `59.2 -> 64.2`：Agent 权重训练与 harness 训练存在互补性。

但最后一项不等于同一个 engineer 自动适应变强后的 Agent。论文明确使用了为 SFT 后目标重新训练的 **target-specific engineer**，所以证据支持“可顺序重训、两种优化互补”，尚不支持无重训的持续共进化。

### 5.3 跨目标模型泛化

训练出的 editing policy 用于 20 个训练时未见的 target configuration；每个 target 提供自己的失败轨迹，engineer 为它新生成 patch，并不是复用同一个固定 patch。

- target-level 平均全部为正，平均提升 **+7.06**；
- 21 个目标（含训练目标）× 3 个 benchmark 的 63 个组合中，56 个提升、4 个不变、3 个退化；
- 三个退化均不超过 2.0 个百分点；
- 按 benchmark 聚合：WebShop +4.15、ALFWorld +9.63、DBBench +7.37。

因此这里证明的是 **editor-policy transfer**：同一个编辑策略能根据新目标自身的失败证据生成新 patch。它不是 patch transfer，也不是 zero-evidence transfer。

### 5.4 Held-out task 泛化

每个 benchmark、每个 seed 只向 engineer 展示同一目标 Agent 的 10 个失败样本，然后把生成的单个 benchmark patch 应用于其余任务。合计 1,270 个 held-out tasks，3 个 matched seeds：

| Editor | Held-out success change |
|---|---:|
| Harness-R1 | **+8.9 ± 1.5** |
| Qwen3.5-397B | -4.3 ± 2.5 |
| DeepSeek-V4-Pro | -0.4 ± 3.6 |

这是对“补丁只是记住同批任务”的重要补充，而且 Harness-R1 三个 seed 均为正。不过只有 3 个 seed，误差条是 sample standard deviation，不是置信区间；论文也没有报告显著性检验。

### 5.5 哪个 hook 最重要？

固定生成好的 patch，再逐个禁用 lifecycle position：

| 配置 | 平均成功率 | 相对 full patch |
|---|---:|---:|
| No intervention | 44.2 | -8.9 |
| Full patch | **53.1** | - |
| w/o pre-decision | 52.5 | -0.6 |
| w/o episode start | 52.2 | -0.9 |
| w/o post-feedback | 49.8 | -3.3 |
| w/o pre-action | 49.2 | -3.9 |

主要收益来自 **action mediation** 和 **failure recovery**，而不是仅在 prompt 前面再塞一段静态提示。这与论文的研究定位一致：真正有价值的是 runtime control，而不只是 prompt optimization。

不能把这张图解释成普适 hook 排名：WebShop patch 只包含 pre-action edit，不同 hook 还可能协同，因此 leave-one-out 的损失不可相加。

## 6. 这篇论文真正的新意

### 6.1 学习的是“编辑器”，不是一次编辑结果

以往系统常把 outcome 用于搜索、筛选、回归测试或 promote patch；Harness-R1 用 outcome 直接更新 proposer/editor 权重。这是最清楚的概念贡献。

### 6.2 reward 穿过了可执行系统

监督链条是：

$$
\text{failure evidence}
\rightarrow \text{code patch}
\rightarrow \text{runtime behavior}
\rightarrow \text{task outcome}
\rightarrow \text{editor update}.
$$

这比“让 judge 评价补丁文字是否合理”更 grounded，因为 credit 最终来自目标 Agent 的真实行为变化。

### 6.3 把模型与外围控制面分开优化

目标 Agent 负责决策，engineer 负责改进决策环境。两者可以采用不同训练数据和 reward，并且前者冻结时仍能改善系统。这给“Agent 系统能力如何分配到 weights / context / tools / middleware / recovery”提供了更明确的工程分层。

### 6.4 受限 action space 是优点，也是边界

Typed hook interface 让补丁可验证、可执行、可归因，显著降低任意代码生成风险。与此同时，系统已由人类提前决定了四个干预位置、context schema 和 structured effect；学习器主要学习**何时以及如何组合这些已有机制**，没有学习新的 runtime architecture。

## 7. 局限与需要谨慎的地方

### 7.1 训练目标是 same-batch transductive utility

Eq. (1) 在产生 failure packet 的同一任务 batch 上计算 reward。这可以控制任务组成，也能惩罚同批回归，但容易把 engineer 推向 batch-specific rule。Held-out 实验证明了部分泛化，却没有改变训练目标本身。

更稳健的 reward 应同时包含：

$$
r = \Delta_{\text{train batch}}
+ \lambda\Delta_{\text{held-out regression}}
- \alpha C_{\text{latency}}
- \beta C_{\text{tokens}}
- \gamma C_{\text{safety}}.
$$

### 7.2 “Lifecycle-wide”不等于任意 runtime 编辑

Patch 只能使用四个预定义 hook，不能 import、不能维护 global state，并且 effect 由宿主解释。它比 prompt editing 广，但距离修改 tool implementation、memory store、retriever、scheduler、权限系统和多 Agent topology 仍有距离。

### 7.3 训练和评测成本没有进入主目标

一次 RL prompt 采样 8 个候选，每个有效候选都要让目标 Agent 重跑整个 task batch。在线训练约 1,500 个 failure packet，且三种训练均使用单节点 8×H800。论文没有报告总 GPU hours、环境调用数、token 成本、训练 wall-clock 或 inference latency。若 patch 每一步都运行 hook，还可能以额外 token 和控制逻辑换取成功率。

### 7.4 安全性评估不足

Engineer 生成会介入 action execution 的代码，尤其 `rewrite_action` / `force_action` 属于高权限控制面。当前验证主要关注可安装性、完成评测和任务 reward，没有专门报告：

- prompt injection 对 failure packet / generated code 的影响；
- privilege boundary 与 sandbox escape；
- 数据投毒导致共享 engineer 学到恶意 intervention；
- patch 的静态分析、资源限制和审计策略；
- reward hacking，例如通过 block/force 绕过任务语义。

### 7.5 统计报告仍有限

主表给出单个成功率，没有跨 rollout seed 的方差或置信区间。目标 Agent temperature=0 能减少采样噪声，但环境、工具和执行仍可能有变动。Held-out task 实验提供了 3 个 seed 的标准差，是更强证据，但样本仍少。

### 7.6 “Co-evolution”是方向，不是已完成的算法

论文只做了：vanilla Agent → Agent SFT → 为新 Agent 训练新的 engineer。没有交替多轮更新、没有分析收敛或相互适应导致的非平稳性，也没有展示旧 engineer 在 Agent 更新后能否保持有效。更准确的措辞是 **sequential complementarity**。

### 7.7 Benchmark-specific runtime scaffolding 较重

每个环境都预定义丰富 context、state 和 predicate，例如 WebShop 的 `product_price_over_budget`、DBBench 的 `premature/empty commit`。这些 hand-engineered observables 已经编码了相当一部分 failure taxonomy。跨模型泛化很强，但跨全新环境是否仍成立，需要重新设计 host runtime 和 schema 后验证。

## 8. 与相近路线的关系

### 与 [[2026-TTHE-Test-Time-Harness-Evolution]]

TTHE 更关注测试时从轨迹中诊断并演化 harness；Harness-R1 的关键推进是把 editor 本身变成可 post-train 的 policy，并由重跑 outcome 更新权重。前者偏在线搜索/演化流程，后者偏 learned optimizer。

### 与 [[2026-SkillClaw]]

SkillClaw 把多用户轨迹压缩成可版本化、可验证的共享 skill，学习对象是外部 Markdown/procedural artifact；Harness-R1 生成的是直接插入 runtime lifecycle 的 executable hook，并训练生成这些 hook 的模型参数。

```text
SkillClaw:  trajectories -> skill evidence -> edit shared skill -> validate/promote
Harness-R1: failures -> engineer policy -> executable hook -> rerun reward -> update policy
```

两者可以组合：SkillClaw 管理长期、可审计的程序性知识；Harness-R1 学习何时把知识转化为 pre-action guardrail 或 post-feedback recovery。组合时必须增加版本账本、held-out regression suite 和权限隔离。

## 9. 对研究和工程的启发

1. **Harness 应被视作独立参数空间。** 不必把所有失败都塞回模型权重；接口校验、状态恢复和动作约束往往更适合外围 runtime。
2. **编辑器需要行为级反馈。** Code compiles、LLM judge 认可、静态规则通过都只是必要条件；最终仍要做 patched-versus-baseline execution。
3. **训练时应保留成功任务。** 只重跑失败任务会奖励局部修补，无法发现对原有能力的破坏。
4. **优先学习 action mediation 和 recovery。** 消融显示，它们比静态 context 注入贡献更大，适合成为下一代 Agent runtime 的一等扩展点。
5. **需要双层回归集。** 一层覆盖触发该 patch 的局部失败，另一层覆盖历史成功任务、其他 task family、安全约束和资源成本。
6. **长期系统要从 patch 生成走向 patch 治理。** 包括 provenance、版本、静态分析、canary、回滚、过期条件和用户级隔离；论文目前主要完成了生成与结果学习。

## 10. 我会如何扩展这项工作

### 实验 A：真正的多轮 co-evolution

交替执行 `Agent update -> Engineer update -> held-out evaluation` 5-10 轮，固定总计算量，与只训练 Agent、只训练 Engineer、同时联合训练比较。观察收益是否叠加、是否振荡，以及旧 patch 是否随 actor 改变而失效。

### 实验 B：跨环境迁移

在三个 benchmark 上训练 engineer，直接转移到新的 tool-use 环境，只提供统一的 typed runtime API，不提供 benchmark-specific predicates；然后逐步加入 schema 与少量 failure packet，区分 policy transfer 与人工 scaffolding 的贡献。

### 实验 C：成本与 Pareto reward

把成功率、平均步骤、target token、hook latency、阻止/重写次数共同纳入 reward，画出 utility-cost Pareto frontier，检验当前增益是否来自更频繁 reprompt 或更长轨迹。

### 实验 D：安全红队

在 failure trajectory 中注入恶意 observation、伪造 tool error 和跨用户污染，测试 engineer 是否生成越权 `force_action`、泄漏数据或持久化后门，并比较 typed DSL、sandbox、static analysis 和 human approval 的防御效果。

## 11. 最终评价

| 维度 | 评价 |
|---|---|
| 问题重要性 | 高。Agent 的大量能力与失败实际位于模型外围 runtime |
| 核心新意 | 强。首次明确用下游执行收益 post-train 独立 harness editor |
| 方法完整性 | 强。SFT 冷启动、受限 patch API、同批重跑、在线 GRPO 构成闭环 |
| 实验证据 | 较强。三环境、强 fixed editor、SFT-only、跨目标与 held-out 泛化、hook 消融 |
| 外部有效性 | 中等。环境与 hook schema 高度定制，尚未覆盖真实异构 Agent codebase |
| 安全与成本 | 偏弱。高权限 executable patch 的治理和完整计算成本未充分量化 |
| 结论可信范围 | 支持 outcome-trained typed runtime editing；不支持任意代码维护或已实现长期 co-evolution |

总体评分：**8.5/10**。它最值得带走的不是某个 GRPO 公式，而是一个更一般的研究范式：

> 不只优化 Agent 生成的答案，也不只搜索一个更好的 harness；应当训练一个能从真实系统结果中学习“如何改系统”的编辑策略，同时让每次改动经过可执行验证、回归约束和权限治理。

## 引用

Shao, S., Zhang, K., Li, Q., et al. (2026). *Harness-R1: Learning to Edit Executable Runtime Harnesses from Agent Failure Trajectories*. arXiv:2608.02276v1.
