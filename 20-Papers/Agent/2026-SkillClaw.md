---
title: "SkillClaw: Let Skills Evolve Collectively with Agentic Evolver"
authors:
  - Ziyu Ma
  - Shidong Yang
  - Yuxiang Ji
  - Xucong Wang
  - Yong Wang
  - Yiming Hu
  - Tongwen Huang
  - Xiangxiang Chu
year: 2026
venue: arXiv preprint
arxiv: 2604.08377v1
tags:
  - paper
  - agent
  - agentic-skills
  - skill-evolution
  - collective-learning
  - procedural-memory
  - openclaw
  - agentic-evolver
status: read
rating: 7.5/10
date-read: 2026-08-02
url: https://arxiv.org/abs/2604.08377
code: https://github.com/AMAP-ML/SkillClaw
pdf: "C:/Users/huawei/Zotero/storage/LUAP93RB/Ma et al. - 2026 - SkillClaw Let Skills Evolve Collectively with Agentic Evolver.pdf"
related:
  - "[[2026-OpenClaw-RL]]"
---

# SkillClaw：让多用户经验沉淀为可验证、可共享的 Agent Skills

## TL;DR

SkillClaw 解决的问题是：Agent 在真实使用中反复踩坑、调试并找到更好流程，但这些经验通常只留在一次 session 里；另一个用户第二天遇到同类任务时，仍会从头犯相同的错。

它不更新模型权重，而是把多用户、跨时间的完整交互轨迹聚合起来，由一个 **agentic evolver** 自动诊断重复出现的成功模式与失败根因，然后：

- 改进已有 skill；
- 创建缺失的新 skill；
- 优化 skill description/触发边界；
- 或在证据不足时保持不变。

候选 skill 不会直接推送给所有用户。系统在夜间的真实执行环境中比较旧版与候选版，只有验证更好的版本才进入共享 SkillHub，并在第二天同步给所有 Agent：

```text
多用户白天使用
  -> 收集完整 action-feedback trajectories
  -> 按 referenced skill 聚合证据
  -> evolver 诊断 skill / agent / environment 根因
  -> refine / create / optimize-description / skip
  -> 夜间真实执行验证
  -> accept 或 reject
  -> 更新共享 skill pool
  -> 次日同步给所有用户
```

一句话抓住论文：

> SkillClaw 把单用户 session 中偶然发现的 procedural knowledge，压缩成可审计、可回滚、经过执行验证的共享 skill，让一个用户发现的解法可以惠及整个 Agent 群体。

这篇论文最有价值的地方不是证明了“skills 会无限自主进化”，而是给出了一个较完整的 **经验到程序化知识的治理闭环**：保留因果轨迹、按 skill 归因、成功与失败联合分析、保守编辑、版本历史、执行验证、best-so-far 部署。

我的判断：方法非常符合真实 Agent 产品的工程结构，尤其适合 API 细节、文件路径、工具顺序、环境 preflight 和约束核验这类 procedural failure；但论文的实验证据仍是小规模 work in progress。它用同一个 Qwen3-Max 同时承担执行、演化和验证，8 个模拟用户只跑 6 轮，60 个任务的 benchmark 仅报告 4/6 类，也没有随机种子、误差条、独立 validator 或严格的跨任务 held-out 验证。因此目前更准确的结论是“一个有效的技能维护原型”，而不是已经证明了大规模 collective intelligence。

## 1. 为什么需要 collective skill evolution？

现代 Claw-style Agent 的能力不只存在于模型权重中，还存在于 `SKILL.md` 一类外部程序性知识中。一个 skill 通常编码：

- 什么时候使用某组工具；
- API endpoint、port、payload 与文件格式；
- 多步工作流的正确顺序；
- 哪些状态必须先检查；
- 出错时如何恢复；
- 输出必须满足哪些约束。

当前 skill 生态通常是静态的：用户从 SkillHub 安装 skill，部署后即使 Agent 通过 trial-and-error 找到了更可靠的流程，也不会反写到 skill。由此产生三个问题：

1. **经验不持久**：同一个 Agent 在新 session 里重复犯错；
2. **经验不共享**：不同用户分别发现相同的 API 或环境陷阱；
3. **skill 边界不被真实使用检验**：作者写下的说明无法覆盖所有设备、任务和上下文。

Memory 能保存过去轨迹，但 instance-specific 记录不一定能抽象为通用行为。SkillClaw 的定位是把重复的交互证据进一步压缩成 reusable procedure。

形式上，共享 skill 集为：

$$
\mathcal{S}=\{s_1,\ldots,s_M\},
$$

多用户轨迹集合为 $\mathcal{T}=\{\tau_i\}$，目标是学习一个外部演化操作：

$$
\mathcal{S}'=\Phi(\mathcal{S},\mathcal{T}),
$$

使某次交互中发现的改进可以改变未来所有用户可访问的 skill repository。

这里的学习对象是 Markdown/程序化 artifact，而不是 neural parameters。

## 2. 系统闭环

SkillClaw 采用中心化 evolution、分布式使用的架构（Figure 1, p. 2）：

```text
Independent Claw agents
  | use shared skill catalogue
  | produce sessions on different tasks/environments
  v
Session aggregation
  | preserve prompt -> action -> feedback -> response
  | attach referenced skills, tool errors, quality estimates
  v
Skill-specific evidence groups
  | G(s): sessions that referenced skill s
  | G(empty): sessions that referenced no skill
  v
Agentic evolver
  | evidence analysis
  | root-cause attribution
  | candidate skill mutation
  v
Execution validator in idle user environments
  | old skill vs candidate skill
  v
Shared SkillHub / best-skill pool
  | synchronize accepted versions
  v
Independent Claw agents
```

中央化的好处是不同用户对同一个 skill 的成功和失败可以放在一起比较；风险是用户数据、共享更新权和 skill supply chain 都集中到一个高价值控制面。

## 3. 从原始 session 到 shared evidence

### 3.1 为什么必须保存中间轨迹？

最终回答不足以诊断多数 skill failure。以下错误可能产生看似合理的文字回复，却没有真正完成任务：

- tool argument 格式错误；
- 调错 API port；
- 没有读取所需文件；
- 工具调用顺序颠倒；
- 输出写错路径；
- 忽略环境返回的失败；
- 满足部分条件却声称全部满足。

因此 SkillClaw 保存：

$$
\text{prompt}\rightarrow\text{action}\rightarrow\text{feedback}
\rightarrow\cdots\rightarrow\text{agent response}.
$$

附录实现中的预处理 session 还包括：

| 字段 | 作用 |
|---|---|
| `session_id`, `task_id`, `num_turns` | 标识与规模 |
| `aggregate.mean_score` | rollout 级结果 |
| `success_count`, `fail_count`, `stability` | 区分稳定成功、稳定失败与不稳定 |
| `_skills_referenced` | 将证据路由到对应 skill |
| `_avg_prm` | 中间过程的粗质量估计 |
| `_has_tool_errors` | 快速定位执行故障 |
| `_trajectory` | 逐步动作、工具参数、结果和得分 |
| `_summary` | 供 evolver 快速理解 session 的压缩摘要 |

这一 schema 的思想比具体字段更重要：**必须同时保留 outcome、process、provenance 和 skill attribution。**

### 3.2 按 referenced skill 分组

若轨迹 $\tau_i$ 引用了 skill 集 $K_i$，则：

$$
G(s)=\{\tau_i\mid s\in K_i\}.
$$

没有引用任何 skill 的 session 进入 $G(\varnothing)$。

- $G(s)$ 用于判断已有 skill 在哪些环境成功、哪些环境失败；
- $G(\varnothing)$ 用于发现没有现有 skill 覆盖的重复 procedure；
- 同一 skill 下的成功/失败对照，为 evolver 提供一种 observational “natural ablation”。

但论文把它称为 natural ablation 略显乐观：不同用户 session 的任务、上下文、工具状态和 Agent 随机性都可能不同，skill 并不是唯一变化因素。这是一种有用的对比证据，不是随机对照实验。

## 4. Agentic evolver 如何改 skill？

正文 Algorithm 1 把操作概括为 `refine / create / skip`。附录真实 prompt 更细，允许四种动作：

| 动作 | 适用证据 | 修改对象 |
|---|---|---|
| `improve_skill` | skill 内容缺失、过期、误导或不清晰 | `SKILL.md` body，必要时 description |
| `optimize_description` | skill 本体正确，但触发条件导致误匹配 | frontmatter description |
| `create_skill` | 出现独立、重复、可教学且现有库未覆盖的 procedure | 新 skill |
| `skip` | 证据弱、失败非 skill 所致，或现有 skill 已足够好 | 不变 |

### 4.1 成功与失败必须联合阅读

论文的核心编辑原则是：

- 成功 session 定义必须保留的 invariant；
- 失败 session 定义需要修正的 target；
- 不能只看失败后重写整个 skill，否则容易修好一个 corner case、破坏已有主路径。

这对应附录中的 conservative editing：当前 skill 是 source of truth，默认做 targeted edit，保留原结构、术语、API contract 和已被成功轨迹支持的内容。

### 4.2 先判断失败归属，再决定是否编辑

附录要求 evolver 区分三种根因：

| 根因 | 示例 | 正确动作 |
|---|---|---|
| Skill problem | endpoint 错、缺少关键步骤、description 误触发 | 修改 skill |
| Agent problem | 没读 skill、错误使用 subagent、context overflow | 不要把 runtime 问题塞进 skill |
| Environment problem | API 不稳定、网络失败、Docker 特性 | 重复出现时只加简短环境说明 |

一个很重要的 anti-pattern：如果 skill 已经包含正确 API 信息，而 Agent 没有遵循它，不能删掉正确说明，再写成“去读源码”。否则 skill evolution 会奖励 Agent 的不服从，把精炼知识退化回重复探索。

### 4.3 创建新 skill 的门槛

新 skill 必须：

- 与当前 skill 有不同目的；
- 压缩领域或环境特定知识，而非泛泛 best practice；
- 有明确的 use-when 和 `NOT for` 触发边界；
- 对应重复、可复用、可教学的 procedure；
- 在 manifest 中不存在同名或同功能 skill。

这个约束防止 skill library 变成大量 session postmortem 的堆积。

## 5. Evolution history：为什么版本账本很重要？

论文附录给出了比正文更成熟的 history 机制。每次修改已有 skill 前，evolver 必须阅读 `history/` 中全部旧版本和 evidence 文件：

```text
skills/<skill-name>/
  SKILL.md
  history/
    v1.md
    v1_evidence.md
    v2.md
    v2_evidence.md
```

修改时：

1. 把当前 `SKILL.md` 原样保存为 `v<N>.md`；
2. 写 `v<N>_evidence.md`，记录 session IDs、task IDs、PRM/ORM、工具错误、历史比较与 edit plan；
3. 再修改当前版本；
4. 未来 evolver 必须先读完整历史，避免重复或撤销旧改进。

Evidence ledger 应回答：

- 为什么现在需要改？
- 哪些成功/失败 session 支持这个判断？
- 以前改过什么，后来表现如何？
- 哪些内容故意保留，哪些内容明确修改？
- 哪些不确定性留给下一轮观察？

这是 SkillClaw 相比权重在线学习非常实际的优势：外部 skill 的变化可读、可 diff、可审核、可回滚，也更容易响应用户删除请求。

## 6. 验证与 best-so-far 部署

白天收集证据，夜间在空闲用户环境中验证。对旧 skill $s$ 与候选 $s'$：

1. 从白天数据选择相关任务；
2. 在相同环境、完整工具链下分别执行；
3. 模型比较总体任务成功与执行稳定性；
4. $s'$ 更好则 `Accept`，否则 `Reject`；
5. 只把 accepted update 合并进共享 skill pool。

论文把这称为 monotonic deployment：用户始终使用上一次验证过的 best pool。

更准确地说，这是 **validation-set monotonicity**，而不是全分布单调性。候选在选定任务上更好，并不保证：

- 对未见任务不退化；
- 对其他引用同一 skill 的任务不退化；
- 对未来变化后的 API/环境仍更好；
- 对不同用户偏好无负面影响；
- validator 的比较没有随机噪声。

真正稳健的发布门应同时包含 candidate-targeted tasks、历史回归集、跨 skill integration tests、安全测试与多次独立 rollout。

## 7. 实验协议

### 7.1 WildClawBench

WildClawBench 有 60 个复杂任务、6 个领域：

| 类别 | 示例 | 主要挑战 |
|---|---|---|
| Productivity Flow | arXiv 分类、排程、SCP | 多步 pipeline |
| Code Intelligence | debugging、puzzle | 执行正确性 |
| Social Interaction | negotiation、chat analysis | 多轮推理 |
| Search & Retrieval | 学术搜索、冲突消解 | API 使用 |
| Creative Synthesis | 视频笔记、海报生成 | 多模态生成 |
| Safety & Alignment | prompt injection、泄漏检测 | 约束满足 |

环境为带工具的完整 Linux container，输入覆盖文本、代码、图片和视频；每任务聚合 3-27 个评价指标，关键错误直接记零，任务长度约 15-50 步，并依赖外部 API 与模型下载。

### 7.2 Day-night simulation

- 6 days / 6 evolution rounds；
- 8 个并发模拟用户；
- 白天使用当前 best skill pool；
- 夜间生成候选并在环境中验证；
- 执行 Agent、skill evolver 和 validator 全部由 **Qwen3-Max** 驱动；
- Day 1 是初始 skill baseline；
- 只考虑当天被触发且被判断“可能改进”的 skill；
- 论文当前只报告 4 个代表类别，Productivity Flow 和 Code Intelligence 留待未来版本。

这不是 6 次独立实验，而是一个带状态的连续部署轨迹；因此 Day 2-6 高度相关，不能当作 5 个重复样本。

## 8. 主结果怎么读

### 8.1 用户侧 best-pool 结果

Table 3（p. 7）：

| 类别 | Day 1 | 最终 Day 6 | 绝对提升 | 相对提升 | 首次达到最终值 |
|---|---:|---:|---:|---:|---:|
| Social Interaction | 54.01% | 60.34% | +6.33 | +11.72% | Day 2 |
| Search & Retrieval | 22.73% | 34.55% | +11.82 | +52.00% | Day 4 |
| Creative Synthesis | 11.57% | 21.80% | +10.23 | +88.41% | Day 2 |
| Safety & Alignment | 24.00% | 32.00% | +8.00 | +33.33% | Day 5 |

四类都提升，但要注意：

- 相对增益大部分来自很低的 baseline，例如 Creative 的 +88.41% 对应绝对 +10.23；
- Social 和 Creative 只在第一次有效更新后提升，之后 4-5 天完全平台化；
- Search 分两阶段提升；
- Safety 的部署轨迹到 Day 5 才反映更高得分。

论文没有给每个类别的任务数、方差、随机种子或显著性检验，因此无法判断 +6.33 是否超出 Agent rollout 噪声。

### 8.2 Accepted/rejected updates 比最终分数更有信息

#### Social Interaction

只有 Night 1 的 `03_task6` 被接受：把跨部门 Slack summary 从描述性说明改为严格顺序的 workflow，加入关键词过滤、财务优先级、变化检测和 COO 联系确认。Night 3 的会议协调 skill、Night 6 的 feasibility grounding 均未进入 best pool。

#### Search & Retrieval

Night 1 接受 `validate-file-existence`；Night 3 是同一 best pool 的复测确认。更复杂的路径恢复、强化 multimodal preflight、约束搜索规划都被拒绝。最有效的改进不是高级 research planning，而是“读取前先确认文件真的存在”。

#### Creative Synthesis

只有 Night 1 的 `validate-tmp-workspace-inputs` 被接受。后续统一多模态 pipeline、图像分类、视频/PDF/海报工作流全部未超过早期 best pool。说明瓶颈主要在 workspace 和输入环境，而非生成能力。

#### Safety & Alignment

Night 1-3 接受 Git auth failure fallback、文件名统一和正确 clone 目录等改进；Night 4 复测确认 best-so-far；Night 5-6 继续添加 push hang、identity config 等规则却没有收益。

这些轨迹传达出一个重要规律：**外部 skill 最擅长修复确定、可执行、环境特定的 procedure；加入更多文本并不自动产生更多能力。**

### 8.3 Controlled validation

Skill Evolve Lite 在三个自定义 query 上做单轮对照（Table 8, p. 10）：

| Query | Baseline | Post-Evolve | Gain |
|---|---:|---:|---:|
| basic extraction | 21.7% | 69.6% | +47.8 |
| deadline parsing | 41.1% | 48.0% | +6.9 |
| save report | 28.3% | 100.0% | +71.7 |
| Average | 30.4% | 72.5% | +42.1 |

`save report` 的输出路径/格式和 `basic extraction` 的重复流程可被 skill 精确编码，因此收益很大；`deadline parsing` 更依赖语义推理，收益仅 +6.9。这是全文最清楚的机制证据：skill evolution 主要改善 **procedural knowledge gap**，不能替代模型本身的 reasoning capability。

不过只有 3 个 custom queries，没有任务数量、置信区间和独立复现细节，+42.1 不应外推为一般 Agent 平均收益。

## 9. Case studies 揭示了什么？

### 9.1 Slack message analysis

原 workflow 全量读取消息、统一处理，API port 错误靠 trial-and-error。演化后：

1. 先扫描 preview 找候选；
2. 只对相关候选读取全文；
3. 提取 actionable items；
4. 把验证过的正确 API 配置写进 skill。

这里同时发生 task decomposition、selective retrieval 和 tool error correction。

### 9.2 ICCV 论文 affiliation 分析

原 Agent 只要 affiliation block 出现大学名就算匹配，容易把非 first affiliation 算进去。演化后明确 first affiliation 的结构定义，用 OpenAccess record 对齐 title，并对 PDF parsing 的歧义样本 targeted re-check。

核心不是“搜更多”，而是把模糊的评价概念编译成可执行判据。

### 9.3 SAM3 不完整执行环境

原 Agent 假设输入文件、输出路径和 CUDA 条件都存在。演化后的 preflight：

- 区分缺失的输入与可新建的输出目录；
- 在附近查找 packaged asset；
- 检查 task-specific ground truth 或示例；
- 根据 CPU/CUDA 实际条件调整执行。

但案例中 monkey patch CUDA 组件也说明自动演化可能写入危险环境 hack。真实系统必须区分“可移植 procedure”与“只对当前 benchmark 生效的 workaround”。

### 9.4 多条件手机选择

原 Agent 找到一个看似合理的候选就提前停止，并把 partial match 当 full match。演化后对每个约束使用官方来源逐项核验，联合判断所有条件；没有完全匹配时明确报告，并给出 partial-match breakdown。

这是将自然语言要求编译为 constraint table 和 calibrated decision policy。

## 10. 论文真正证明了什么？

### 证据较支持

- 完整 action-feedback trace 能暴露仅看最终回答无法发现的 procedural failure；
- 按 referenced skill 聚合多条 session，有助于发现重复错误与成功 invariant；
- Agent 可以从轨迹中提出实际可执行的 skill 修改；
- 验证门可以拒绝大量没有收益的文本膨胀和 speculative update；
- 文件、路径、API、工具顺序、约束核验等程序性问题尤其适合 skill 化；
- 在报告的四类 WildClawBench 轨迹中，best skill pool 相比 Day 1 有一致的最终提升；
- 外部 skill 适合作为模型之外可版本化、可共享的 collective procedural memory。

### 尚未被充分证明

- **大规模多用户 collective learning**：只有 8 个模拟用户、6 轮；
- **跨用户而非单用户累积的必要性**：没有 1-user vs 8-user、isolated vs shared 的核心消融；
- **跨时间持续增长**：两类任务 Day 2 后即平台化；
- **真正独立的泛化**：validation tasks 来自白天 interaction data，可能对同一任务模式过拟合；
- **单调不退化**：只在选择的验证任务上 best-so-far；
- **全 benchmark 有效**：6 类中只报告 4 类；
- **模型无关性**：只有 Qwen3-Max；
- **validator 可靠性**：执行、evolution、比较均由同一模型家族完成；
- **成本可接受**：承认需要真实工具执行和额外 token，但没有总成本、延迟或性价比数字；
- **安全可部署**：没有恶意 session、poisoned skill、跨用户隐私或 supply-chain attack 实验。

## 11. 方法与实验中的关键问题

### 11.1 缺少 collective 的决定性消融

论文的标题主张“collectively”，但没有比较：

- 每用户独立 skill evolution；
- 只聚合单用户多 session；
- 聚合 2/4/8 用户；
- 共享全部原始轨迹 vs 只共享抽象 evidence；
- 同质用户 vs 冲突用户。

因此当前结果能证明 skill evolution 有效，却不能单独量化 cross-user aggregation 带来的收益。

### 11.2 同一个 Qwen3-Max 同时当 actor、evolver、validator

三种角色共享偏差：actor 看不出的错误，evolver 和 validator 也可能看不出；evolver 写出的符合自身偏好风格的 skill，也更容易被同模型 validator 接受。需要独立的 outcome checker、异构 validator 或人工 blind audit。

### 11.3 验证任务复用带来过拟合风险

候选由当天 session 生成，又在从当天数据选择的相关任务上验证。即使执行环境真实，这仍接近 within-distribution repair。更强协议应分为：

```text
evolution evidence set
validation gate set
future-user held-out set
historical regression set
```

四者不能完全重合。

### 11.4 Skill attribution 不是因果 attribution

某轨迹引用 skill $s$ 并失败，不代表 $s$ 导致失败。可能是：

- Agent 根本没遵循 skill；
- 还引用了其他 skill；
- environment 独立失败；
- task 超出 backbone 能力；
- skill routing 本身错了。

附录用 LLM root-cause reasoning 缓解此问题，但没有定量 attribution accuracy。

### 11.5 Skill library 会出现膨胀、冲突和路由退化

长期创建 skill 可能导致：

- 相似 skills 重复；
- description 互相重叠；
- 旧 skill 与新 skill 的 API 假设冲突；
- catalogue 太长，占用 context；
- Agent 选错 skill；
- local fix 破坏跨任务通用性。

需要定期做 deduplication、dependency graph、conflict test、deprecation 和 archive，而不仅是 append/refine。

### 11.6 共享 skill 是新的供应链攻击面

恶意用户可以构造轨迹诱导 evolver 写入：

- 错误 endpoint 或参数；
- 数据外传步骤；
- prompt injection；
- 隐蔽的特定用户触发条件；
- benchmark-specific shortcut；
- 危险 shell command。

一旦通过中央同步，单用户污染会放大为全体用户风险。越是“collective”，越需要 provenance、权限、签名、sandbox 和 staged rollout。

## 12. 与 OpenClaw-RL 的关系

[[2026-OpenClaw-RL]] 和 SkillClaw 都把部署后的交互变成持续改进信号，但学习载体不同：

| 维度 | SkillClaw | OpenClaw-RL |
|---|---|---|
| 更新对象 | 外部 `SKILL.md` / skill repository | policy model 权重 |
| 监督来源 | 多用户完整 session 与执行结果 | action 后的 next-state signal |
| 核心变换 | trajectory -> reusable procedure | next state -> reward + token hint |
| 更新节奏 | day-night 批量演化和验证 | 异步在线训练 |
| 可解释性 | 高，可阅读 diff 与 evidence | 低，行为压进权重 |
| 回滚/删除 | 版本回退相对容易 | 难，需要 checkpoint/unlearning |
| 推理开销 | skill catalogue 与内容占 context | 更新后通常无额外 prompt |
| 训练资源 | LLM evolver + 环境重放验证 | PRM + policy training GPUs |
| 擅长问题 | API、路径、workflow、约束、工具细节 | 风格偏好、广义行为分布、隐式策略 |
| 主要风险 | skill poisoning、冲突、library bloat | feedback poisoning、遗忘、隐私进权重 |

二者不应被看作竞争方案。更合理的分层系统是：

```text
interaction / next state
  -> immediate episodic memory
  -> repeated verifiable procedure -> shared skill evolution
  -> stable high-frequency behavior -> user adapter / RL
  -> broadly validated capability -> cautiously merge into base policy
```

SkillClaw 适合先作为低风险、可回滚的知识沉淀层；只有跨任务长期稳定、难以用显式 procedure 表达的行为，才值得进入权重。

## 13. 更可信的生产架构

```text
Raw sessions
  -> consent + PII/secret redaction
  -> signed provenance and tenant scope
  -> session summarization with raw-trace link
  -> skill routing / causal attribution
  -> evidence threshold and cross-user diversity check
  -> candidate generation in isolated workspace
  -> static policy and security scan
  -> targeted execution tests
  -> historical regression suite
  -> held-out and adversarial tests
  -> independent validator ensemble
  -> tenant canary rollout
  -> global rollout or rollback
```

必须补充的治理机制：

1. **最小证据阈值**：一个用户的一次失败默认不能修改 global skill；
2. **跨用户一致性**：候选应由多个独立 tenant 的重复模式支持；
3. **作用域控制**：user-local、organization、public 三层 skill 不自动互相提升；
4. **敏感信息检查**：API key、路径、姓名、内部域名不得写入公共 skill；
5. **语义版本与依赖**：记录 skill 依赖的工具/API 版本；
6. **安全能力边界**：evolver 只允许修改 sandbox copy，并限制命令和网络；
7. **多维发布门**：正确率、稳定性、成本、延迟、安全、context 长度共同判定；
8. **负向迁移监控**：candidate-targeted task 提升不够，必须检查其他调用方是否退化；
9. **自动退役**：过时、重复和长期未触发的 skill 应归档；
10. **用户可见审计**：说明某个 skill 为什么改变、来自哪些脱敏证据、如何回滚。

## 14. 值得迁移的研究启发

### 启发 1：把 skill evolution 看作“持续编译”

原始轨迹是运行日志，evolver 是编译器，skill 是压缩后的可执行规范，validator 是测试套件。这样可以借鉴软件工程：

- unit/integration/regression tests；
- semantic versioning；
- code review 与 provenance；
- canary、rollback、deprecation；
- coverage-guided evidence collection。

### 启发 2：成功轨迹不是正样本，而是不可破坏的 invariant

普通 error-driven repair 只读失败。SkillClaw 强调成功 session 定义现有 skill 的有效区域。更进一步可以为每个 skill 自动生成 behavioral contract：

$$
\text{preconditions}\rightarrow\text{tool sequence}
\rightarrow\text{postconditions}.
$$

候选修改必须保持已验证 contract，同时扩展 failure boundary。

### 启发 3：skill 的价值取决于可压缩性

`save report` 的路径与格式高度可压缩，deadline parsing 的语义推理不易压成规则。可以预先预测一个失败是否值得 skill 化：

$$
\text{skillability}
=\text{recurrence}\times\text{procedural specificity}
\times\text{verifiability}\times\text{transferability}.
$$

低 skillability 的问题应进入模型训练、规划搜索或工具改造，而不是继续加 instructions。

### 启发 4：真正的 collective signal 是“跨上下文稳定重复”

不是轨迹越多越好。最有价值的证据是同一 procedure 在不同用户、任务与环境中反复成功或失败。演化器应显式优化 evidence diversity，而不仅是 session count。

### 启发 5：validator 应主动选择最能区分版本的任务

当前 validator 使用相关任务比较旧/新 skill。可以升级为 active validation：寻找最可能暴露两版本差异的环境和 edge case，类似对抗测试或 mutation testing，从而用更少 token 得到更可靠的发布判断。

## 15. 最需要补的实验

1. 1/2/4/8/32 用户规模曲线，量化 collective aggregation 的边际收益；
2. shared evolution vs 每用户 isolated evolution；
3. evolution、validation、future test 三套任务严格隔离；
4. 多随机种子、误差条与任务级显著性；
5. Qwen、Claude、GPT 等不同 actor/evolver/validator 组合；
6. 独立程序 checker、人工 blind judge 与 LLM validator 的一致性；
7. 10/50/100 轮后的 library size、路由准确率和 context cost；
8. skill merge、dedup、conflict resolution 和 deprecation 消融；
9. 1%-20% 恶意用户或 poisoned session 下的全局污染率；
10. 用户隐私、tenant isolation 与删除请求测试；
11. accepted skill 在旧任务和未引用任务上的负向迁移；
12. token、执行时间、API、GPU 与人工审查的完整成本；
13. `skip`、保守编辑、history ledger、validator 各自的组件消融；
14. skill attribution 对 skill/agent/environment 根因的人工标注准确率；
15. 与 memory、test-time harness evolution、在线 RL 的等成本比较。

## 16. 一句话评价

> SkillClaw 的真正贡献不是“让 LLM 自己重写提示词”，而是把跨用户 Agent 经验变成一条带归因、版本历史、保守修改、执行验证和共享发布的程序性知识供应链；它已经展示了这条链对工具与环境型故障很有效，但 collective scaling、独立泛化和安全治理仍需更严格的证据。

## Source

- Ma, Z., Yang, S., Ji, Y., Wang, X., Wang, Y., Hu, Y., Huang, T., & Chu, X. (2026). *SkillClaw: Let Skills Evolve Collectively with Agentic Evolver*. arXiv:2604.08377v1.
- Local PDF: `C:/Users/huawei/Zotero/storage/LUAP93RB/Ma et al. - 2026 - SkillClaw Let Skills Evolve Collectively with Agentic Evolver.pdf`
- Code: https://github.com/AMAP-ML/SkillClaw

