---
title: "SkillRise: Agentic Reinforcement Learning for Cross-Task Skill Evolution"
authors:
  - Zhiyuan Yao
  - Yuxin Chen
  - Zhengxi Lu
  - Zishan Xu
  - Yueqing Sun
  - Yifu Guo
  - Yuquan Lu
  - Zhengzhou Cai
  - Kangning Zhang
  - Zhuowen Han
  - Zihan Wang
  - Ziang Ye
  - Qi Gu
  - Xunliang Cai
  - Weiwen Liu
  - Yongliang Shen
year: 2026
venue: arXiv preprint
arxiv: 2607.26784v1
tags:
  - paper
  - agentic-rl
  - skill-learning
  - memory
  - cross-task-transfer
  - agentic-robotics
  - reinforcement-learning
status: read
rating: 8/10
date-read: 2026-08-01
url: https://arxiv.org/abs/2607.26784
code: https://github.com/Within-yao/SkillRise
pdf: "C:/Users/huawei/Zotero/storage/4BSE82UX/Yao et al. - 2026 - SkillRise Agentic Reinforcement Learning for Cross-Task Skill Evolution.pdf"
related:
  - "[[2026-Harness-VLA-Deep-Read]]"
  - "[[2026-TTHE-Test-Time-Harness-Evolution]]"
  - "[[2026-Tau0-VLA]]"
  - "[[2026-Hi-VLA-Deep-Read]]"
---

# SkillRise: 用 RL 学会跨任务沉淀技能

## TL;DR

SkillRise 解决的是 agentic RL 的一个细问题：标准 RL 把每个任务当独立 episode，任务结束经验就丢了；但真实 agent 会遇到一串“相关但不相同”的任务，前一个任务中学到的 procedure、pitfall、检查顺序，应该能帮助后面的任务。

它的做法很简洁：

```text
related tasks ordered from easy to hard
-> solve task i using current skill document S_{i-1}
-> get outcome reward r_i
-> curate / rewrite skill document S_i from trajectory and reward
-> use S_i on task i+1
```

训练上，它把“解当前任务”和“写技能文档”分开 credit assignment：

- solving phase 用当前任务 reward；
- curation phase 用后续任务的 discounted reward。

因此 skill document 写得好不好，不由当前任务是否成功直接决定，而由它能否帮助后续相关任务决定。这点对 Harness VLA 很有启发：机器人 memory 不应该只记录“这次成功/失败”，而应该优化成能提升下一批相似任务成功率的可迁移 skill artifact。

## 1. 论文想解决什么问题

很多 LLM agent benchmark 都是交互式长程任务，例如 ALFWorld、WebShop、ScienceWorld。标准 agentic RL 的训练目标是：

$$
J(\theta)=\mathbb{E}_{x\sim D,\tau\sim\pi_\theta(\cdot|x)}[R(\tau)]
$$

这等价于每个任务独立优化。问题是：

- 同一任务族的不同实例经常共享 procedure；
- 前一个任务里的失败经验可以抽象成后续任务规则；
- 但独立 episode RL 不会显式训练 agent 把经验转成可复用技能；
- 现有 skill bank / memory pipeline 又常常把 extraction、retrieval、execution 拆成多个模块，credit 很混。

SkillRise 的中心问题是：

> 能否让同一个 policy 同时学会 solve task 和 curate skill document，并用后续任务表现来监督“技能是否真的可迁移”？

## 2. 核心设定：相关但不同的任务序列

SkillRise 不从单任务采样，而是构造任务序列：

$$
\mathbf{x}=(x_1,\ldots,x_K)\sim G
$$

这些任务来自同一 family，但具体实体和目标不同。例如：

- ALFWorld：同类 household task；
- WebShop：同一产品类别；
- ScienceWorld：同类科学实验任务。

任务被按难度从简单到复杂排序。这个排序的意义是给技能迁移一个自然 curriculum：早期简单任务暴露基本 workflow，后期复杂任务检验这个 workflow 是否能扩展。

作者用环境提供的 family metadata 来分组，再用任务属性估计难度。这个假设很实用，但也是局限：真实开放世界里，“哪些任务属于同一族”本身就需要学习。

## 3. 方法：同一个 policy 交替 solve 和 curate

SkillRise 维护一个文本技能文档 $S_i$，表示前 $i$ 个任务沉淀出的可迁移知识。每个 sequence 从空文档开始：

$$
S_0 = \emptyset
$$

第 $i$ 个任务先用当前文档解题：

$$
\tau_i \sim \pi_\theta(\cdot | x_i, S_{i-1}), \quad r_i=R(\tau_i)
$$

然后如果不是最后一个任务，同一个 policy 切换到 curation 角色，基于旧文档、轨迹和结果重写文档：

$$
S_i \sim \pi_\theta(\cdot | S_{i-1}, \tau_i, r_i), \quad i<K
$$

非常关键的一点：

> 后续任务只看到 $S_i$，看不到前面完整 trajectory。

这迫使 skill document 成为跨任务信息的唯一通道。如果文档只是复读历史、塞具体物体名、写泛泛建议，就不会稳定帮到后面的任务。

## 4. Decoupled credit assignment

SkillRise 的训练核心是把两个阶段的 credit 分开。

对第 $n$ 条 trial、序列位置 $i$、阶段 $z\in\{\text{solve},\text{curate}\}$，定义 phase return：

$$
G_{i,z}^{(n)} =
\begin{cases}
r_i^{(n)}, & z=\text{solve} \\
\sum_{j=i+1}^{K}\gamma^{j-i} r_j^{(n)}, & z=\text{curate}, i<K
\end{cases}
$$

直觉：

- solving 发生在当前任务中，所以用当前任务 reward 监督；
- curation 发生在任务结束后，只影响未来，所以用后续 reward 监督。

然后作者做 role-aware group-relative optimization：只在相同 task group、相同 sequence position、相同 phase 的 trials 之间比较 advantage。

这样避免两个混淆：

- 不拿 solve return 去当 curate 的 baseline；
- 不拿早期简单任务和后期困难任务直接比较。

这其实是论文最漂亮的技术点。它把“写好经验总结”的学习信号变得干净：如果你写的 skill 帮了后续任务，你才得到 credit。

## 5. Prompt 设计：技能文档如何被写

SkillRise 的 curation prompt 反复强调几条规则：

- abstract, do not memorize；
- 删除具体实例，例如具体 object id、location number、品牌、价格；
- 每条规则必须 grounded in observed trajectory；
- 失败时提取 concrete pitfall；
- concise；
- 不要复制轨迹，distill reusable procedure。

推荐结构是：

```text
## When to use
## Workflow
## Pitfalls
```

这个 prompt 很值得搬到机器人 memory：

```text
不要记录“抓 red cup 1 在 table 2 成功”；
要记录“当目标是多个同类物体时，每抓取并放置一个后要重新定位剩余实例”。
```

它追求的是 skill abstraction，而不是 episodic memory dump。

## 6. 实验设置

Benchmarks：

| Benchmark | 任务类型 | 测试规模 |
|---|---|---:|
| ALFWorld | 文本版 embodied household tasks，含 Pick、Look、Clean、Heat、Cool、Pick2 | 128 held-out |
| WebShop | 搜索、配置并购买符合约束的商品 | 128 held-out |
| ScienceWorld | 文本科学实验环境 | 128 held-out |

Backbone：

- Qwen3-4B 主结果；
- Qwen3-1.7B 做模型规模分析。

训练配置：

- 每个 training batch 有 16 条 sequences；
- 每条 sequence 长度 $K=3$；
- 每条 sequence 有 $N=8$ independent trials；
- 每次更新所有 trainable methods 都处理 384 task plays；
- cross-task discount $\gamma=0.6$；
- 训练最多 150 updates；
- 8 张 NVIDIA H800；
- evaluation temperature 0.7。

对比方法：

- prompting：Zero-shot、ReAct、Reflexion；
- task-independent RL：PPO、RLOO、GRPO、GiGPO；
- repeated-attempt meta-RL：LaMer。

## 7. 主结果：Pass@1

Qwen3-4B 的 Pass@1：

| Method | ALFWorld Avg. | WebShop SR | ScienceWorld SR |
|---|---:|---:|---:|
| Zero-shot | 4.4 | 1.4 | 0.8 |
| ReAct | 25.0 | 2.1 | 7.8 |
| Reflexion | 28.9 | 2.7 | 7.8 |
| PPO | 67.2 | 74.2 | 39.1 |
| RLOO | 74.2 | 76.6 | 43.0 |
| GRPO | 79.7 | 75.8 | 42.2 |
| GiGPO | 83.6 | 77.3 | 46.1 |
| LaMer | 76.7 | 74.2 | 41.4 |
| SkillRise | 85.9 | 84.4 | 54.6 |

相对最强 baseline GiGPO，SkillRise 提升：

- ALFWorld：+2.3 pp；
- WebShop：+7.1 pp；
- ScienceWorld：+8.5 pp。

这说明 skill curation 不只是增加上下文，而是给 RL 一个更适合跨任务 transfer 的训练目标。

ALFWorld 分任务族结果：

| Family | SkillRise |
|---|---:|
| Pick | 100.0 |
| Look | 87.5 |
| Clean | 88.5 |
| Heat | 81.0 |
| Cool | 80.8 |
| Pick2 | 68.4 |
| Avg. | 85.9 |

SkillRise 在 Pick 上满分，但 Pick2 仍只有 68.4，说明“多实例/重复处理”仍是难点。附录 case 也正好展示了它从失败里学会 repeat-for-each-instance。

## 8. Pass@2/3：跨任务训练也能迁移到同任务重试

虽然 SkillRise 训练时看的是 distinct tasks sequence，但作者也测了同一个 held-out task 重试多次、每次携带更新后的 skill document。

| Method | ALFWorld Pass@3 | WebShop Pass@3 | ScienceWorld Pass@3 |
|---|---:|---:|---:|
| GiGPO | 89.1 | 80.5 | 55.5 |
| LaMer | 91.4 | 94.5 | 46.8 |
| SkillRise | 92.2 | 96.1 | 61.0 |

SkillRise 在 Pass@3 上三个 benchmark 都最好。尤其 ScienceWorld 比 LaMer 高 14.2 pp，说明它学到的 curation policy 并不只适用于跨任务，也能用于同任务 repeated attempts。

这对机器人很重要：真实部署中我们常有两类时间结构：

- 同一任务失败后重试；
- 同类任务连续出现，例如今天连续整理杯子、盘子、调料瓶。

SkillRise 表明，一个好的 skill curator 可以同时服务这两类结构。

## 9. Cross-task test-time scaling

作者把同一 held-out set 分成长度 $K\in\{2,4,6\}$ 的相关任务序列，每个任务只尝试一次，skill document 在序列中持续更新。

在 ALFWorld 上，SkillRise 从：

- K=2：83.6%
- K=6：87.5%

单调提升。相比最强 baseline 的 margin 从 +3.9 pp 扩大到 +8.6 pp。

这个结果的含义很关键：

> 性能提升不是因为同一个任务多采样，而是因为 agent 遇到越多相关但不同的任务，越能把 earlier interaction 转成 transferable skill。

对机器人来说，这提示我们应该评估“任务流中的成长”，而不是只评估 isolated task success。

## 10. 模型规模

ALFWorld 上：

| Method | Qwen3-1.7B | Qwen3-4B | 提升 |
|---|---:|---:|---:|
| GRPO | 75.0 | 79.7 | +4.7 |
| LaMer | 73.4 | 76.7 | +3.3 |
| SkillRise | 78.1 | 85.9 | +7.8 |

SkillRise 在 1.7B 和 4B 都最好，而且从小模型到大模型收益更大。作者解释为：SkillRise 同时需要 solve、抽象 regularities、写 reusable skill document，更吃模型容量。

这个点迁到 VLA 时要谨慎：机器人里低层动作模型不一定需要更大语言模型，但 high-level curator / memory writer 可能明显吃推理能力。

## 11. 消融

### Cross-task discount 不敏感

作者试了 $\gamma\in\{0.3,0.4,0.6,0.7\}$，最终训练曲线收敛差距约 1 个百分点。这说明方法不是靠精调 future reward 权重才成立。

### Curation 是必要的

No-curation variant 保留相同 $K=3$ 任务序列和交互预算，但去掉 inter-task curation，不允许技能跨任务传递。

结果：

- 早期训练三者都涨；
- 后期 SkillRise 明显拉开；
- 结束时 SkillRise 比 no-curation 高约 3 pp，比 GRPO 高 6 pp 以上。

结论：

> 只把相关任务放成 sequence 不够，必须显式把轨迹转成可复用 skill document。

## 12. 训练效率

作者把 SkillRise 和更复杂的 skill-learning pipeline 对比：

| Method | ALFWorld Avg. | Relative running time |
|---|---:|---:|
| RetroAgent | 85.9 | 6.0x |
| SkillRL | 73.4 | 4.3x |
| SkillRise | 85.9 | 1.0x |

SkillRise 和 RetroAgent 成绩相当，但运行时间只有 1/6。它的优势不是堆模块，而是把 solve 和 curate 放进同一个 policy，用下游结果直接监督 skill document。

这个对 Harness VLA 的启发很实际：不要一开始就做一个复杂 skill bank + retriever + external teacher + hierarchy distillation。可以先用一个 sequence-local skill document，让它作为唯一跨任务通道，看看是否能带来 test-time scaling。

## 13. Case study

### Case 1：从失败提取 repeat-for-each-instance

第一个任务要求把两个 spraybottle 放进 cabinet，agent 只放了一个就失败。curation 后 skill document 增加了规则：

```text
If multiple instances of the target object are present,
repeat the workflow for each instance.
```

第二个任务是把两个 peppershaker 放进 cabinet，agent 受这个 skill 指导，成功放了两个。

这体现了 SkillRise 想学的东西：不是记住 spraybottle 或 cabinet，而是抽象出“多实例目标必须逐一重复处理”。

### Case 2：成功后增量补充 open-if-closed

第一个任务把 cellphone 放到 bed，不需要打开容器；skill v1 记录了基本 placement workflow。第二个任务把 tomato 放进 microwave，需要先 open microwave。成功后 curation 给 skill v2 增加：

```text
If the destination receptacle is closed, open it before placing.
```

这说明 skill evolution 不只从失败学习，也从成功轨迹里提取条件化 procedure。

## 14. 局限

论文的局限比较清楚：

- 依赖 task-family metadata 来构造相关任务序列；
- 实验只到 4B 模型；
- benchmark 都是文本交互环境；
- reward 是 verifiable outcome，真实机器人很多任务 reward 不容易自动判定；
- skill document 是短文本，缺少和视觉、几何、接触状态绑定的结构化表示；
- 没有真实 embodied manipulation 或 multimodal VLA 实验。

所以它不能直接证明“真实机器人可以靠文本技能文档自我进化”，但它给了一个干净的训练范式。

## 15. 对 Harness VLA 的启发

### 启发 1：把任务流组织成相关但不同的 robot skill family

Harness VLA 不应该只测孤立任务，例如 “put cup on tray”。可以构造 sequence：

```text
put one cup on tray
put two cups on tray
put mug into cabinet
put bowl into drawer
put fragile glass on shelf
```

它们共享：

- 定位目标；
- 预抓取；
- 接触抓取；
- 运送；
- 目标容器可达性检查；
- 放置后验证。

但具体物体、容器、遮挡、数量、可开合状态不同。这样后续任务表现才能检验 memory 是否真正 transferable。

### 启发 2：Harness VLA 的 memory writer 应该用 downstream success 训练

现有 Harness VLA 记忆更像：

```text
这类任务中 VLA 负责抓取，analytic primitive 负责 transport。
```

SkillRise 提醒我们：memory 写得好不好，不能只看当前任务是否成功，而要看它对后续任务有没有帮助。

可定义 robot curation return：

$$
G_i^{curate} = \sum_{j=i+1}^{K}\gamma^{j-i} R_j
$$

其中 $R_j$ 可以是：

- task success；
- subgoal predicate pass；
- collision-free；
- retry count penalty；
- execution time penalty；
- human intervention penalty；
- object disturbance penalty。

这样 memory writer 会被训练成“写后续能用的东西”，而不是写漂亮总结。

### 启发 3：技能文档要抽象，但必须 grounded

SkillRise 的 prompt 原则非常适合 Harness VLA：

- 不记具体 xyz 坐标，记相对关系和条件；
- 不复制动作轨迹，记 primitive workflow；
- 失败要写 concrete pitfall；
- 成功要写适用条件；
- 所有规则必须来自实际执行或可验证观测。

机器人版 skill document 可以长这样：

```yaml
skill: container_placement
when_to_use:
  - target object must be placed inside or on a destination receptacle
workflow:
  - localize target and destination from current observation
  - use VLA_ACT for contact-rich grasp
  - verify object is attached before transport
  - if destination is closed or occluded, open or clear before placement
  - use analytic transport after stable grasp
  - release only after pose and clearance checks pass
pitfalls:
  - if multiple target instances exist, repeat verification and placement for each instance
  - if gripper closes but object pose does not move, treat as empty grasp and re-localize
```

这比原始 episodic memory 更可执行，也更容易被 ablation。

### 启发 4：SkillRise 可以补 Harness VLA 的“全局记忆学习”短板

[[2026-Harness-VLA-Deep-Read]] 强在部署时用 LLM agent 调度 frozen VLA 和 analytic primitives，但它的 memory 主要由 exploration / manual-like trace 生成，尚未被明确训练成 downstream-useful。

SkillRise 可以补上训练目标：

```text
robot sequence rollout:
solve task i with memory S_{i-1}
curate S_i from trajectory + outcome
solve task i+1 with S_i
credit S_i by downstream task outcomes
```

这会把 memory 从 heuristic prompt engineering 变成一个可优化模块。

### 启发 5：不要急着做庞大 SkillBank，先做 sequence-local memory

机器人领域很容易想做全局 skill library + retriever + vector DB + verifier + teacher distillation。SkillRise 的经验是：简单、端到端、唯一信息通道，反而更清楚。

建议 Harness VLA 的下一步实验可以先做：

- 每个 task family 一个 rolling skill document；
- 不做复杂 retrieval；
- 每条 sequence 从空 memory 开始；
- 只允许 memory 通过 curator 更新；
- 测 K=1/2/4/6 的 test-time scaling。

如果 K 越长成功率越高，才说明 memory 真的在跨任务学习。

### 启发 6：把 SkillRise 和 TTHE 合起来

SkillRise 学的是 skill document；TTHE 学的是 executable harness。对 Harness VLA 来说，可以分两层：

| 层 | 学什么 | 监督 |
|---|---|---|
| SkillRise-like memory | 可迁移 task workflow、pitfall、适用条件 | 后续任务成功率 |
| TTHE-like harness | primitive routing、verifier、retry branch、stop condition | 执行日志 proxy + safety checks |

也就是说：

```text
SkillRise: what should be remembered?
TTHE: how should remembered evidence change the executable controller?
```

这两个拼起来，比单纯 “LLM + memory + VLA_ACT” 更有论文味。

## 16. 我对这篇的判断

我给 8/10。

优点：

- 问题定义清楚：跨任务技能学习，而不是单任务重试；
- credit assignment 很干净，solve 和 curate 的时间角色被分开；
- 结果覆盖 ALFWorld、WebShop、ScienceWorld；
- 证明了 cross-task test-time scaling；
- pipeline 简洁，效率比复杂 skill-learning pipeline 好很多。

主要扣分：

- 还停留在文本环境；
- 依赖 task family metadata；
- 技能文档没有和多模态状态绑定；
- reward 都是相对容易验证的 benchmark outcome；
- 真实机器人里的不可逆失败、安全成本、长时漂移都还没碰。

我的一句结论：

> SkillRise 的精华不是“写一个 memory prompt”，而是把 memory writing 变成一个由后续任务表现监督的行为；这正是 Harness VLA 想从经验变成可学习系统时缺的那块骨架。

