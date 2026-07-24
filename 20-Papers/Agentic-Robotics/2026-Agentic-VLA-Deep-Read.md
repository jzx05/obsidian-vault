---
title: "Agentic-VLA 深度解读"
tags:
  - paper-deep-read
  - VLA
  - online-adaptation
  - RL-finetune
  - robot-manipulation
related:
  - "[[2026-Hi-VLA-Deep-Read]]"
  - "[[2026-Hi-VLA-Orchestration]]"
date: 2026-07-24
authors: Ruofan Jin, Zaixi Zhang (Scinetics)
venue: ICML 2026
arxiv: 2605.22896v1
---

> 与 [[2026-Hi-VLA-Deep-Read|Hi-VLA]] 对比读:Hi-VLA 是"把架构空间划清楚",Agentic-VLA 是"把训练回路塞满 agent"。两篇几乎不重叠——一个谈**推理时如何分层**,一个谈**微调时如何自动课程**。

## 0. 一句话把握全篇

**三块正统 idea 的整合工程,不是新范式**:

1. **ARS**(自适应奖励合成) = LLM 拆子目标 + capability-aware 加权 dense reward
2. **LGE**(语言引导探索) = 用预训练 VLM 给出自然语言 hint,拼到 prompt 里再采样动作
3. **EM**(经验记忆) = task embedding 检索历史 policy 权重做 warm-start

跑在 OpenVLA-OFT 上 + GRPO 优化,LIBERO 平均 97.8%,长程 +12.3%。**每个组件都不新**——ARS 是 Eureka/VLAC 那条线,LGE 是 SayCan 变体,EM 是 task-conditioned policy retrieval。**真正的贡献是把三者放进同一个 loop 并做出完整消融**。

工程价值:高。学术新颖性:中。**数字可信度:需要小心几个红旗**(§9)。

## 1. 问题定义

**背景痛点**(作者陈述,基本准确):

- SFT VLA 靠记轨迹,一步错步步错,泛化脆弱
- 每个任务要几百条示教,不 scalable
- 已有 RL fine-tune 方案(VLA-RL、SimpleVLA-RL、EVOLVE-VLA、π-RL)存在三个短板:
  - 奖励信号噪声大或需要 oracle
  - 探索靠随机采样,VLA 动作空间维度高,效率低
  - 没有跨任务知识迁移

**MDP 形式化**:$s_t = (o_t^{\text{vis}}, o_t^{\text{prop}}, l_{\text{task}})$,策略 $\pi_\theta$ 自回归输出 tokenized 动作。**关键假设**:部署时没有 oracle reward——所以必须自己造。

这个 setup 本身没什么问题,但注意:**"部署时无 oracle reward"** 是它整个方法论的支点。ARS 靠 VLAC 这个 pre-trained critic 给 progress 打分——这就等于把 oracle reward 的问题**外包给一个 pre-trained 模型**,不是真正解决了。

## 2. Adaptive Reward Synthesis (ARS,§3.3)

**三个子模块**,拼在一起才是 ARS:

### 2.1 Task decomposition(公式 1)

$$\mathcal{G} = \{g_1, \ldots, g_K\} = \text{LM}_{\text{decompose}}(l_{\text{task}})$$

用 Llama-3-8B 把任务拆成子目标。例:"turn on stove and put moka pot on it" → 5 步。

**这是老 idea**(Voyager、Inner Monologue、Code as Policies 都做过)。**新意在于配合下面的 capability weighting**。

### 2.2 Capability-aware reward adjustment(公式 2-3,核心创新点)

维护每个子目标的能力估计 $\hat{c}_k$:

$$\hat{c}_k^{(t+1)} = \alpha \hat{c}_k^{(t)} + (1-\alpha)\,\mathbb{I}[\text{success at } g_k]$$

奖励权重取补:$w_k = 1 - \hat{c}_k$。

**做了什么**:已经学会的子目标权重降下来,还没学会的权重拉满。这是一个 **auto-curriculum,而且是子目标粒度的**。

**为什么值得关注**:比 EVOLVE-VLA 的"progressive horizon extension"(任务粒度)细一层。§4.7 的对照实验(Table 6)专门比了 uniform / fixed-schedule / learning-progress / ARS 四种课程,ARS 最好。

**但要小心**:$\hat{c}_k$ 的更新依赖"success at $g_k$"这个信号——**这个信号哪来的?** 论文没说清。看上下文,应该也是 VLAC critic 给的一个阈值判断。也就是说 **ARS 的自适应性完全依赖 VLAC 的准确性**。

### 2.3 Progress-based dense rewards(公式 4)

$$R(\tau) = \sum_{k=1}^{K} w_k \cdot \Delta_k(\tau), \quad \Delta_k(\tau) = C_\phi(o_{\text{start}}^k, o_{\text{end}}^k, g_k)$$

VLAC (Zhai et al. 2025) 给每个子目标算 progress,乘上 capability 权重。

**这里 VLAC 是命门**。整篇论文的 reward signal 都是从它来的。VLAC 是 2025 年 arXiv preprint(§References),训练数据、泛化边界都还没建立。作者自己在 §C 的 failure analysis 也承认 **12% 的失败来自 reward hacking**——VLAC 给了高分但环境没判成功。

## 3. Language-Guided Exploration (LGE,§3.4)

### 3.1 VLM-based exploration critic(公式 5)

$$s_{\text{explore}} = \text{VLM}(o_t^{\text{vis}}, l_{\text{task}}, l_{\text{prompt}})$$

用 Qwen3-VL-8B-Instruct(零样本,无微调)看图 + 读任务,吐一句自然语言 hint:

- "The gripper is approaching from an occluded angle; try positioning from the left side"
- "The grasp point is too close to the object edge; target the center for stability"

### 3.2 Suggestion-conditioned action generation(公式 6)

$$a_t \sim \pi_\theta(a \mid s_t, l_{\text{task}} \oplus s_{\text{explore}})$$

把 hint 拼到 task instruction 后面,让 VLA 采样条件动作。**评估时不加 hint**——只训练时用。

### 3.3 Adaptive suggestion frequency(公式 7)

$$p_{\text{suggest}}(t) = p_{\text{max}} \cdot \exp(-\lambda \bar{R}(t))$$

随着 policy 变强,减少 hint 频率——防止 policy 一直"抱大腿"。$p_{\text{max}}=0.8$,$\lambda=0.5$。

**评价**:这个东西**本质是 SayCan / Inner Monologue 的变种**。真正的技术亮点是**adaptive frequency**——它自动把训练分成两阶段:早期靠 VLM 引导,后期自主。§4.6 消融显示去掉 LGE 掉 1.3%,加成有限但确实有。

**红旗**:附录 A.4 的 prompt template 和 A.3 的 task decomposition prompt **完全一样**——都是"You are a robotic manipulation expert..."。这提示 LGE 和 ARS 的 prompt 工程可能没做区分,或者是笔误。

## 4. Experience Memory (EM,§3.5)

### 4.1 结构

Memory bank $\mathcal{M} = \{(e_i, \theta_i, m_i)\}$:

- $e_i$:frozen encoder 出的 task embedding(768 维)
- $\theta_i$:该任务微调后的 policy 参数**全套**(!)
- $m_i$:成功率、迭代数等元数据

### 4.2 Warm-start retrieval(公式 9-10)

余弦相似度 top-$k$ ($k=3$),softmax 加权($\tau=0.1$,很尖锐):

$$\theta_{\text{init}} = \sum_j \frac{\exp(\cos(e_{\text{new}}, e_j)/\tau)}{\sum_{j'} \exp(\cdot)} \theta_j$$

### 4.3 "为什么权重空间插值不炸"(§3.5.2 特设段落)

作者显然预料到审稿人会拍——因为**独立训练的神经网络权重直接平均通常是灾难**(model soup 之所以能做是因为 fine-tune 起点相同)。他们给了三条辩护:

1. 所有 $\theta_j$ 都是从**同一个 OpenVLA-OFT 初始化**微调而来——loss basin 相近
2. 检索尖锐(low $\tau$、small $k$),主导项是最相似的那一个
3. 结果只作为**初始化**,后续 online adaptation 可以纠正

**这条辩护基本站得住**——但也告诉你 EM 的适用边界:**只在同一个 base VLA 微调出来的子任务之间有效**,不是通用机制。

### 4.4 存储成本问题

**Memory 存的是全套 policy weights**(§E "Scalability"). OpenVLA-OFT 大概 7B 参数,单条 ~14GB(fp16)。Capacity=100,那就是 **1.4 TB 的 memory bank**。作者承认这是问题,并提示未来做 LoRA / task-specific residual——所以现在这个方案在实验室能跑,产品化不现实。

## 5. 训练算法(Algorithm 1)

关键循环(简化):

```
θ ← WarmStartRetrieval(M, e_task)         # EM
G ← LM_decompose(l_task)                  # ARS: 拆子目标
init capability estimates c_k
for iter in 1..N:
    for i in 1..batch:
        τ_i ← rollout(π_θ, VLM_hint)      # LGE
    for τ_i in B:
        R_i ← Σ_k (1-c_k) · Δ_k(τ_i)      # ARS: 加权 progress
    θ ← GRPO(θ, B, {R_i})
    update {c_k}
M ← M ∪ {(e_task, θ, m)}                  # EM: 写回
```

**注意 GRPO**——这是 DeepSeekMath 那篇的 group relative policy optimization。用 group baseline 代替 value function,对 VLA 这种 tokenized 动作序列比较合适。

## 6. 主结果(§4.2)解读

Table 1 上 Agentic-VLA 平均 97.8%,vs. EVOLVE-VLA 95.8%(+2.0),vs. OpenVLA-OFT 89.2%(+8.6)。

**做了什么**:LIBERO 四个 suite 都刷了 SOTA,长程 +12.3% 最亮眼。5 seeds mean±std,std 都很小(±0.4-0.8)。

**发现什么**:
- 长程任务提升最大——ARS 的子目标课程在这里起作用最明显,符合直觉
- vs. EVOLVE-VLA 的 +2.0% **不算大**,但作者论证 seed variance ~0.7,所以确实显著
- One-shot(Table 2)的提升非常戏剧化:+26.9%,几乎接近 full-supervised。这是 EM warm-start 的功劳

**为什么值得关注**:说明**在有 pre-trained critic 的前提下**,online adaptation 确实能大幅补偿数据不足。

**但要打问号**:LIBERO 本身已经饱和——OpenVLA-OFT 就 89.2%,天花板 100% 只剩 10.8% 空间,任何合理方法都能挤进去。真正的 stress test 是 RoboTwin 2.0 Hard(§4.9),那里 subset avg 从 π0 的 16.3% 涨到 34.7%(+18.4)——**这个数字比 LIBERO 的 +2 说服力强得多**。

## 7. 消融(§4.6)与对照(§4.7)

Table 5 的消融很干净:

| 加法 | SR% | Iters |
|---|---|---|
| OpenVLA-OFT (SFT only) | 85.8 | - |
| + Vanilla RL (binary) | 87.7 | 2100 |
| + Progress reward | 91.3 | 1850 |
| + ARS | 94.6 | 1200 |
| + LGE | 96.2 | 880 |
| + EM | 98.1 | 700 |

**每一步都是 monotone 涨的,而且 iteration 数一路降**——这种曲线太干净反而让人怀疑是不是挑过 seed 展示的。但至少定性上支持每个组件都有用。

Table 6 的**对照实验**(controlled comparison)才是真正有说服力的部分——用 RND / ICM 替换 LGE,用 uniform / fixed-schedule / learning-progress 替换 ARS,预算匹配。ARS 和 LGE 都赢了替代方案。**这一节是本文相对 EVOLVE-VLA 最扎实的加分项**。

## 8. Cross-task transfer(§4.4,最激进的实验)

Table 3:LIBERO-Long 上训练,LIBERO-Object 上评估(无 target-task 示教)。

- Direct SFT transfer: **0%**
- EVOLVE-VLA: 20.8%
- Agentic-VLA: **31.2%**

**声称**:experience memory 让跨任务迁移从零变成非零。

**审慎解读**:
- 31.2% 还是远低于 full-supervised 的 96.6%,并没有"解决"迁移
- 关键机制不是 memory 本身,是**online adaptation + memory warm-start**——policy 还是在 target 任务上跑了大量 rollout(§4.5 用了 22.4k),只是起点更好
- **更公平的比较是"随机 warm-start + online adaptation"**——§4.7 Table 6 底部有:random retrieval 得 97.0%(在 LIBERO-Long 内部),EM 98.1%。**random retrieval 已经拿到 97%,EM 的边际提升只有 1.1%**。这暗示 warm-start 本身比"检索什么"更重要

## 9. 需要打问号的地方(critical read)

### 9.1 Per-task 数字过于整齐

Table 10-13 的单任务成功率**几乎全是 96 / 98 / 100 三个值**,50 trials 下这种整齐度可疑。50 trials 的最小分辨率是 2%,所以 98 = 49/50、96 = 48/50。**很多任务恰好停在 96/98/100**,而不是 94、97、99——**这与二项分布的期望不符**。

可能解释:
- 报告的是 5 seeds 的平均,平均后自然向中位数聚集
- 或者只报了 seed=0 的结果
- 或者是四舍五入到 2% 精度

作者应该给完整的 50 trials 分布,或者至少 seed-level 数据。

### 9.2 VLAC 依赖过重

整个 ARS 的 reward、capability 判定都靠 VLAC。VLAC (Zhai et al. 2025) 是 arXiv preprint,LIBERO 应该在其训练分布内——**LIBERO 的强表现有多少是 VLAC 见过任务,而不是 Agentic-VLA 学会了泛化?** RoboTwin 上的表现(§4.9)部分回答了这个,但需要 VLAC 在 RoboTwin 上的独立评估才能说清。

### 9.3 Reward hacking 12% + Exploration collapse 8%

§C 承认 20% 的失败是**系统性缺陷**(reward hacking + exploration collapse),不是随机噪声。这在 LIBERO 上还能被高天花板掩盖,真实场景放大会很严重。

### 9.4 与 EVOLVE-VLA 的复现问题

Table 1 的 EVOLVE-VLA 95.8% 是**作者复现的**(†标记),不是原论文数字。原 EVOLVE-VLA 论文(Bai et al. 2025)的报告是多少?**如果原始报告更高,那作者的复现可能压低了 baseline**。这是 VLA 领域常见问题,需要交叉验证。

### 9.5 arXiv ID 异常

论文标注 `arXiv:2605.22896v1 [cs.RO] 21 May 2026`——**2605 这个前缀不存在于当前 arXiv 编号系统**(现在是 25xx-26xx)。可能是模板占位符没改,也可能是笔误。不影响科学内容,但提示编辑不严谨。

### 9.6 RoboTwin 2.0 引用是 (?)

"the dual-arm RoboTwin 2.0 benchmark (?)" ——**bibkey 缺失**,§4.9 的关键 benchmark 引用没写出来。审稿版本没修好。

## 10. 与 Hi-VLA 的定位对比

| | [[2026-Hi-VLA-Deep-Read\|Hi-VLA]] | Agentic-VLA |
|---|---|---|
| 目标 | 推理时如何分层 | 微调时如何自适应 |
| 抽象层次 | 架构论文,划设计空间 | 系统论文,做组件整合 |
| 新颖性来源 | 用 options framework 统一多家分层方案 | 三个 agent 组件 + 完整消融 |
| 与 SFT 关系 | 高层保留 VLM,低层复用 VLA | 完全基于 SFT VLA 再做 RL |
| 数据需求 | 与底层 VLA 同 | **主打 low-data:1-shot、cross-task** |
| 主要贡献 | 概念/工程整理 | 训练算法 + 数字 |
| 弱点 | 抽象过度、创新有限 | 数字可信度、组件皆非首创 |

**两篇可以互补读**——Hi-VLA 告诉你**分层怎么切**,Agentic-VLA 告诉你**每一层内部怎么微调**。放在一起就是一套完整的 VLA 部署方案。

## 11. 值得偷的 idea

如果做 VLA 相关工作,可以直接拿走的东西:

1. **Capability-aware weighting**(公式 2-3):比 fixed curriculum 稍好,实现极简单,几乎零成本
2. **Adaptive suggestion frequency**(公式 7):把两阶段训练自动化,避免手动 schedule
3. **Task embedding + policy retrieval**:如果 base model 固定,warm-start 确实有用——但**存 LoRA / residual 而非全量参数**
4. **控制对照 vs. 消融的区分**(§4.7):不只是"去掉这个组件掉几分",而是"用标准替代品替换,谁更好"——审稿人看了会更服气

## 12. 一句话总结

**如果你要在 LIBERO 类 benchmark 上刷 SOTA,这三个组件都值得实现;如果你要理解 VLA 的本质挑战,这篇论文帮助有限——它把三个已知机制精致地拼起来,而没有触及 reward hacking、真实世界安全、参数存储这些底层问题(作者在 §E 也承认)**。工程 8/10,学术贡献 5/10,数字可信度 6/10。
