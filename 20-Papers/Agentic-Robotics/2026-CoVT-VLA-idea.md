---
title: "CoVT-VLA (working title): Action-Grounded Continuous Visual Thought Tokens for Vision-Language-Action Models"
authors:
  - "(你)"
year: 2026
venue: "(target: ICLR / ICML / CoRL 2027)"
tags:
  - paper-idea
  - VLA
  - visual-chain-of-thought
  - robot-manipulation
  - auxiliary-supervision
  - efficient-VLA
status: idea
date-created: 2026-07-31
related:
  - "[[2026-TurboVLA]]"
  - "[[2026-Hi-VLA-Deep-Read]]"
  - "[[2026-Agentic-VLA-Deep-Read]]"
  - "[[2026-Harness-VLA-Deep-Read]]"
  - "[[2026-ASPIRE]]"
inspired-by:
  - "Chain-of-Visual-Thought (arXiv 2511.19418)"
---

## One-liner

> Chain-of-Visual-Thought showed that VLMs *think* better with continuous visual tokens.
> We show that VLAs *act* better with them —
> and we get that "thought" almost for free by distilling from **action-grounded** experts
> (target mask, affordance, contact frame) and from the **demonstration trajectory itself**.

---

## 1. Motivation

### 1.1 The supervision asymmetry no one talks about

一个 VLA 训练样本的输入信息量和监督信息量差 3-4 个数量级：

| 项目 | 每样本 scalar 量级 |
|---|---:|
| Visual input (3 view × ~500 ViT tokens × 768 dim) | ~10^6 |
| Language input (~20 tokens × 768 dim) | ~10^4 |
| **Action supervision (LIBERO H=12, 7-DoF)** | **84** |
| Action supervision (RoboTwin H=50, 14-DoF) | 700 |

也就是说：**输入维度和监督维度之间存在 3-4 个数量级的失衡**。
模型需要从几十个 end-effector 回归 scalar，反推出：

- 目标物在哪
- 参照物在哪
- 该抓哪一点
- 障碍在哪
- 未来 H 步该怎么走

这是 VLA 数据饥渴、object/scene 泛化差、attention map 频繁"看错但抓对"（shortcut）的**根本原因**——不是模型不够大，而是**监督不够密**。

### 1.2 Existing reasoning interfaces for VLA are wrong for this problem

看 `V + L → 中间态 → A` 中的"中间态"，现有做法有四类：

| 类型 | 代表 | 中间态 | 缺点 |
|---|---|---|---|
| Autoregressive text-action | OpenVLA, RT-2 | 无 / 隐式 | 无中间监督，慢 |
| Action-expert VLA | π₀, π₀.₅ | 隐式 latent | 大 LLM backbone，中间态黑箱 |
| Embodied CoT (text CoT) | ECoT, RoboBrain | **文本** CoT | 空间信息丢失、串行慢 |
| Token-fusion 轻量 | TurboVLA | 双向 V-L attention (隐式) | 对象关系隐式，组合泛化不确定 |

**文本 CoT 的核心问题**：manipulation 里的推理是**空间的**（这个杯子在那个盘子右边），
用离散 token 序列表示会丢失可微分的空间结构。

TurboVLA 反过来：**快、但完全隐式**，等于没中间态。
它自己都承认 "视觉-语言-动作之间的对象关系和目标关系仍然是隐式的"。

### 1.3 CoVT 提供的第四条路

Chain-of-Visual-Thought (arXiv 2511.19418) 的核心 claim：

> 视觉信息稠密、空间结构化，把它压进文本 CoT 会丢失。
> 用**连续可微的 visual token 链**才能保留可传递的空间推理状态。

CoVT 的做法（我的理解，需与 PDF 对齐）：

- 在视觉 token 后、文本输出前插入 K 个 learnable "visual thought tokens"。
- 每个 thought token 有一个轻量 decoder，训练时被视觉专家（depth / seg / sketch / keypoints）监督。
- 推理时**不需要专家**，thought tokens 是模型 forward pass 里的中间态。
- 关键：连续 > 离散，因为空间信息稠密可微，比离散 language token 更适合视觉推理。

**问题在于**：CoVT 用的是**通用感知专家**（depth, generic segmentation, sketch）——
和 manipulation 的下游目标弱耦合。对 VLA 来说，我们能用更强、更 action-oriented 的信号。

### 1.4 Our claim

> **传统 video/action loss 太弱不是因为架构，而是因为监督信号错位。**
> 视频/图像给的是 10^6 scalar 的输入，动作只有几十 scalar 的输出；
> 中间没有任何一层监督告诉模型"你视觉推理对不对"，只告诉它"末端位姿差多少"。
>
> **我们把 CoVT 的 continuous visual thought 从"通用感知"改造成"动作可蒸馏的中间态"**：
> - 一部分 thought token 由 **manipulation-specific 视觉专家**（SAM/AnyGrasp/FoundationPose）监督；
> - 一部分 thought token 直接由**演示轨迹本身**（ee 投影、future frame latent、接触帧）监督——零专家依赖。
>
> 这把 100 个 scalar 的动作监督，扩展成 K × H×W 的稠密像素级 + trajectory 级监督，
> 同时保留 TurboVLA 的效率优势（推理时无专家）。

---

## 2. Method

### 2.1 Overall architecture

Backbone 直接沿用 TurboVLA 的骨架（DINOv3 + BERT + 6-layer bidirectional V-L interaction + ACT-style action decoder），因为它已经是最强的轻量 baseline，且效率天花板已知。

改动集中在 **V-L interaction 输出**和**Action decoder 输入**之间：

```
multi-view images ──► DINOv3 ──► Z_v
instruction        ──► BERT   ──► Z_l
                        │
                        ▼
              Bidirectional V-L Interaction (6 layers, 与 TurboVLA 相同)
                        │
                        ▼
                Z_vl = [V_N ; L_N]
                        │
                        ▼
         ┌──────────────────────────────────────────┐
         │  Visual Thought Chain (K learnable Qs)   │
         │  Q_1, Q_2, ..., Q_K                      │
         │  cross-attend to Z_vl                    │
         │  (optional causal self-attn among Q_i)   │
         └──────────────────────────────────────────┘
                        │
        ┌───────────────┼──────────────────────┐
        │               │                      │
        ▼               ▼                      ▼
   Aux Decoder_1   Aux Decoder_k         Action Decoder
   (mask / grasp / (subgoal latent /      (ACT-style,
    contact heat)   trajectory)            K,V = [Q_1..Q_K, s_t])
        │               │                      │
        ▼               ▼                      ▼
  L_distill_1    L_distill_k              L_action (L1)
```

关键设计点：
- **Information bottleneck**: Action decoder 的 K/V **不再**直接使用 Z_vl，
  只使用 `[Q_1..Q_K, robot_state]`。这强制 thought tokens 承担全部推理信息传递，
  避免 "auxiliary head 学得很好但 action 走 residual 旁路" 的 decoupling 失败。
- **推理时**：Aux decoders 全部丢弃；模型只跑到 Q_1..Q_K 和 Action decoder，效率与 TurboVLA 同量级。

### 2.2 Thought token 的三层监督（重点）

按信号来源分成三类，逐类蒸馏成本递增：

#### Layer A — Free-from-demo (无外部专家)

演示数据里天然包含 action-grounded 的密集监督，但传统 VLA loss 完全没用它们：

| Thought token | 监督信号 | 计算方式 | Loss |
|---|---|---|---|
| **Q_ee_now** | 当前 ee 在图像的 2D/3D 位置 | 相机内外参 × proprioception | MSE + heatmap |
| **Q_ee_future** | 未来 H 步 ee 轨迹在图像投影 | 演示 action chunk 投到当前帧 → polyline / heatmap | Heatmap BCE |
| **Q_contact** | 首次接触帧的 gripper closure mask | 演示 gripper state 事件 | BCE |
| **Q_goal_delta** | (t=T) - (t=0) 的像素差 mask | 演示始末帧差分 | BCE |

**这一层是核心**。零专家依赖、零标注成本，直接把 100 个 scalar 的动作监督
扩展为 K × 224 × 224 的 heatmap 监督。**这也是与 CoVT 最不同的部分**——
CoVT 完全依赖外部专家，而 VLA 有免费的时序自监督。

#### Layer B — Distilled-from-manipulation-experts

补充语义 grounding 能力（"该对哪个物体"、"该抓哪里"）：

| Thought token | 蒸馏来源 | 语义 |
|---|---|---|
| **Q_target_mask** | Grounded-SAM (指令 → mask) | "指令说的物体" |
| **Q_reference_mask** | Grounding DINO | "参照物" |
| **Q_grasp_map** | AnyGrasp / GraspNet-1B / Robo-Point | "抓取点 heatmap" |
| **Q_depth** | Depth-Anything V2 | 相对深度 |
| **Q_pose** | FoundationPose (可选，昂贵) | 6-DoF 目标位姿 |

MVP 只用前两个（Q_target_mask, Q_grasp_map），跑通了再加。

#### Layer C — Distilled-from-world-model (最 novel 也最贵)

| Thought token | 监督信号 | 备注 |
|---|---|---|
| **Q_subgoal_latent** | 演示未来第 H 帧过 VAE 的 latent | 只蒸 latent 不生成像素，成本可控 |
| **Q_flow** | RAFT 从 t → t+H 的光流 | |
| **Q_delta_graph** | SGG(t) 与 SGG(t+H) 的差 | 与 ActionGraph 方向的过渡形态 |

Layer C 建议放在 v2，第一篇不做。

### 2.3 Chain 结构 (thought tokens 之间怎么交互)

三种选择：

1. **Parallel**: Q_1..Q_K 只 cross-attend to Z_vl，彼此独立。最快、最并行。
2. **Causal chain**: Q_k 可以 self-attend to Q_{<k}，Q_1 最粗（target），Q_K 最细（action-ready latent）。有 "chain" 的语义，但慢一些。
3. **Ordered layers**: 每一层解一个 thought token，逐层精化。最像 CoVT。

推荐 **Parallel 起步**（保持 TurboVLA 的 32Hz），如果性能不够再上 Causal。这个也是消融点。

### 2.4 训练课程

分三阶段（防止 auxiliary loss 主导前期训练）：

1. **Stage 1 — Thought warm-up** (前 10k steps): 冻结 action decoder，只训 Q_i 和 aux decoders。
2. **Stage 2 — Joint** (10k–70k steps): 全部联合训练，loss 权重初期 λ_aux 大，逐步退火。
3. **Stage 3 — Action fine-tune** (70k–80k steps): λ_aux → 0，只优化 action loss。

总 loss:
```
L = L_action + Σ_i λ_i(t) · L_distill_i + λ_consistency · L_ee_consistency
```

其中 `L_ee_consistency` 是把 Q_ee_future 解码的轨迹终点与 action chunk 展开的 ee 终点做 L1
——**免费的、几何一致性约束**，也是回答 "action 到底对不对" 的一个关键锚点。

### 2.5 Failure modes to actively guard against

| Failure | 表现 | 防御 |
|---|---|---|
| **Decoupling** | Aux 学得好但 action 不涨 | Information bottleneck (§2.1) |
| **Misalignment** | SAM 说 "apple" 但任务要抓 "apple stem" | Layer A (from-demo) 优先；Layer B 加权低 |
| **Auxiliary domination** | 前期 aux loss 太大压垮 action | 课程 §2.4 |
| **Latency regression** | Aux 增加 forward cost | 推理时丢弃 aux decoders |

---

## 3. Experiments (MVP first)

### 3.1 MVP: 最小可行实验（决定这条路是否成立）

**目标**：回答一个二值问题——加 thought tokens 到底让 success rate 涨还是不涨？

- **Backbone**: TurboVLA 原版 (DINOv3-ViT-B + BERT, 0.2B)
- **Dataset**: LIBERO (4 suites)
- **Variants**:
  - `T0` = TurboVLA 原版 (baseline)
  - `T1` = + Q_target_mask (Grounded-SAM 蒸馏)
  - `T2` = + Q_ee_future (演示投影)
  - `T3` = T1 + T2
- **Metrics** (关键 — 不能只看 avg success):
  - Success rate per suite (Spatial / Object / Goal / Long)
  - **Target grounding IoU** (Q_target_mask decoder 输出 vs GT mask)
  - **Trajectory endpoint error** (Q_ee_future 解码轨迹终点 vs 演示)
  - Latency & VRAM (确保没退化超过 15%)

**判定标准**:
- 如果 T3 在 Spatial + Long 上比 T0 平均涨 **≥ 2 pp**，方向成立。
- 如果只有 grounding IoU 涨但 success 不涨 → decoupling，说明 bottleneck 没做到，回炉 §2.1。
- 如果 grounding IoU 不涨也 success 不涨 → thought tokens 没有学到东西，回炉课程 §2.4。

### 3.2 Full experiments (MVP 成立后)

1. **主实验**: LIBERO 4 suites + RoboTwin 2.0 上完整比较 (与 TurboVLA / π₀.₅ / OpenVLA 同表)。
2. **消融**:
   - Thought token 数量 K ∈ {1, 3, 5, 8}
   - Chain 结构 (Parallel / Causal / Ordered)
   - 各 aux loss 逐一 drop
   - Information bottleneck on/off (直接切断证明 §2.1 的必要性)
3. **诊断指标**:
   - Compositional relation split (left/right, on/under 替换)
   - Distractor object 数量 vs wrong-object rate
   - Grounding IoU 与 success rate 的 correlation
4. **Real-world**: AgileX Piper 4 任务（沿用 TurboVLA 的设置），主要看 grasp 精度。

### 3.3 Efficiency claim to protect

TurboVLA 已经把 "快" 做到 31.2ms/32Hz。我们**不与它正面争速度**，只需保证：

- Inference latency ≤ 1.15 × TurboVLA
- VRAM ≤ 1.5 × TurboVLA
- Params 增加 ≤ 100M（thought queries + aux decoders 都在训练时用，推理时丢）

论文卖点应改成 "same-efficiency-class, better reasoning"，而不是 "faster"。

---

## 4. Related work positioning

| Type | 代表 | 中间推理 | 我们的差异 |
|---|---|---|---|
| Autoregressive text-action | OpenVLA, RT-2 | 隐式 / 无 | 显式连续视觉中间态 |
| Action-expert VLA | π₀, π₀.₅ | 隐式 latent | 加显式蒸馏监督 |
| Embodied text CoT | ECoT, RoboBrain | 文本 CoT | 连续、可微、快 |
| Token-fusion 轻量 | TurboVLA | 无 | 加 action-grounded thought chain |
| Continuous visual CoT (VLM) | **CoVT** | Learnable + generic expert | 换成 **action-grounded expert + demo-derived** |
| Graph-centric VLA | ActionGraph-VLA (future) | 显式关系图 | Ours 是 CoVT → Graph 的过渡形态 |

关键的一句 positioning:

> TurboVLA showed execution-level VLA need not be LLM-centric.
> CoVT showed VLMs can reason via continuous visual tokens instead of text.
> **We combine them: an execution-level VLA that reasons via continuous, action-grounded visual thought tokens — cheaper than text CoT, denser than TurboVLA, more general than any single expert.**

---

## 5. Open questions / 待与 PDF 校准

1. CoVT 的 chain 是 causal 还是并行？影响我们 §2.3 的选型。
2. CoVT 的 aux loss weight schedule 具体怎么退火？可能直接抄。
3. CoVT 是否冻结 backbone？我们希望冻结 DINOv3 保留 TurboVLA 的效率。
4. CoVT 论文里的 continuous > discrete 消融是否包含 language-token quantized 版本？如果有可以直接借用论点。

## 6. Risks & 反驳预案

- **"这不就是 auxiliary loss / multi-task learning 吗？"**
  → 是，但关键差异在 (a) information bottleneck 强制 thought tokens 是 action 的唯一入口，
  (b) demo-derived 的 Layer A 监督是新颖的，
  (c) 我们首次把 CoVT-style 连续视觉 CoT 用于 action，而不是 language output。

- **"直接加 SAM/Grasp head 效果一样吗？"**
  → 需要消融验证。假设是：连续 thought token 允许信息在 aux 和 action 之间**双向流动**，
  而普通 aux head 只是单向 supervision。这是 CoVT continuous claim 的直接迁移。

- **"LIBERO 已经饱和 (~98%)，还有涨点空间吗？"**
  → 平均分是饱和了，但 Long/Spatial 还差 3-6 pp；同时 compositional split 和 distractor 场景下所有方法都不饱和。这是我们的主战场。

- **"和 π₀.₅ 有本质区别吗？"**
  → π₀.₅ 中间态是隐式 latent，没有蒸馏监督；我们显式蒸馏 + information bottleneck。
  参数量少 15x，中间态可解释。

---

## 7. TODO

- [ ] 拿 CoVT PDF 补全 §5 四个开放问题
- [ ] 跑 MVP (§3.1): T0/T1/T2/T3 in LIBERO，预计 4 × 80k steps
- [ ] 写 Layer A 的数据处理脚本（相机投影 + heatmap 生成）
- [ ] 决定 Grounded-SAM 用离线蒸馏（预算标签）还是 online（贵但灵活）
- [ ] 准备 compositional relation split 的测试集
- [ ] 想清楚论文一句话卖点，锁死不再改

---

## 8. 一句话汇总

> Video/action loss 弱的根源不是 diffusion vs autoregressive，
> 而是**监督维度和输入维度的 3-4 个数量级落差**。
> CoVT 用连续视觉 token 让 VLM 思考；
> 我们把它改造成 action-grounded 版本，让 VLA 在**同 TurboVLA 效率下**做出显式、稠密、可诊断的空间推理。
