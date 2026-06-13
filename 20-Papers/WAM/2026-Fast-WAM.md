---
title: "Fast-WAM: Do World Action Models Need Test-time Future Imagination?"
authors: [Haoyuan Yuan, Zhilin Dong, Yucheng Lin, Hang Zhao]
year: 2026
venue: arXiv
tags: [paper, WAM, world-action-model, embodied-AI, video-prediction, policy-learning]
status: reading
rating: 
date-read: 
url: https://arxiv.org/abs/2603.16662
pdf: "C:\\Users\\Administrator\\Zotero\\storage\\QB328I9K\\Yuan 等 - 2026 - Fast-WAM Do World Action Models Need Test-time Future Imagination.pdf"
priority: high
---

# Fast-WAM: Do World Action Models Need Test-time Future Imagination?

## TL;DR
> WAM（World Action Models）的性能增益主要来自训练时的视频建模（video co-training），而非推理时的未来帧生成（test-time imagination）。Fast-WAM 保留训练时的视频目标，但在推理时完全跳过未来视频解码，直接从潜空间产生动作，大幅降低推理开销同时保持甚至超越完整 WAM 的性能。

## 核心问题
- 现有 WAM 遵循 imagine-then-execute 范式：先生成未来视频，再基于视频做决策
- 这带来巨大推理开销（视频解码 + 多步生成）
- **本文追问**：WAM 的收益到底来自哪里？是训练时的视频建模？还是推理时的未来想象？

## 相关工作 
VLA 在语义理解很强，但是 action-conditioned dynamics (给定当前状态和动作，预测下一个状态) 不一定好。Fast-WAM 在测试时也像 VLA 一样是直接输出动作的 policy interface，不需要先生成未来视频；但它在训练阶段额外加入了基于未来视觉预测的 world-modeling objective。

## 方法：Fast-WAM

### 架构图
![[Excalidraw/Fast-WAM-architecture.excalidraw]]

### 架构设计
- 基于 Video [[30-Notes/concepts/Diffusion-Transformer|DiT]]（Diffusion Transformer），构建在预训练 DiT（Wan2.5-1B）之上
- 输入分三组 token：
  1. **Clean first-frame tokens**：当前观察（清晰帧）
  2. **Noisy future video tokens**：用于视频重建训练
  3. **Action tokens**：动作预测（learnable tokens）
- **关键设计**：结构化注意力掩码（structured attention mask）
  - 训练时：video tokens 和 action tokens 共享视觉上下文，但 action tokens 不能看到 future video tokens
  - 推理时：直接丢弃 future video tokens，仅用 clean first-frame + action tokens

### 训练目标
三个损失的加权组合：
- $\mathcal{L}_{act}$：动作预测（flow matching）
- $\mathcal{L}_{vid}$：视频重建（flow matching on video）
- $\mathcal{L} = \mathcal{L}_{act} + \mathcal{L}_{vid}$

### 推理流程
- 不生成未来视频帧
- 仅输入当前 clean first-frame tokens → 直接输出动作
- 推理开销远低于标准 WAM

## 可控实验（Controlled Variants）

为解耦各因素贡献，设计了一组变体：
| 变体 | 视频 co-training | 推理时视频生成 | 说明 |
|------|:---:|:---:|------|
| Fast-WAM | ✓ | ✗ | 完整方案 |
| Fast-WAM-joint | ✓(joint attn) | ✗ | action 可看 video tokens |
| Fast-WAM-SIM | ✗ | ✗ | 无视频建模（纯 action） |
| Fast-WAM w/ video co-run | ✓ | ✓ | 完整生成+推理 |

## 实验结果

### RoboTwin（模拟）
- Fast-WAM 达到 **91.8%** 成功率（无 embedded pretraining）
- 超越多数使用 embedded pretraining 的 WAM 基线
- 关键发现：去掉 video co-training（SIM 变体）性能大幅下降 → 证明训练时视频建模是关键

### LIBERO（模拟）
- Fast-WAM 达到 **97.6%** 平均成功率
- 在 LIBERO-Spatial/Object/Goal/Long 四个子任务均表现竞争力强

### 真实机器人（Galaxea R1 Lite）
- 毛巾折叠任务，106 段演示，10K 步训练
- Fast-WAM 策略版本（$r_5$）成功率最高，同时完成时间最短
- 去掉 video co-running 同时降低成功率和增加完成时间

### 关键发现
1. **视频 co-training 是性能增益的主要来源**，而非推理时的未来生成
2. **去掉推理时视频生成对性能影响很小**，但推理效率大幅提升
3. Fast-WAM 推理延迟约 ~10ms，而完整 WAM 需要数百毫秒
4. 适合实时闭环控制场景

## 为什么读
- 直接挑战 WAM 的核心设计假设（imagine-then-execute）
- 提供了一个简洁高效的 WAM 设计范式
- 实验设计清晰，可控变体有效解耦各因素贡献
- 对实时机器人部署有直接参考价值

## 关联
- 同主题：[[20-Papers/WM/2026-minWM-Real-Time-Video-World-Models]]
- 概念：[[30-Notes/concepts/World-Model]]
- 相关方法：[[20-Papers/WM/2024-Self-Forcing]]


## 追问

## MoT 核心设计

### Q1: 为什么 video 在 self-attention 中看不到 action，但 action 能看到 video？

这是一个**推理效率驱动的不对称设计**。

**如果双向都开（video ↔ action）：**
- 训练时没问题，但推理时灾难性地慢。因为 video 的 K/V 依赖 action tokens 的状态，而 action 每步去噪都在变化 → video KV 每步都得重算 → 无法 cache → 推理耗时 × N步。
- 本质上退化为"每步去噪都完整跑一次 MoT"，丧失了 prefill + cache 的可能性。

**如果反过来（video 看 action，action 看不到 video）：**
- 语义上不合理。Action expert 的任务是"看到当前画面后决定下一步动作"。如果 action 连视觉信息都看不到，它只能从 text description 猜动作，退化为语言条件的 policy，丧失了视觉-动作 grounding。
- Video 看 action 在训练时有意义（知道动作才能生成正确视频），但这已经通过 cross-attention 里的 gt action 实现了，不需要在 self-attention 层重复。

**当前设计的妙处：**
- Video 不看 action → video 的 KV 不依赖 action → 可以 cache
- Action 看 video 第一帧 → action 能从视觉获取信息
- 信息流方向：视觉观察 → 动作决策（符合机器人控制的因果关系）

---

### Q2: 为什么 action 只能看 video 第一帧，而不是所有帧？

**核心 trade-off：推理速度 vs 训练时的信息量。**

如果 action attend to 所有 video frames：
- 训练时：action 能获取更丰富的时空视觉信息（看到未来帧的视觉线索），可能学到更好的策略
- 推理时：必须先生成所有未来帧 → 再用 action 看这些帧 → 回到了"需要 test-time imagination"的老路，论文的核心贡献（快速推理）就不成立了

当前设计的代价：
- Action 只能从一帧静态画面获取视觉信息，无法看到运动趋势（motion cue）
- 这要求 video expert 的第一帧 KV 必须编码足够丰富的特征（因为 action 只看这些）

为什么依然够用：
- 第一帧经过整个 video expert（所有层），其 KV 不只是原始像素信息，而是经过多层 attention + FFN 后的高级语义表示
- Text context 补充了 high-level task 信息
- 对于短 horizon 的桌面操作（LIBERO/RoboTwin），当前观察 + 任务指令通常足以确定接下来的动作

---

### Q3: 为什么 MoT 共享的是 self-attention 而不是 cross-attention 或 FFN？

**Self-attention 是唯一能实现"跨模态 token 级交互"的选择。**

各层的职责分工：
- **Self-attention**：token 之间的信息交互（"谁和谁对话"）
- **Cross-attention**：从外部条件注入信息（"参考什么指令"）
- **FFN**：每个 token 独立的非线性变换（"如何处理信息"）

为什么不共享 cross-attention：
- 两个 expert 的 cross-attention context 不同（video 看 text+action，action 只看 text）
- 共享 cross-attention 意味着用同一个投影读同一个 context → 和独立 Transformer 无异，没有跨模态交互

为什么不共享 FFN：
- Video tokens 是高维视觉 patch（像素特征），action tokens 是低维运动向量（关节角度）
- 两种 representation 的分布、语义、需要的非线性变换完全不同
- 如果共享 FFN（完全共享 Transformer），模型被迫用同一套变换处理差异极大的数据 → 要么欠拟合视觉、要么欠拟合动作

MoT 的本质：
- 共享 self-attention = "允许对话，但各自独立思考"
- 这比完全独立（无交互）表达力强，又比完全共享（强制统一）更灵活

---

### Q4: 为什么 num_heads 和 attn_head_dim 必须相同？

**因为 Q/K/V 是物理上拼接在一起做 attention 的。**

MoT 的 attention 计算：
```python
Q = cat([Q_video, Q_action], dim=1)   # [B, Sv+Sa, num_heads * attn_head_dim]
K = cat([K_video, K_action], dim=1)
V = cat([V_video, V_action], dim=1)
# → reshape to [B, num_heads, Sv+Sa, attn_head_dim]
# → Q @ K^T → softmax → @ V
```

如果 num_heads 或 attn_head_dim 不同：
- 拼接后无法 reshape 成统一的 [B, H, S, D] 形状
- `Q_video @ K_action^T` 维度不匹配，矩阵乘法无法执行

但 hidden_dim 可以不同，因为：
- hidden_dim = num_heads × attn_head_dim × (fan_in/fan_out of Wq/Wk/Wv)
- 实际上 `Wq` 的输入是 hidden_dim，输出是 `num_heads * attn_head_dim`
- 只要输出维度（attn space）相同，输入维度（hidden space）可以不同

---

## Video Expert 设计

### Q5: 为什么第一帧不加噪 + timestep=0，而不是用 concat 或 cross-attention？

**三种 conditioning 方案对比：**

| 方案 | 做法 | 优点 | 缺点 |
|------|------|------|------|
| Concat | 干净帧 latent 拼接在 channel 维度 | 简单 | 需要改 patch embedding 的 in_channels；无法直接复用 Wan2.2 预训练权重 |
| Cross-attention | 图像编码为额外 context | 灵活 | 增加 cross-attention 计算量；图像信息和 text 竞争 attention 带宽 |
| **Fuse in latent (当前方案)** | 直接替换第一帧、用 t=0 告诉模型 | 零额外参数；完美复用 Wan2.2 预训练 | 需要 separated timestep 机制 |

当前方案的核心好处：
1. **完全复用 Wan2.2 TI2V 预训练权重** — Wan2.2 的 TI2V（text+image to video）就是这样做的（第一帧作为条件）。FastWAM 直接继承这个能力，不需要从头训练视频生成。
2. **零额外参数** — 不需要额外的 image encoder 或额外的 cross-attention 头。
3. **统一表示空间** — conditioning 帧和生成帧在同一个 latent space 里，模型可以自然地从 conditioning 帧"延伸"生成后续帧。

per-token timestep 是为此付出的"小代价"：让模型通过 t=0 的 modulation 学会"这帧不需要去噪、直接保留"。

---

### Q6: 为什么用 per-token timestep 而不是 binary mask + 统一 timestep？

**表达能力有本质区别。**

方案 A（当前）：每个 token 有独立的 shift/scale/gate
```
frame 0 tokens: modulate(x, shift(t=0), scale(t=0))  → 几乎恒等变换
frame 1 tokens: modulate(x, shift(t=350), scale(t=350)) → 适度去噪变换
```

方案 B（binary mask + 统一 timestep）：
```
所有 tokens: modulate(x, shift(t=350), scale(t=350))
frame 0: 额外乘以 mask=0 或 skip 某些操作
```

区别在于：
- **方案 A** 中，t=0 经过 MLP 映射后产生的 shift/scale/gate 是模型学到的"如何处理干净 token"的策略。这些值是连续函数 f(t=0) 的输出，模型可以学到丰富的行为（比如 gate 不一定是 0，可能是某个值让第一帧也做轻微变换来适应后续帧的表示空间）。
- **方案 B** 中，binary mask 只能粗暴地"跳过"或"保持"，没有中间态。而且如何与 AdaLN 的 6 个 modulation 向量交互需要手动设计。

另外，per-token timestep 自然支持更复杂的场景（比如未来如果每帧加不同程度的噪声做 progressive denoising）。

---

### Q7: Video cross-attention 里的 gt action 造成 train-infer gap？

**确实存在 gap，但通过架构设计最小化了影响。**

训练时：Video cross-attn 看 [text + gt_action]
推理时（action-only）：Video cross-attn 看 [text]（action=None）

为什么 gap 影响不大：
1. **推理时 video 的角色变了** — `infer_action` 时，video expert 只需要编码第一帧（t=0，无噪声），它不需要生成未来帧。它的任务退化为"把第一帧的视觉信息编码成好的 KV 供 action 读取"。这个任务不需要知道 gt action。
2. **第一帧不 attend to action（by design）** — 看 cross-attention mask，frame 0 的 tokens 对 action context 全部是 False。所以训练时第一帧就已经在不看 action 的条件下学习了，没有 gap。
3. **训练时 gt action 主要帮助后续帧** — 后续帧需要知道"往左还是往右"来生成正确画面。但推理 action-only 时，后续帧根本不存在（只有第一帧），所以这些帧的 action-context 逻辑不会被触发。
4. **如果需要 `infer_joint`（生成视频+动作）** — 此时 gt action 作为 conditioning 传入，所以 joint 推理也没有 gap。

---

### Q8: 为什么默认用 `first_frame_causal` 而不是 `bidirectional`？

**直接约束：`infer_action` 的 KV cache 等价性要求。**

![[Excalidraw/Fast-WAM-first-frame-causal-mask.excalidraw]]

如果用 `bidirectional`：
- 第一帧的 self-attention 输出 = f(frame0, frame1, frame2, ..., frameN)
- 即第一帧的 KV 依赖所有后续帧
- 推理 action-only 时只有第一帧 → 计算出的 KV ≠ 训练时的 KV → cache 不等价 → action 预测错误

`first_frame_causal` 保证：
- 第一帧的 self-attention 输出 = f(frame0 only)
- 不依赖后续帧 → 推理时单帧 prefill 产出的 KV 和训练时一致
- 后续帧（frame 1, 2, ...）依然可以看所有帧（包括 frame 0），不损失太多生成质量

**trade-off：**
- `bidirectional`：视频生成最好，但 action-only 推理无法使用
- `first_frame_causal`：视频生成略有损失（第一帧看不到未来），但 action-only 推理高效
- `per_frame_causal`：最强的因果性，视频生成质量最差，但支持 autoregressive 生成

---

## Action Expert 设计

### Q9: 为什么 Action Expert 的 timestep 是全局共享的？

**因为所有 action token 的噪声水平完全相同。**

Video 需要 per-token 是因为同一序列内有两种状态：
- Frame 0 tokens: t=0（干净）
- Frame 1-N tokens: t=采样值（有噪声）

Action 序列中：
- 所有 timestep 的 action 被统一加噪：`noisy_action = (1-σ)·action + σ·noise`
- σ 对所有时间步一样 → 所有 action token 处于相同的噪声水平
- 不需要区分处理 → 全局一组 modulation 足够

什么场景下 action 也需要 per-token：
- 如果做 progressive denoising（每步只去噪 action 的一部分时间步，类似 autoregressive）
- 如果 action horizon 中部分是已知的（比如前几步是已执行的历史动作，需要 t=0），类似 video 的第一帧处理

---

### Q10: 为什么 Action Expert 用 1D RoPE？

**RoPE 提供了"相对时间距离感知"，这对动作预测很关键。**

没有 RoPE 的话：
- 模型只知道"这是第 i 个 token"，但不知道 token i 和 token j 之间的时间距离关系
- 所有位置关系只能通过学到的绝对位置模式隐式表达

有 RoPE 的好处：
- `Q_i @ K_j` 的 attention score 自然包含 i-j 的相对距离信息
- 时间相近的 action step 之间的 attention 天然更强 → 平滑的运动轨迹
- 泛化到不同 horizon 长度（比如训练时 32 步，推理时 64 步）比绝对位置编码好

为什么不用 3D RoPE（像 video）：
- Action 只有时间维度，没有空间结构（每个时间步就是一个 7 维向量）
- 用 1D 已经编码了全部位置信息

如果去掉 RoPE：
- 模型需要完全从数据中学习位置关系，训练更困难
- 对未见过的 horizon 长度泛化能力变差
- attention pattern 会更均匀（远近不分），可能导致轨迹不够平滑

---

### Q11: 为什么从 Video Expert 初始化 Action Expert backbone？

**迁移的不是"视频理解能力"，而是"attention 和 FFN 的通用表示学习能力"。**

Video Expert 经过 Wan2.2 大规模预训练后，其 DiT blocks 学到了：
- 如何在 self-attention 中做高效的信息路由
- FFN 中的通用非线性变换模式
- Modulation 机制的使用方式（何时放大、何时抑制）

这些是**模态无关**的 Transformer 通用能力。类似于 NLP 中用语言模型初始化 code model，或用 ImageNet 预训练初始化检测模型。

`preprocess_action_dit_backbone.py` 的具体做法：
- 取 video expert 的 DiTBlock 参数（self-attn, cross-attn, FFN, norm, modulation）
- 对维度不匹配的层做线性插值（比如 hidden_dim 不同时）
- 跳过 `action_encoder` 和 `head`（这些是 action-specific 的，随机初始化）

为什么不完全随机初始化：
- Action Expert 要和 Video Expert 在同一个 attention 空间中交互
- 如果 Action Expert 完全随机，训练初期 Q_a 和 K_v 的 attention score 是纯随机的 → 学不到有效的跨模态对齐
- 从 Video Expert 初始化保证两个 expert 从相近的表示空间出发 → 更快收敛、更好的最终性能

---

## Flow Matching 与训练策略

### Q12: 为什么 video 和 action 用独立的 scheduler？

**两种信号的"去噪难度曲线"不同。**

Video latents：
- 高维（16 channels × 空间 × 时间），信息冗余大
- 视觉信号的低频成分占主导，高噪声时也能恢复大致结构
- 可能需要更高的 shift（偏向高噪声训练）来学好 global structure

Action sequences：
- 低维（7-dim × 时间步），信息密度高
- 每个维度都很关键（1mm 的误差可能导致抓取失败）
- 可能需要不同的 shift 来重点学好 low-noise regime 的精细去噪

如果共享 timestep：
- 必须用同一个 σ 加噪 video 和 action
- 但 σ=0.8 对 video 可能是"粗略轮廓已可见"，对 action 可能是"完全无法用的噪声"
- 两种模态对噪声的敏感度不同 → 共享 σ 会让某一方训练效率低下

独立 scheduler 还允许推理时使用不同步数（虽然当前实现强制同步了）。

---

### Q13: 为什么 target 是 v = ε - x₀？

**在 Rectified Flow（直线流）框架下，三种参数化的关系：**

```
加噪路径: x_t = (1-t)·x₀ + t·ε

速度场: v = dx/dt = ε - x₀

三种等价的预测目标:
  · 预测 v = ε - x₀ (velocity)     → 直接给出流的方向
  · 预测 ε = x₀ + v (noise)        → v = pred_ε - x₀
  · 预测 x₀ = ε - v (clean sample) → v = ε - pred_x₀
```

为什么选 velocity：
1. **数值稳定性** — 在 t≈0 时预测 ε 和预测 x₀ 的 SNR 差异极大（一个接近 +∞，一个接近 0），而 v 在整个 t 范围内相对均匀
2. **与 Euler step 的对应** — 去噪步 `x_{t-Δ} = x_t + v·Δσ`，velocity 预测直接对应一步积分，无需额外换算
3. **这是 Rectified Flow 的标准选择** — 原论文（Liu et al. 2022）证明直接学 velocity field 可以让采样路径更直，减少 ODE 求解的步数需求
4. **与 Wan2.2 一致** — FastWAM 继承 Wan2.2 的训练范式，保持一致减少 hyperparameter tuning

---

### Q14: 为什么 loss weight 是高斯形状？

```
weight(t) ∝ exp(-2·((t - T/2) / T)²)
```

**直觉：中间 timestep 信息量最大。**

两端发生了什么：
- **t≈T（高噪声端）**：输入几乎是纯噪声，模型的预测基本是"猜"全局分布的 mean → gradient 信号噪比很低，学到的主要是数据集的统计先验而非有用的生成能力
- **t≈0（低噪声端）**：输入已经很接近 clean sample，模型只需做很小的修正 → gradient 很小，但每步的误差对最终质量影响也小

中间 t 为什么最有价值：
- 模型需要同时利用"已有的信号"（partial structure）和"学到的先验"来填充缺失信息
- 这是真正需要"理解内容"的区域 — 不是纯猜，也不是纯 copy
- 训练效率最高：gradient 信号强且有意义

这个 weighting scheme 类似于 P2 weighting (Choi et al., 2022) 和 Min-SNR weighting 的思想，只是用了更简单的高斯形式。

---

### Q15: 为什么 video loss 跳过第一帧？

```python
if inputs["first_frame_latents"] is not None:
    pred_video = pred_video[:, :, 1:]    # 跳过第一帧的预测
    target_video = target_video[:, :, 1:]  # 跳过第一帧的目标
```

**原因：第一帧是恒等映射，对它算 loss 会产生误导信号。**

发生了什么：
- 第一帧 latent 没有加噪（直接替换为 clean latent）
- 第一帧的 timestep = 0
- 模型的 target = noise - sample，但这里 noise 和 sample 是什么？

由于第一帧没加噪：
- 加噪步 `x_t = (1-σ)·x₀ + σ·ε` 中 σ=0 → x_t = x₀
- target = ε - x₀，但这个 ε 是为这帧采样的独立噪声
- 第一帧的"正确预测"应该是 0 向量（因为不需要修改），但 target 计算公式给出的是随机噪声减 clean → 无意义

如果不跳过：
- 模型被迫对一个无意义的 target 优化 → gradient 是纯噪声 → 浪费训练容量、可能干扰其他帧的学习

---

## 推理效率

### Q16: 如果想让 action 看更多帧（比如前 3 帧）怎么改？

需要改两个地方 + 接受一个代价：

**修改 1：Attention mask**
```python
# 当前: action 只看第一帧 tokens
mask[video_seq_len:, :first_frame_tokens] = True

# 改为: action 看前 3 帧 tokens
mask[video_seq_len:, :3 * tokens_per_frame] = True
```

**修改 2：Prefill 逻辑**
- 当前只 prefill 1 帧 → 改为 prefill 3 帧
- 这 3 帧在训练时必须都是干净的（t=0），否则 cache 不等价
- 意味着 `first_frame_causal` 需要扩展为 `first_N_frames_causal`（前 3 帧互相可见但看不到后续帧）

**修改 3：Video attention mask**
- 当前 `first_frame_causal`：frame 0 看不到 frame 1+
- 需要改为：frame 0-2 互相可见，但看不到 frame 3+
- 否则前 3 帧的 KV 依赖后续帧 → 推理时只有 3 帧无法复现训练时的 KV

**代价：**
- 推理时需要 3 帧输入图像（而不是 1 帧）→ 可能需要短 buffer
- Prefill 计算量 ×3（但仍是常数，不随去噪步数增长）
- 训练时前 3 帧都不加噪 → video loss 有效监督的帧数减少 3

---

### Q17: 为什么用固定步数 Euler 而不是自适应 ODE solver？

**机器人实时控制对 latency 确定性的需求 > 精度。**

固定步数 Euler 的优势：
1. **确定性延迟** — 精确知道推理需要多少 forward pass（= num_inference_steps），可以保证控制频率
2. **无额外开销** — Dopri5 需要计算 error estimate（额外的 forward pass），还可能 reject step 并重试
3. **KV cache 友好** — 固定步数意味着可以精确预分配内存，不存在动态 allocation
4. **已经够了** — Flow Matching 的直线路径特性意味着 Euler 积分的误差远小于 DDPM 的 ancestral sampling。10-20 步通常足够

为什么 Dopri5 在这里不值得：
- 自适应步长的好处是"在困难区域用更多步"，但 FM 的路径本身很直，很少需要额外步数
- 自适应 solver 的最坏情况是不确定的 → 不适合实时控制
- Action 的维度很低（7 dim × 几十步），Euler 的积分误差对最终精度影响有限

---

### Q18: video 50 步、action 10 步的情况？

**当前实现确实强制同步，这是一个已知的局限。**

代码中（`infer_joint`）：
```python
for step_t_video, step_delta_video, step_t_action, step_delta_action in zip(
    infer_timesteps_video, infer_deltas_video,
    infer_timesteps_action, infer_deltas_action,
):
```

`zip` 意味着两个 schedule 必须长度相同。实际上 `num_inference_steps` 对两者都相同。

为什么这合理（短期内）：
- `infer_joint` 主要用于调试和可视化，不是部署路径
- 部署用 `infer_action`，那里只有 action scheduler，没有同步问题
- 20 步对两者都够用（action 甚至 10 步就不错，video 可能需要 25-50，但 20 是合理折中）

如果真的需要解耦：
- 可以改为 outer loop = max(steps_video, steps_action)
- 先完成步数少的那个（freeze 其 latent），继续跑步数多的
- 或者用"跳步"：video 每 5 步更新一次，action 每 2 步更新一次
- 但这增加实现复杂度，收益有限（因为部署时不需要 infer_joint）

---

## 数据与对齐

### Q19: 为什么下采样视频帧而不是下采样 action？

**两个独立的原因：**

**原因 1：VAE 的 temporal_downsample_factor 刚好是 4**

```
原始 33 帧 ──(每4帧取1帧)──→ 9 帧视频 ──(VAE temporal compress 4x)──→ ceil 映射
```

如果保持 33 帧视频 + VAE temporal compress 4x：
- VAE latent 时间维 = (33-1)/4 + 1 = 9 帧
- DiT 需要处理 9×(H/2)×(W/2) ≈ 9×56×112 = 56,448 tokens（224×448 video at patch 2×2）
- 计算量和显存爆炸

下采样后只有 9 帧：
- VAE latent 时间维 = (9-1)/4 + 1 = 3 帧
- DiT 处理 3×56×112 = 18,816 tokens → 内存友好 3x

**原因 2：Action 需要高频率**

机器人控制要求细粒度的动作（10-50Hz）。如果下采样 action：
- 32 步 → 8 步：每步动作跨度太大，抓取时精度不够
- 丢失了关键的 transition 信息（比如"先靠近→再抓→再提起"的三阶段在 8 步内可能被模糊化）

视频帧率可以低：
- 视觉信息变化相对缓慢（物体位置在 4 帧内变化不大）
- VAE 有时间压缩能力，可以从低帧率 latent 恢复视觉连续性
- 人眼/模型理解场景不需要 30fps

---

### Q20: 为什么用 delta action？

**Delta action 的分布特性更适合扩散模型预测。**

Absolute action 分布：
```
位置: [0.3, 0.5, 0.2, ...]   → 取决于工作空间的绝对坐标
每个样本差异大，分布多峰（不同任务的目标位置不同）
```

Delta action 分布：
```
位移: [0.01, -0.005, 0.002, ...]  → 每步的微小变化量
大部分时间接近 0，分布接近单峰高斯
```

为什么 delta 更好学：
1. **分布更集中** — delta action 的方差小得多，扩散模型更容易从噪声中恢复
2. **任务无关的先验** — "不动"（delta=0）是一个强先验，模型只需学"什么时候偏离 0 以及偏多少"
3. **泛化性** — 不同初始位置执行相同动作的 delta 是一样的；absolute action 则完全不同
4. **数值尺度友好** — delta 值通常在 [-0.05, 0.05] 范围内，和高斯噪声的尺度接近；absolute 值可能在 [0, 1] 范围且分布不均

唯一的 caveat：
- gripper 用绝对值（`delta_action_dim_mask` 最后一维是 False）
- 因为 gripper 是 binary 的（开/关），delta 没有意义

---

### Q21: group causal mask 的整除约束能否放宽？

**当前是硬约束（assert），但技术上可以放宽。**

为什么代码里用 assert：
```python
assert action_emb.shape[1] % num_temporal_groups == 0
```
- 如果不整除，每组 action 数量不同 → `create_group_causal_attn_mask` 的矩形块大小不一致 → 需要更复杂的 mask 构建逻辑
- 简化实现：所有组大小相同，mask 是规则的 block pattern

如果要放宽：
1. **Padding 方案** — 补齐到整除，用 `action_is_pad` 标记 padding 部分，loss 不计算
2. **不均匀分组** — 前 N-1 组大小为 `action_len // groups`，最后一组包含余数。mask 构建需要改为逐组循环而非简单 repeat
3. **放弃分组，用连续 mask** — 每个 video frame 看 action 的某个时间窗口（可以重叠），用 sliding window attention mask

实际上当前约束在实践中不是问题：
- `action_video_freq_ratio` 控制了 video 和 action 的频率比
- 数据采样时 `(num_frames - 1) // ratio` 就确保了整除
- 这是一个在数据预处理层面就能满足的约束

---

## 消融与 Trade-off

### Q22: 去掉 video expert 效果会差多少？

**这等于把 FastWAM 退化为一个简单的 Vision-Language-Action（VLA）模型。**

没有 video expert 的替代方案：
```
Image → frozen encoder (e.g., CLIP/DINOv2) → image tokens
Text → T5 → text tokens
Action expert: cross-attend to [image_tokens + text_tokens] → 预测 action
```

MoT 相比这个方案额外提供了什么：
1. **视频预训练的表示** — Video Expert 是 Wan2.2 预训练的，理解物体、运动、3D 空间关系，远比 CLIP/DINO 的静态图像理解丰富
2. **Action-aware 视觉特征** — 虽然 video 的 self-attention 看不到 action，但 video 是和 action 一起训练的（通过 video loss + cross-attn 里的 gt action），学到了"与动作预测相关的视觉特征"
3. **共享 attention 的深度交互** — 不是简单把 image feature 作为 context，而是在 30+ 层 Transformer 中逐层让 action 从 video KV 中提取信息，交互深度远超一次 cross-attention
4. **Video loss 作为辅助训练信号** — 见 Q23

---

### Q23: video loss 对 action 预测有什么作用？

**Video loss 是一个强大的辅助训练信号，通过 shared attention 间接提升 action 的视觉特征质量。**

如果 `loss_lambda_video=0`：
- Video expert 的参数只通过 MoT shared attention 的反向传播间接获得梯度
- 这些梯度来源：action loss → action tokens → Q_a @ K_v^T → ∂L/∂K_v → video expert 的 self-attn 参数
- 这条梯度路径很长且稀疏，不足以充分训练 video expert

有 video loss 的好处：
1. **直接监督 video expert 学好视觉表示** — video expert 必须理解场景才能生成正确的未来帧
2. **学到的 KV 更有信息量** — video loss 迫使 video expert 编码足够丰富的时空信息到 KV 中，action expert 通过 shared attention 免费获取这些信息
3. **正则化效果** — 防止 video expert 的参数退化为只服务于 action prediction 的"adapter"，保持通用的视觉理解能力
4. **世界模型效应** — 学会预测视觉后果 = 理解物理规律 → action prediction 质量间接受益

类比：就像 NLP 中多任务学习。翻译任务（主任务）+ 语言建模任务（辅助）一起训练，辅助任务帮助 encoder 学到更好的语言表示。

---

### Q24: 为什么 proprio 只是一个 Linear + concat 到 context？

**Proprio 的信息量太少，不值得复杂编码。**

Proprio 是什么：
```
[x, y, z, rx, ry, rz, gripper_left, gripper_right]  # 8 维向量
```

这就是一个瞬时状态描述（当前末端执行器在哪、手指张开多少）。相比 video（数十万维像素）和 action（数十维 × 数十步），proprio 的信息量极小。

为什么 Linear + 1 token 足够：
1. **维度太低** — 8 维映射到 4096 维 = 一个 token 就能容纳所有信息
2. **语义简单** — proprio 是精确的数值量（坐标、角度），不需要 attention 来"理解"
3. **使用方式单一** — proprio 对所有 token 等同可见（不需要分组、mask），放在 context 里让所有人都能读到就够了
4. **Baseline 验证** — 简单方案先跑通再说。如果实验证明 proprio 是瓶颈，才值得加复杂度

什么时候可能需要更复杂的编码：
- 如果 proprio 包含历史轨迹（多步状态）→ 可能值得用 attention 建模时序关系
- 如果 proprio 维度很高（比如全身 humanoid 的 50+ 关节）→ 可能需要分组或层次编码
- 如果 proprio 对任务影响很大但当前方案提取不够 → 实验说了算

当前场景（7-8 维桌面操作）下，一个 Linear token 是正确的工程选择：简单、有效、不增加额外的 debug 复杂度。