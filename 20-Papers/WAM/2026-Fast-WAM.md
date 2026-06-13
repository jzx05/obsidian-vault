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