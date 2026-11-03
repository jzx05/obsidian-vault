---
title: "02 · 序列与 Chunk 损失"
tags: [vla, loss, chunk, action-chunking]
created: 2026-11-03
source: "VLA-Handbook/theory/foundation/vla_loss_functions_handbook.md"
status: complete
---

# 02 · 序列与 Chunk 损失

> [!abstract] 一句话
> 动作 chunk 让模型一次预测未来 K 步，解决“观测频率低、控制频率高”的错配，并减少单步预测的误差累积；代价是远期标签更难学、分布漂移更敏感。

## 1. 为什么需要 chunk

典型 VLA / teleop 系统里：

- 视觉/触觉输入频率：10–30 Hz；
- 控制环频率：125–500 Hz（RTDE / servoJ）。

模型不可能每个控制周期都做一次大模型前向。常见做法：

```text
模型低频输出一段动作 chunk（K 步）
        ↓
控制端插值 / 滤波 / 保持
        ↓
高频命令流
```

好处：

- 降低策略推理频率；
- 一次预测多步，减少 compounding error；
- 给下游平滑/优化留出窗口。

## 2. Chunking BC

对 chunk 维度做 BC：

$$
\mathcal{L}_{\text{chunk}}=\frac{1}{BKD_a}\sum\|a_{t:t+K-1}-\hat a_{t:t+K-1}\|^2
$$

**张量**：

- `pred_action`: `[B, K, D_a]`
- `gt_action`: `[B, K, D_a]`

**关键超参**：

- `K`：chunk 长度。K 越大越能“看远”，但越难学、越怕分布漂移；K 越小越稳，但误差累积更快。
- `receding horizon`：推理只执行前 `M` 步（`M < K`），然后滚动重规划。常见 `M = 1~K/2`。

**实现**：

```python
# pred_chunk: [B, K, D_a], gt_chunk: [B, K, D_a]
loss = F.mse_loss(pred_chunk, gt_chunk)

# 推理：只执行前 M 步
action = pred_chunk[:, :M]
```

**K 的选择**：

| K | 优点 | 缺点 |
|---|---|---|
| 小（1–4） | 稳定、易学、响应快 | 误差累积、频繁重规划 |
| 中（8–16） | 平衡 | 需要调 RHC |
| 大（32–100） | 看得远、动作连贯 | 难学、远期标签噪声大、分布漂移敏感 |

## 3. 时间加权（越近越重要）

$$
\mathcal{L}=\sum_{k=0}^{K-1} w_k \cdot \ell(a_{t+k},\hat a_{t+k}),\quad w_0\ge w_1\ge\cdots
$$

**为什么**：

- 远期标签更不可靠（示教噪声、动力学误差、任务分支）；
- 近期动作对当前闭环更重要；
- 时间加权相当于给模型一个“先保证短期可控”的归纳偏置。

**常见权重**：

- 线性衰减：`w_k = 1 - k/K`；
- 指数衰减：`w_k = γ^k`；
- 只监督前 M 步：`w_k = 1 if k < M else 0`。

**注意**：权重过强会让模型忽略远期，导致动作短视；权重过弱则失去作用。通常先不加，确认远期标签质量后再加。

## 4. 工程细节

- **chunk 边界**：相邻 chunk 的重叠部分要做 temporal ensemble 或平滑，否则拼接处会跳变。
- **动作平滑**：chunk 输出后仍可能有抖动，见 [[07-安全正则与联合训练]]。
- **分布漂移**：K 越大，模型越容易在训练分布外犯错；receding horizon 是常用的缓解手段。
- **多模态**：chunk BC 的损失仍然可以是 MSE/GMM/Diffusion；chunk 只是“输出结构”，不解决多解问题。多解要换损失，见 [[04-GMM负对数似然]]、[[05-Diffusion与FlowMatching]]。

## 5. 相关笔记

- 基础回归：[[01-行为克隆与回归损失]]
- 生成式动作头：[[05-Diffusion与FlowMatching]]
- 经典实现：[[20261022-ACT]]（action chunking + temporal ensemble）
