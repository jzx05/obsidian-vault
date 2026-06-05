---
title: "Causal Forcing: Bridging Bidirectional and Autoregressive Video Diffusion"
authors: []
year: 2024
venue: arXiv 2024
tags:
  - paper
  - video-diffusion
  - autoregressive
  - causal
  - distillation
status: reading
rating: 
date-read: 2026-06-05
url: 
pdf: 
priority: high
---

# Causal Forcing — Bridging Bidirectional and Autoregressive Video Diffusion

> minWM Section 2.2, **Stage 2a** 的核心技术（论文引用 [23]）
> 位置：minWM 论文 Page 3–4，Section 2.2 第一段，公式 (1)

---

## TL;DR

> 用一个**最小改动**把双向视频扩散模型变成因果 AR 模型：
> 训练时给模型戴上因果 attention mask，同时把历史干净帧作为条件输入。
> 推理时滑动窗口逐段生成，保留多步去噪，质量接近双向。

---

## 问题与动机

### 双向模型为什么不能直接用于实时？

双向视频扩散（Wan2.1、HY1.5）训练时每帧能看到所有其他帧：

```
去噪第 3 帧时：
  f1 ←─┐
  f2 ←─┤
  f3 ◀─┼── 全对全 attention（无 mask）
  f4 ←─┤
  f5 ←─┘
```

推理时必须等**所有帧同时完成**才能输出 → 269s / 771s 首帧延迟 → 无法实时。

### 直接改成 AR（next-frame prediction）的问题

最简单的 AR 做法：训练一个因果 mask 的 next-frame 模型。
但这有两个严重问题：

| 问题 | 原因 |
|------|------|
| 质量大幅下降 | 单步预测，信息不足 |
| 预训练权重浪费 | 和原双向模型差距太大，fine-tune 困难 |

**Causal Forcing 的思路**：最小改动——只改 attention mask，保留扩散去噪过程。

---

## 核心方法

### Step 1：把双向 attention 换成因果 attention mask

训练时把原来的全对全 attention 替换为**下三角 mask**：

```
双向（原始）：              因果（Causal Forcing）：
f1 ↔ f2 ↔ f3 ↔ f4 ↔ f5    f1 → f2 → f3 → f4 → f5
每帧看所有帧               每帧只看自己和历史帧
```

在 attention 实现里，只需把 `attn_mask` 从 `None` 改为下三角矩阵。

### Step 2：历史干净帧作为条件（Teacher Forcing）

训练时，用**真实的历史帧（ground truth）**作为 condition 输入：

```
去噪当前窗口（帧 n+1 ~ n+W）时：
  条件输入  = 真实干净帧 x^s_{1:n}  ← ground truth 历史
  待去噪   = 加噪后的当前窗口 x^noisy_{n+1:n+W}
```

**关键公式（minWM 论文公式 1）**：

```
x* = argmin_{x_0} || G_θ(x_t, t, c, x^s_{0:n-1}) - x_0 ||²

其中：
  G_θ     = 扩散模型（预测干净帧）
  x_t     = 当前窗口加噪后的帧
  t       = 噪声时间步
  c       = 文本/相机条件
  x^s_{0:n-1} = 前 n 帧干净帧（teacher forcing 的"答案"）
```

这就是"Teacher Forcing"的名字来源：训练时喂真实答案，而不是模型自己的预测。

### Step 3：推理时滑动窗口

训练完成后，推理时用**自己生成的帧**替代 ground truth（因为没有 GT 了）：

```
推理滑动窗口（窗口大小 W = 4 帧）：

Window 1：条件 = [] → 生成 [f1, f2, f3, f4]
Window 2：条件 = [f1, f2, f3, f4] → 生成 [f5, f6, f7, f8]
Window 3：条件 = [f5, f6, f7, f8] → 生成 [f9, f10, f11, f12]
...
```

每个窗口独立做多步扩散去噪（仍是多步，比如 50 步），然后整个窗口一起输出。

---

## Causal Forcing 的两个局限（Stage 2a 结束后仍存在）

这正是为什么 minWM 还需要 Stage 2b 和 2c：

### 局限 1：仍需多步去噪 → 延迟高

```
Stage 2a 之后：
Wan2.1: 多步 AR → 首帧延迟 28.6s（从 269s 降了，但还不够）
HY1.5:  多步 AR → 首帧延迟 81s（从 771s 降了，但还不够）

目标：1.1s / 3.4s（需要 Stage 2b/2c 的蒸馏压缩步数）
```

### 局限 2：训练-测试分布偏移（Exposure Bias）

训练时条件 = ground truth 历史帧（完美）
推理时条件 = 自己生成的历史帧（有误差）

→ 误差随时间**累积**，长视频质量下降
→ 这是 AR 模型的经典问题，Self-Forcing 等方法专门解决这个

---

## 在 minWM 流水线中的位置

```
双向扩散大模型（Wan2.1 / HY1.5）
        ↓
  Stage 1：PRoPE 相机控制训练（Section 2.1）
        ↓
  Stage 2a：Causal Forcing ← 你在这里
        ↓ 把双向变成多步 AR
  Stage 2b：Causal ODE / CD 初始化（压缩到少步）
        ↓
  Stage 2c：Asymmetric DMD（修正分布偏移）
        ↓
  最终：few-step AR，1.1s 首帧延迟
```

---

## 训练超参数（来自 minWM 消融）

| 参数 | 值 | 说明 |
|------|---|------|
| 训练步数（Stage 2a） | 4k–6k steps | |
| 批量大小 | 4–16 | bs 越大越稳定 |
| 学习率 | 1×10⁻⁴ 附近 | 标准 fine-tune 范围 |
| 窗口大小 W | 论文未明确给出 | 推测 4–8 帧 |

---

## 和相关方法的对比

| 方法 | 核心思路 | 和 Causal Forcing 的关系 |
|------|---------|------------------------|
| **Causal Forcing（本文）** | 双向→因果，多步 AR | — |
| **Causal Forcing++**（引用 [24]）| 进一步支持 few-step | Causal Forcing 的扩展 |
| **Self-Forcing**（引用 [22]）| 解决 exposure bias | 解决 Causal Forcing 的局限 2 |
| **Teacher Forcing（NLP）** | 训练时喂 GT，推理时不喂 | Causal Forcing 借用了这个思想 |

---

## 我的思考

- **改动极小是最大优点**：只改 attention mask + 加条件输入，几乎保留全部预训练权重
- **多步 AR 仍然慢**：Stage 2a 本身不解决延迟问题，只是"让模型能因果生成"的必要前置步骤
- **窗口大小的取舍**：W 大 → 每窗口质量好，但延迟高；W 小 → 延迟低，但上下文少
- **Exposure Bias 是实际部署的大问题**：minWM 的 Asymmetric DMD（Stage 2c）间接缓解了这个问题，但没有正面解决

---

## 引用与延伸阅读

- 应用：[[20-Papers/WM/2026-minWM-Real-Time-Video-World-Models]]（Section 2.2 Stage 2a）
- 概念笔记：[[30-Notes/concepts/Causal-Forcing]]
- 后续改进：Causal Forcing++（引用 [24]，待读）
- 解决 exposure bias：[[20-Papers/WM/2024-Self-Forcing]]
- 下一步蒸馏：[[30-Notes/concepts/Asymmetric-DMD]]
