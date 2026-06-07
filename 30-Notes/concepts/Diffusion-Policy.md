---
title: "Diffusion Policy"
tags: [concept, policy, diffusion, robot-learning, imitation-learning]
created: 2026-06-07
related:
  - "[[30-Notes/concepts/World-Model]]"
  - "[[30-Notes/concepts/MBRL-World-Model]]"
  - "[[30-Notes/concepts/Generative-World-Model]]"
---

# Diffusion Policy（扩散策略）

## 核心动机

传统 policy 直接预测动作：`π(o) → a`，用 MSE loss 训练。

问题在于机器人操作数据天然**多模态**——同一观测下，专家可能有多种合理动作（比如绕过障碍物可以走左或右）。MSE 会把多峰分布**平均**掉，输出一个"哪条路都不走"的模糊中间值。

**解法**：把动作预测建模为**去噪过程**，直接建模完整的条件动作分布 $p(a|o)$。

---

## 训练 vs 推理

### 训练：有监督的去噪

```
步骤 1：取专家示范中的干净动作 a₀
步骤 2：随机选噪声程度 t ∈ {1,...,T}
步骤 3：按公式加噪
          ε ~ N(0, I)
          aₜ = √ᾱₜ · a₀ + √(1-ᾱₜ) · ε
          （t 越大噪声越多，t=T 时接近纯噪声）
步骤 4：让网络预测加进去的噪声
          ε̂ = ε_θ(aₜ, o, t)
步骤 5：Loss = ||ε - ε̂||²
```

网络学的不是"动作是什么"，而是"带噪动作里，噪声的方向是什么"。

### 推理：从噪声迭代还原

```
初始化：aT ~ N(0, I)   ← 与真实动作完全无关的随机噪声

重复 T 步：
  ε̂ = ε_θ(aₜ, o, t)           ← 预测当前噪声方向
  aₜ₋₁ = aₜ - α·ε̂ + 小扰动     ← 沿去噪方向走一步

最终输出：a₀  → 直接执行
```

### 训练 vs 推理 对比

| | 训练 | 推理 |
|---|---|---|
| **起点** | 干净动作 a₀（已知） | 随机噪声 aT（未知） |
| **方向** | a₀ → aT（加噪） | aT → a₀（去噪） |
| **噪声程度** | 随机采样单个 t | 从 T 到 1 依次走完 |
| **监督信号** | 真实噪声 ε（已知） | 无，靠学到的网络 |
| **步数** | 单步前向，轻量 | T 步迭代，**计算重** |

> 推理慢是核心痛点，实践中用 DDIM（10~20步）或 Consistency Models（1~4步）加速。

---

## 为什么适合机器人

| 问题 | 传统 MSE Policy | Diffusion Policy |
|------|---------------|-----------------|
| 多模态动作分布 | 平均化，输出模糊 | 能建模多峰分布 |
| 动作序列相关性 | 逐步预测，不连贯 | chunk 联合去噪，时序一致 |
| 表达能力 | 受限于网络输出维度 | 分布级别的表达 |

---

## Action Chunking（动作块预测）

Diffusion Policy 通常一次预测**一段动作序列**（chunk），而非单步动作：

```
观测 oₜ
    │
    ▼
[Diffusion Policy]
    │
    ▼
动作 chunk：[aₜ, aₜ₊₁, ..., aₜ₊H]   （H ≈ 16）
```

好处：
- 避免每帧都跑一次昂贵的去噪
- 强制动作在时间上连贯
- 减少高频抖动

---

## Policy Backbone 的选择

去噪网络 $\varepsilon_\theta$ 的骨干可以替换：

```
原版 Diffusion Policy（Chi et al. 2023）
  ├── CNN-based（适合低维状态）
  └── Transformer-based（适合图像输入）

接入 World Model Backbone 后（综述 §3.3）：
  ├── DreamVLA   — 生成式 WM 特征 + Diffusion action head
  ├── CoWVLA     — 同上
  └── UniVLA     — 统一 backbone，WM 预训练
```

> **Backbone** = 把原始输入（图像/语言/关节状态）转成特征向量的主干网络。
> 换 WM 做 backbone，相当于给 policy 换了一个"理解动作因果"的感知大脑，
> 而不只是"理解图像语义"的视觉脑。

---

## 在 WM 综述里的位置

Diffusion Policy 是 **Visuomotor Policy** 的代表方法（综述 §2.2 背景）——它是 WM 要去**增强**的下游 policy 之一。

```
§3 WM for Policy 的典型结构：

观测 o
  │
  ▼
[WM Backbone]  ← 预训练世界模型，提供时序动态特征
  │
  ▼
特征 z
  │
  ▼
[Diffusion Action Head]  ← Diffusion Policy 负责输出动作分布
  │
  ▼
动作 chunk a₀:H
```

---

## 代表方法

| 方法 | 年份 | 关键特点 |
|------|------|---------|
| **Diffusion Policy** (Chi et al.) | 2023 | 原版；CNN/Transformer 骨干 |
| **ACT** (Zhao et al.) | 2023 | 类似思想，CVAE 替代 diffusion；Action Chunking |
| **RDT-1B** | 2024 | Diffusion + 大规模预训练 |
| **DreamVLA** | 2025 | WM backbone + Diffusion head；综述 §3.3 |

---

## 相关概念
- [[30-Notes/concepts/World-Model]] — WM 的上位定义
- [[30-Notes/concepts/MBRL-World-Model]] — WM 作为 RL 训练场（§4）
- [[30-Notes/concepts/Generative-World-Model]] — WM 作为感知 backbone（§3.3）
- 综述位置：§2.2（背景）、§3.3（单骨干 VLA 的 action head）
