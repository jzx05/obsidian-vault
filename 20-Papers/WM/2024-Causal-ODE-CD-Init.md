---
title: "Stage 2b: Causal ODE / Causal CD Initialization for Few-Step AR Diffusion"
authors: []
year: 2024
venue: 
tags:
  - paper
  - video-diffusion
  - distillation
  - consistency
  - few-step
  - causal
status: reading
rating: 
date-read: 2026-06-05
url: 
pdf: 
priority: high
---

# Stage 2b — Causal ODE / Causal CD 初始化

> minWM Section 2.2，**Stage 2b**（论文 Page 3–4，公式 (2)）
> 对应引用：[24] Causal Forcing++ · Consistency Models · LCM

---

## TL;DR

> Stage 2a（Causal Forcing）产出的是**多步 AR 模型**（仍需 50 步去噪，28.6s）。
> Stage 2b 的目标：**把 50 步压成 4 步**，质量尽量保留。
> 两条路：Causal ODE（高质量，需离线数据）vs Causal CD（省资源，在线蒸馏）。

---

## 问题：多步 AR 仍然太慢

Stage 2a 结束后的延迟：

```
Wan2.1 多步 AR：28.6s   目标：1.1s   → 还差 26× 
HY1.5  多步 AR：81s     目标：3.4s   → 还差 24×
```

去噪步数（比如 50 步）是直接瓶颈。Stage 2b 核心：**把去噪步数从 ~50 压到 ~4**。

---

## 背景：扩散模型的 ODE 轨迹

扩散模型的反向去噪过程可以写成一条 ODE（常微分方程）：

```
dx/dt = f(x, t)   （连续形式）

离散化：
  x_T → x_{T-1} → x_{T-2} → ... → x_1 → x_0
         ← 50步，每步是一次 ODE 积分步 →
```

**ODE 轨迹** = 精确按照这条路径走的序列，计算代价高但质量好。

**一致性（Consistency）约束** = 对于同一条 ODE 轨迹上的任意两点，模型的预测应该一致：

```
G_θ(x_{t1}, t1) ≈ G_θ(x_{t2}, t2) ≈ x_0
（不同噪声水平，预测的干净帧应该相同）
```

---

## 路径 A：Causal ODE 初始化

### 操作流程

```
Step 1. 用 Stage 2a 的多步 AR 模型（50 步）跑大量视频
         记录完整去噪轨迹：[x_T, x_{T-1}, ..., x_0] × N 视频
         ↓ 写到磁盘（存储开销大）

Step 2. 用这些 ODE 轨迹训练少步学生模型
         目标：学生 4 步 ≈ 教师 50 步的质量
         ↓ 标准 ODE 蒸馏 loss

Step 3. 得到初始化好的少步 AR 模型
```

### 优缺点

| 优点 | 缺点 |
|------|------|
| 有精确轨迹作为监督，质量最好 | 离线生成轨迹：计算贵、存储大 |
| 蒸馏目标明确 | 需要完整 pipeline 两遍（先生成，再训练）|

---

## 路径 B：Causal CD 初始化（minWM 推荐）

### 核心 trick：单步 ODE 近似

不需要预存完整轨迹。训练时实时做**一步 ODE 近似**：

```
x_{s_i}   →（一步 ODE，用 AR teacher 算）→   x_{s_i^-}

x_{s_i^-} 是 x_{s_i} 沿 ODE 方向走一小步后的邻近点
计算开销 = 1次 forward pass，无需存储
```

### 关键公式（minWM 公式 (2)）

```
S* = argmax_{s_i} E[ w(s_i, v_i) · 
     || G_θ(x_{s_i},  s_i,  x^s_{0:n-1}, c)
      - G_θ(x_{s_i^-}, s_i^-, x^s_{0:n-1}, c) ||² - Δt ]

符号解释：
  G_θ             = 因果 AR 扩散模型（Stage 2a 产物）
  x_{s_i}         = 当前噪声水平 s_i 下的加噪帧
  x_{s_i^-}       = 用 AR teacher 单步 ODE 得到的邻近帧
                     （x_{s_i^-} ≈ x_{s_i} - Δt · f_teacher(x_{s_i}, s_i)）
  x^s_{0:n-1}     = 历史条件帧（因果 AR 特有的条件输入）
  c               = 文本/相机条件
  w(s_i, v_i)     = 时间步权重（v_i 是方差标量，加权不同噪声步的重要性）
  Δt              = 预定义范数下的距离项（防止两侧预测退化成同一个值）
```

### 直觉解释

```
"同一条去噪轨迹上相邻两点，
 模型对它们的预测（最终干净帧）应该一致"

        x_{s_i}   ──一步──→  x_{s_i^-}
           ↓                      ↓
       G_θ预测x_0          G_θ预测x_0
           └────── 应该相等 ──────┘
```

一致性约束强迫模型"一步到位" — 从任意噪声水平都能直接预测干净帧，自然就支持少步推理。

### 为什么叫 Causal CD（因果一致性蒸馏）

标准 CD 用双向 teacher 提供轨迹。Causal CD 的"因果"体现在：

```
G_θ(..., x^s_{0:n-1}, c)  ← 条件里包含历史干净帧（因果 AR 特有）

标准 CD 没有这个条件，Causal CD 在 CD 基础上加入了因果历史条件
```

---

## 两条路对比

| 维度 | Causal ODE | Causal CD |
|------|------------|-----------|
| **数据需求** | 需要离线预生成 50 步轨迹 | 训练时在线单步近似 |
| **存储开销** | 大（50步 × N帧 × N视频）| 接近零额外存储 |
| **质量** | ★★★★★ 最好 | ★★★★ 略低 |
| **计算成本** | 高 | 低 |
| **训练步数** | ~2k steps | ~2k steps |
| **minWM 推荐** | 高资源预算 | **低预算首选** ✓ |
| **引用来源** | [24] Causal Forcing++ | 类似 LCM / iCD 思想 |

---

## 在整体流水线中的位置

```
双向扩散大模型（Wan2.1 / HY1.5）
        ↓
  Stage 1：PRoPE 相机控制
        ↓
  Stage 2a：Causal Forcing → 多步 AR（28.6s / 81s）
        ↓
  Stage 2b：ODE 或 CD 初始化 ← 你在这里
        ↓  把 50 步 → 4 步，质量有损
  Stage 2c：Asymmetric DMD → 质量修复（1.1s / 3.4s）
```

**为什么需要 Stage 2c？**

Stage 2b 压步数后质量会下降（分布偏移）。Stage 2c 用双向 teacher 做 DMD 校正，把质量拉回来。

---

## 涉及的文献

### 直接引用

| 论文 | 角色 | minWM 编号 |
|------|------|-----------|
| **Causal Forcing++** | ODE 路径的直接来源，把 Causal Forcing 扩展到 few-step | [24] |

### 原始理论基础

| 论文 | 角色 |
|------|------|
| **Consistency Models**（Song et al. 2023）| 定义一致性约束的数学框架，CM 的 loss 和 Causal CD 直接对应 |
| **Improved Consistency Training (iCT)**（Song & Dhariwal 2023）| 修复了 CM 训练不稳定问题，Causal CD 的工程基础 |
| **Latent Consistency Models (LCM)**（Luo et al. 2023）| 把 CM 应用到 latent 扩散空间，和 minWM 的视频 latent 架构直接相关 |
| **Flow Matching**（Lipman et al. 2022）| ODE 蒸馏的理论背景，理解 ODE 轨迹蒸馏需要 |

### 延伸阅读

| 论文 | 角色 |
|------|------|
| **DMD（Distribution Matching Distillation）**[30] | Stage 2c 的算法基础，与 Stage 2b 衔接 |
| **Add-it / TCD** | 其他 consistency 蒸馏变种，对比视角 |

---

## 我的思考

- **CD 路径是工程上的聪明设计**：用在线单步近似替代昂贵的离线轨迹生成，让资源有限的团队也能跑完蒸馏流程
- **"初始化"这个词很关键**：Stage 2b 不是最终产物，它只是给 Stage 2c 提供一个"合理的起点"（少步但有偏差），Stage 2c 再做最后修正
- **步数压缩的极限**：4 步以下质量急剧下降（见 Table 1），这个数字和 Causal Forcing++ 的结论一致
- **Exposure Bias 问题**：Stage 2b 用的是 teacher 的 ODE 轨迹（完美历史条件），推理时用的是自己生成的历史（有误差），和 Stage 2a 有同样的 Exposure Bias 问题

---

## 引用与延伸阅读

- 应用：[[20-Papers/WM/2026-minWM-Real-Time-Video-World-Models]]（Section 2.2 Stage 2b，公式 2）
- 前置：[[20-Papers/WM/2024-Causal-Forcing]]（Stage 2a，生产多步 AR 模型）
- 后续：[[30-Notes/concepts/Asymmetric-DMD]]（Stage 2c，修正 Stage 2b 的质量偏差）
- 理论基础：Consistency Models (Song et al. 2023) — 待建笔记
- 直接引用：Causal Forcing++ [24] — 待建笔记
