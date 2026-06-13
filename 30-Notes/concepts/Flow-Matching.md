---
title: "Flow Matching"
tags: [concept, generative-model, flow-matching, diffusion, continuous-normalizing-flow]
created: 2026-06-13
related:
  - "[[30-Notes/concepts/Diffusion-Policy]]"
  - "[[30-Notes/concepts/Diffusion-Transformer]]"
---

# Flow Matching（流匹配）

## 核心思想

把生成建模为**连续的概率流**：从噪声分布 $p_1 = \mathcal{N}(0, I)$ 到数据分布 $p_0 = p_{data}$，学一个 velocity field $v_\theta(x_t, t)$ 来描述这条流的方向。

与 DDPM 的关键区别：**路径是直线**，不是弯曲的随机扩散路径。

---

## 数学框架

### 加噪路径（Rectified Flow）

$$x_t = (1-t) \cdot x_0 + t \cdot \epsilon, \quad t \in [0, 1], \quad \epsilon \sim \mathcal{N}(0, I)$$

- $t=0$：干净样本 $x_0$
- $t=1$：纯噪声 $\epsilon$
- 中间：$x_0$ 和 $\epsilon$ 的线性插值

### Velocity Field

$$v = \frac{dx_t}{dt} = \epsilon - x_0$$

这就是模型要学的 target：给定 $x_t$ 和 $t$，预测流的方向。

### 训练目标

$$\mathcal{L}_{FM} = \mathbb{E}_{t, x_0, \epsilon}\left[\|v_\theta(x_t, t) - (\epsilon - x_0)\|^2\right]$$

### 推理（采样）

从 $x_1 \sim \mathcal{N}(0, I)$ 出发，用 ODE solver 积分：

$$x_{t-\Delta} = x_t - v_\theta(x_t, t) \cdot \Delta t$$

通常用 Euler 方法，10-20 步即可。

---

## 三种等价的预测 target

```
v = ε - x₀ (velocity)       ← Flow Matching 标准选择
ε = x₀ + v (noise)          ← DDPM 传统选择
x₀ = ε - v (clean sample)   ← 有时用于图像生成
```

为什么选 velocity：
1. 数值稳定性更好（在 t 的两端 SNR 更均匀）
2. 与 Euler step 直接对应，无需额外换算
3. 直线路径 → 更少的积分步数

---

## Flow Matching vs DDPM

| | DDPM | Flow Matching |
|---|---|---|
| 路径形状 | 弯曲（前向 SDE） | 直线（ODE） |
| 采样方式 | 离散 T 步去噪 | 连续 ODE 积分 |
| 典型步数 | 50-1000 步 | 10-20 步 |
| 数学框架 | 随机微分方程 (SDE) | 常微分方程 (ODE) |
| 噪声 schedule | 离散 $\beta_t$ 序列 | 连续 $t \in [0,1]$ |
| 推理速度 | 慢（步数多） | 快（步数少） |
| 训练 target | $\epsilon$（噪声） | $v = \epsilon - x_0$（速度） |

---

## 在机器人控制中的优势

为什么 Fast-WAM / DiT4DiT 都选 Flow Matching 而不是 DDPM：

1. **更少的推理步数** → 更低延迟（机器人实时控制 critical）
2. **确定性 ODE** → 给定初始噪声，输出确定 → 更稳定的动作
3. **连续时间** → 天然支持不同模态用不同 scheduler（Fast-WAM Q12）
4. **直线路径** → Euler 积分误差小，固定步数就够了（Fast-WAM Q17）

---

## Shifted Distribution（时间步偏移）

实践中不均匀采样 $t$，而是用偏移分布让中间 timestep 权重更大：

```python
# 高斯形状的 loss weight
weight(t) ∝ exp(-2 * ((t - 0.5) / 1.0)²)
```

原因：中间 $t$ 的训练信号最有效（Fast-WAM Q14）

---

## 代表方法

| 方法 | 年份 | 应用 |
|------|------|------|
| **Rectified Flow** (Liu et al.) | 2022 | 提出直线流框架 |
| **Flow Matching** (Lipman et al.) | 2022 | 条件流匹配理论 |
| **Stable Diffusion 3** | 2024 | MMDiT + flow matching |
| **Wan2.5** | 2025 | 视频生成 |
| **Fast-WAM** | 2026 | video + action 联合 flow matching |
| **DiT4DiT** | 2026 | dual flow matching (video/action 独立 scheduler) |

---

## 相关概念
- [[30-Notes/concepts/Diffusion-Policy]] — 扩散策略（可用 FM 加速）
- [[30-Notes/concepts/Diffusion-Transformer]] — DiT backbone
- [[20-Papers/WAM/2026-Fast-WAM]] — 应用 flow matching 的 WAM
- [[20-Papers/WAM/2026-DiT4DiT]] — dual flow matching
