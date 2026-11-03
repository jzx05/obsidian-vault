---
title: "05 · Diffusion 与 Flow Matching"
tags: [vla, loss, diffusion, flow-matching]
created: 2026-11-03
source: "VLA-Handbook/theory/foundation/vla_loss_functions_handbook.md"
status: complete
---

# 05 · Diffusion 与 Flow Matching

> [!abstract] 一句话
> Diffusion 学“给定加噪动作，预测噪声”；Flow Matching 学“给定中间动作，预测速度场”。两者都能建模多模态动作分布，Diffusion 质量高但采样慢，Flow Matching 用更直的路经换更少步数，适合高频部署。

## 1. Diffusion Policy：条件去噪损失

### 1.1 前向过程

$$
x_t=\sqrt{\bar\alpha_t}\,x_0+\sqrt{1-\bar\alpha_t}\,\epsilon,
\qquad \epsilon\sim\mathcal{N}(0,I)
$$

- `x_0`：真实动作 chunk；
- `x_t`：加噪后的动作；
- `\bar\alpha_t`：噪声 schedule 的累积乘积；
- `t`：随机时间步。

### 1.2 训练损失（ε-prediction）

$$
\mathcal{L}_{\text{diff}}
=\mathbb{E}_{t,x_0,\epsilon}\left\|\epsilon-\epsilon_\theta(x_t,t,c)\right\|^2
$$

`c` 是条件（观测、语言指令、 proprio 等）。

**张量**：

- `x0`（动作 chunk）：`[B, K, D_a]`
- `t`：`[B]`
- `eps`：`[B, K, D_a]`
- `eps_pred`：`[B, K, D_a]`

**最小实现**：

```python
t = torch.randint(0, num_timesteps, (B,), device=device)
eps = torch.randn_like(x0)
x_t = sqrt_alpha_bar[t] * x0 + sqrt_one_minus_alpha_bar[t] * eps
eps_pred = model(x_t, t, condition)
loss = F.mse_loss(eps_pred, eps)
```

### 1.3 采样

从 `x_T ~ N(0,I)` 开始，迭代去噪：

```text
x_{t-1} = 1/√α_t · (x_t - (1-α_t)/√(1-ᾱ_t) · ε_θ(x_t,t,c)) + σ_t z
```

推理可以用 DDPM（随机、步数多）或 DDIM（确定性、步数少）。

### 1.4 超参与工程点

- **噪声 schedule**：cosine / linear；影响训练稳定性和样本质量。
- **训练时间步**：通常 1000；推理步数可独立减少。
- **推理步数**：DDIM 20–50 步常见；DPM-Solver 可更少。
- **动作 chunk**：Diffusion Policy 通常一次去噪一段动作，配合 receding horizon 执行。
- **条件注入**：FiLM、cross-attention、DiT 的 adaLN 等。

### 1.5 优缺点

**优点**：

- 能建模高精度、多模态轨迹；
- 训练稳定，比 GAN 容易；
- 对动作块的整体分布建模，避免逐维独立。

**缺点**：

- 推理多步，延迟高；
- 高频控制（50Hz+）需要少步采样或蒸馏；
- 训练/调参成本高于 BC/GMM。

## 2. Flow Matching / Flow Policy：向量场损失

### 2.1 基本形式

学习一个向量场 `v_θ(x,t,c)`，用 MSE 拟合目标速度：

$$
\mathcal{L}_{\text{flow}}
=\mathbb{E}\left\|v_\theta(x_t,t,c)-v^\ast(x_t,t)\right\|^2
$$

**条件路径**（直线插值）：

$$
x_t=(1-t)x_0+t x_1,
\qquad v^\ast=x_1-x_0
$$

- `x_0`：噪声；
- `x_1`：真实动作；
- `t ∈ [0,1]`；
- `v^\ast`：从噪声指向数据的直线速度。

**采样**：解 ODE

$$
\frac{dx}{dt}=v_\theta(x_t,t,c),
\qquad x_0\sim\mathcal{N}(0,I)
$$

从 `t=0` 积分到 `t=1`，得到动作。

### 2.2 张量

- `x0`：`[B, K, D_a]`（噪声）
- `x1`：`[B, K, D_a]`（真实动作）
- `t`：`[B]`
- `v_target`：`[B, K, D_a]`
- `v_pred`：`[B, K, D_a]`

**最小实现**：

```python
t = torch.rand(B, device=device)
x_t = (1 - t)[:, None, None] * x0 + t[:, None, None] * x1
v_target = x1 - x0
v_pred = model(x_t, t, condition)
loss = F.mse_loss(v_pred, v_target)
```

### 2.3 目标速度的定义不止一种

不同论文/实现的 `v*` 不同：

- **OT-CFM**：条件最优传输路径；
- **Rectified Flow**：用直线路径，reflow 进一步拉直；
- **Stochastic Interpolants**：更一般的插值框架。

工程上可以把它当作“diffusion 的少步替代”：训练目标简单，采样是 ODE，步数可以更少。

### 2.4 优缺点

**优点**：

- 采样步数少，适合高频；
- 训练目标直接、稳定；
- 直线路径容易积分；
- 与 diffusion 共享很多工程经验（chunk、条件注入、RHC）。

**缺点**：

- 目标速度的定义依赖路径设计；
- 少步采样可能损失多样性/质量；
- 复杂分布仍可能需要多步或更强模型。

## 3. Diffusion vs Flow Matching

| 维度 | Diffusion | Flow Matching |
|---|---|---|
| 学什么 | 噪声 ε / score | 速度场 v |
| 训练目标 | ε-prediction MSE | 速度 MSE |
| 采样 | 反向扩散，随机/确定 | ODE 积分 |
| 步数 | 多（DDPM），可 DDIM 减少 | 通常更少 |
| 路径 | 噪声 schedule 决定 | 可设计为直线 |
| 多模态 | 强 | 强 |
| 高频部署 | 需少步/蒸馏 | 更友好 |
| 工程成熟度 | 高 | 快速增长 |

## 4. 在 VLA 里的挂载

- action head 是去噪网络（U-Net / Transformer / DiT）；
- 条件 `c` 来自视觉-语言编码器；
- 输出动作 chunk，配合 receding horizon；
- 训练损失只作用于动作头，视觉/语言主干可冻结或 LoRA。

## 5. 相关笔记

- 离散对照：[[03-离散动作与CrossEntropy]]
- GMM 对照：[[04-GMM负对数似然]]
- chunk 结构：[[02-序列与Chunk损失]]
- 信息瓶颈与连续/离散：[[20261028-Compression Gap]]
- 经典 chunk：[[20261022-ACT]]
