---
title: "04 · GMM 负对数似然"
tags: [vla, loss, gmm, multimodal]
created: 2026-11-03
source: "VLA-Handbook/theory/foundation/vla_loss_functions_handbook.md"
status: complete
---

# 04 · GMM 负对数似然

> [!abstract] 一句话
> 用 K 个高斯分量的混合密度建模动作分布，负对数似然让真实动作在“某个模式”上获得高密度。它解决 BC 的平均化问题，又比 diffusion 轻量；代价是分量数、协方差和坍缩问题。

## 1. 公式

对动作 `a ∈ R^{D_a}`，条件于观测 `o`：

$$
p(a \mid o)=\sum_{k=1}^{K}\pi_k(o)\,\mathcal{N}\!\left(a;\mu_k(o),\Sigma_k(o)\right)
$$

负对数似然：

$$
\mathcal{L}_{\text{GMM}}=-\log p(a\mid o)
=-\log\sum_{k=1}^{K}\pi_k\,\mathcal{N}(a;\mu_k,\Sigma_k)
$$

实际计算用 log-sum-exp：

$$
\mathcal{L}_{\text{GMM}}
=-\operatorname{logsumexp}_k\left[\log\pi_k+\log\mathcal{N}(a;\mu_k,\Sigma_k)\right]
$$

对角高斯的 log 密度：

$$
\log\mathcal{N}(a;\mu_k,\Sigma_k)
=-\frac12\sum_{d}\left[\log(2\pi)+2\log\sigma_{k,d}
+\frac{(a_d-\mu_{k,d})^2}{\sigma_{k,d}^2}\right]
$$

## 2. 张量

常见对角协方差：

- `logits_pi`: `[B, T, K]` → softmax → `π`
- `mu`: `[B, T, K, D_a]`
- `log_std`: `[B, T, K, D_a]` → exp → `σ`

**参数量**：

- 对角：`K × (1 + 2D_a)`；
- 全协方差：`K × (1 + D_a + D_a(D_a+1)/2)`，高维动作通常不用。

## 3. 最小实现

```python
import math
import torch
import torch.nn.functional as F

# pi_logits: [B, T, K]
# mu:        [B, T, K, D]
# log_std:   [B, T, K, D]
# action:    [B, T, D]

log_pi = F.log_softmax(pi_logits, dim=-1)          # [B,T,K]
sigma = torch.exp(log_std).clamp_min(1e-6)          # [B,T,K,D]

log_norm = -0.5 * (
    ((action.unsqueeze(2) - mu) / sigma).pow(2)
    + 2 * log_std
    + math.log(2 * math.pi)
).sum(-1)                                           # [B,T,K]

log_prob = torch.logsumexp(log_pi + log_norm, dim=-1)  # [B,T]
loss = -log_prob.mean()
```

## 4. 推断

训练完得到 `π, μ, Σ` 后：

1. **采样分量 + 采样动作**：`k ~ Categorical(π)`，`a ~ N(μ_k, Σ_k)`。适合需要多样性。
2. **选最可能分量 + 取均值**：`k = argmax π_k`，`a = μ_k`。简单、确定性，但不一定是全局众数。
3. **混合均值**：`a = Σ π_k μ_k`。多模态时可能落在两个模式中间，控制任务通常不要用。

机器人控制常用 1 或 2，再配合动作分块、时间平滑，避免每步采样到不同模式导致抖动。

## 5. 一个多模态例子

一维动作，两个模式：

```text
K = 2
π = [0.5, 0.5]
μ = [-0.4, +0.4]
σ = 0.05
```

- 真实动作 `a = 0.4`：第二个分量密度很高，`p ≈ 3.99`，`NLL ≈ -1.38`；
- 真实动作 `a = 0`：两个分量都离得很远，`p ≈ 1e-13`，`NLL ≈ 29.9`。

连续分布的 NLL 可以为负，因为它比较的是密度，不是离散概率；看相对大小即可。MSE 会把两个模式平均到 0，而 GMM 的 NLL 会强烈惩罚 0 附近的预测。

## 6. 优缺点

**优点**：

- 能表达多解；
- 推理快（单次前向 + 采样/选分量）；
- 显式似然，可解释；
- 比 diffusion 轻量，适合作为强基线。

**缺点**：

- 高维动作下训练不稳；
- 分量坍缩（只用到一个高斯）；
- `K` 难选；
- 高斯假设限制了复杂分布；
- 全协方差参数 `O(D_a²)`，动作块高维时不可行。

## 7. 常见坑

| 问题 | 现象 | 缓解 |
|---|---|---|
| 方差坍缩 | `σ → 0`，NLL 趋向负无穷 | 方差下限、`log_std` 裁剪、正则化 |
| 分量坍缩 | 所有分量学成同一模式 | 初始化、熵/排斥正则、调整 K |
| K 太大 | 过拟合、空分量 | 验证集 NLL + 下游成功率 |
| K 太小 | 抓不住多模态 | 看数据分布，逐步增大 |
| 高维协方差 | 参数爆炸、矩阵求逆不稳 | 对角、低秩、共享协方差 |
| 数值下溢 | `exp` 后为 0 | 用 `logsumexp`，不要直接 `exp` |

## 8. 和其他动作损失对比

| 方法 | 输出 | 多模态 | 精度 | 采样速度 | 主要问题 |
|---|---|---|---|---|---|
| MSE | 一个点 | 否，会平均 | 高 | 快 | 多模态时平均出无效动作 |
| CE bins | 每维离散分布 | 是（每维） | 受 bin 宽限制 | 快 | bin 边界跳变、维度独立 |
| GMM NLL | K 个高斯参数 | 是（K 个模式） | 连续 | 快 | 方差/分量坍缩，K 难选 |
| Diffusion | 隐式分布 | 是 | 高 | 慢（多步） | 采样慢、训练复杂 |

## 9. 相关笔记

- 离散对照：[[03-离散动作与CrossEntropy]]
- 生成式替代：[[05-Diffusion与FlowMatching]]
- 回归基线：[[01-行为克隆与回归损失]]
- 多模态与信息瓶颈：[[20261028-Compression Gap]]
