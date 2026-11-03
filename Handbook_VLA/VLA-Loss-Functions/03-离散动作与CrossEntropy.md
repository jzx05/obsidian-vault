---
title: "03 · 离散动作与 Cross Entropy"
tags: [vla, loss, tokenization, cross-entropy]
created: 2026-11-03
source: "VLA-Handbook/theory/foundation/vla_loss_functions_handbook.md"
status: complete
---

# 03 · 离散动作与 Cross Entropy

> [!abstract] 一句话
> 把每个动作维度离散成 `N_bins` 个 bin，用每维分类的交叉熵训练。它天然能表达每维的多解，也天然兼容 LLM 的 token 接口；代价是量化误差、bin 边界抖动，以及维度间独立时联合分布会因子化。

## 1. 做法

代表路线：RT-1 / RT-2 风格。

- 动作 `a ∈ R^{D_a}`，每维归一化到固定范围（如 `[-1, 1]`）；
- 每维分成 `N_bins` 个 bin（常见 256）；
- 模型对每个 `(时间步, 维度)` 输出 `N_bins` 个 logits；
- 训练用交叉熵，推理用 argmax 或采样。

**张量**：

- `logits`: `[B, T, D_a, N_bins]`
- `target_bins`: `[B, T, D_a]`（每个元素是 0..N_bins-1 的整数）

## 2. 交叉熵

$$
\mathcal{L}_{\text{CE}} = -\sum \log p_\theta(y)
$$

对单个元素：

$$
\text{CE} = -\log \operatorname{softmax}(\text{logits})[y]
$$

**最小实现**：

```python
import torch.nn.functional as F

loss = F.cross_entropy(
    logits.view(-1, logits.size(-1)),
    target_bins.view(-1),
)
```

**这两行在做什么**：

- `logits.view(-1, N_bins)`：把 `[B, T, D_a, N_bins]` 压成 `[B*T*D_a, N_bins]`；
- `target_bins.view(-1)`：把 `[B, T, D_a]` 压成 `[B*T*D_a]`；
- `F.cross_entropy` 内部做 `log_softmax` + NLL，默认对所有元素取平均；
- 传入的必须是 **raw logits**，不要先 softmax。

## 3. 一个具体例子

设定：7 维动作、`T = 8`、`N_bins = 256`、`B = 1`。

```text
logits:      [1, 8, 7, 256]
target_bins: [1, 8, 7]
```

取第 0 步、第 0 维（Δx），归一化真值 `0.4`：

```text
bin 宽 = 2 / 256 = 0.0078125
bin 索引 = floor((0.4 + 1) / 2 × 256)
        = floor(179.2) = 179
```

所以 target 是 `179`。如果模型给 bin 179 的概率是 `0.7`：

```text
CE = -ln(0.7) ≈ 0.357
```

如果概率只有 `0.1`，`CE ≈ 2.303`。整个 batch 的损失是 `8 × 7 = 56` 个 CE 的平均。

**反量化**：

```text
预测 bin = 179
预测动作 = (179 + 0.5) / 256 × 2 - 1
        = 0.40234375
```

真实 `0.4`，量化误差约 `0.00234`（归一化单位）。如果 `[-1, 1]` 对应实际 `±5 cm`，bin 宽约 `0.39 mm`，这个例子的误差约 `0.12 mm`。

## 4. 多模态：能表达，但有前提

同一观测下有两类合理动作：

- 向左：Δx ≈ -0.4 → bin ≈ 76；
- 向右：Δx ≈ +0.4 → bin ≈ 179。

MSE 会预测 0 附近（中间），可能是无效动作。CE 可以输出：

```text
p[76]  ≈ 0.45
p[179] ≈ 0.45
其他    ≈ 0.10
```

argmax 选一个模式，采样可能选左也可能选右。这是“天然多模态”。

> [!warning] 维度独立时，联合分布会因子化
> 如果 7 个维度是并行独立分类的，联合分布是各维边缘分布的乘积，可能组合出“左手右转”这种训练里没出现过的动作。要建模维度间相关性，需要：
> - 自回归解码（每个维度条件于前面的维度）；
> - 或联合动作 tokenizer（如 OAT，见 [[20261028-Compression Gap]]）。

## 5. Bin 边界抖动

真实动作 `0.4` 靠近 bin 179 和 180 的边界（边界在 `0.40625`）。观测有一点噪声，模型可能从 179 翻到 180：

```text
bin 179 中心：0.40234
bin 180 中心：0.41016
跳变：0.0078125（归一化单位，约 0.39 mm）
```

动作输出是分段常数的，靠近边界时会左右跳，表现为抖动。

**缓解**：

- 增加 `N_bins`（减小 bin 宽，但分类更难、输出更大）；
- label smoothing（缓解过拟合/过度自信，但不能消除边界跳变）；
- 动作分块、滑动平均、低通滤波；
- 推理时用分布采样 + 平滑，而不是逐帧独立 argmax。

## 6. 超参与取舍

| 超参 | 常见值 | 影响 |
|---|---|---|
| `N_bins` | 256 | 越大精度越高，但输出/分类难度越大 |
| label smoothing | 0.0–0.1 | 缓解过拟合，可能轻微损失精度 |
| reduction | mean | 每个位置等权；也可按维度加权 |
| ignore_index | 视情况 | 忽略 padding 时间步 |

**优点**：

- 天然表达每维多解；
- 与 LLM 的 next-token 接口兼容；
- 训练稳定，实现简单；
- 推理可以 argmax（确定性）或采样（多样性）。

**缺点**：

- 量化误差，精密任务受限；
- bin 边界抖动；
- 维度独立时联合分布不准确；
- `N_bins` 大时输出头和内存开销增加；
- 信息容量有上限（见 [[20261028-Compression Gap]] 的 80 bits 分析）。

## 7. 相关笔记

- 连续多模态替代：[[04-GMM负对数似然]]
- 生成式替代：[[05-Diffusion与FlowMatching]]
- 信息瓶颈：[[20261028-Compression Gap]]
- 基础回归：[[01-行为克隆与回归损失]]
