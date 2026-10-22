---
title: "RT-1 — 开山之作"
tags: [handbook, vla, robot-transformer]
created: 2026-10-22
source: "https://arxiv.org/abs/2212.06817"
status: read
related:
  - "[[RT-2 — VLA 的诞生]]"
---

# RT-1 — 开山之作

> [!abstract] 核心思想
> 把机器人控制建模为 **Token 生成问题**：图像和语言进 Transformer，输出离散动作 token。

> [!tip] Eureka Moment
> 把机器人策略做成“小模型 + 少 token + 离散动作 + 并行解码”，就能在 3 Hz 的硬预算内吃下 700+ 任务；限制机器人 scaling 的往往不是模型不够大，而是推理预算和数据多样性。

## 🏗️ 架构与信息流

```text
6 × RGB (300×300)
      │
      ▼
EfficientNet-B3 (16M, ImageNet 预训练)
      ▲
      │ FiLM(USE(ℓ))  ← 自然语言指令
      ▼
81 tokens / frame
      │
      ▼
TokenLearner (K=8)  →  8 tokens / frame  →  6 帧拼接 = 48 tokens
      │
      ▼
decoder-only Transformer (8 层, 19M)
      │
      ▼
11 个并行分类头 × 256 bins
      │
      ▼
11 维动作  →  机器人 @ 3 Hz
```

### 系统组件对比

| 模块 | 输入 | 输出 | 频率/时序 | 训练/推理差异 |
|------|------|------|-----------|----------------|
| 图像编码器 | 6 帧 RGB，300×300 | 每帧 9×9×512 → 81 个视觉 token | 每个控制步 | ImageNet 预训练 EfficientNet-B3，16M 参数；推理时复用重叠窗口的 token |
| 语言条件 | 自然语言指令 ℓ | Universal Sentence Encoder 嵌入 | 每条指令一次 | 通过 identity-initialized FiLM 注入 EfficientNet |
| TokenLearner | 每帧 81 token | 每帧 8 token，6 帧共 48 token | 每个控制步 | 学习式空间注意力压缩；论文报告 2.4× 加速 |
| Transformer | 48 token + 位置编码 | 11 个动作 token 的 logits | 每步一次前向 | 8 层 decoder-only，19M 参数；非自回归并行输出 |
| 动作反 token 化 | 11 个 bin 索引 | 11 维连续/离散动作 | 3 Hz | 每个维度 256 bin，取 bin 内代表值 |
| 推理优化 | 重叠图像窗口 | 复用已算好的视觉 token | 每个控制步 | 论文报告 1.7× 加速 |

总参数：图像侧 16M + Transformer 19M = 35M。这个规模是刻意压出来的：论文把实时控制目标定为 3 Hz，模型推理预算 <100 ms，35M 是当时能在该预算内塞进最多容量的位置。

## 🔑 关键机制

- **动作即 token**：11 维动作（7 维机械臂 + 3 维底盘 + 1 维模式）各离散为 256 个 bin，模型并行输出 11 个动作 token。
- **TokenLearner 压缩**：每帧 81 个视觉 token 压到 8 个，6 帧共 48 个 token；论文报告 2.4× 推理加速。
- **并行解码**：11 个动作 token 一次输出，不做自回归；自回归版本推理慢 2× 以上，性能没有提升。
- **早期语言融合**：语言经 Universal Sentence Encoder 编码后，通过零初始化的 FiLM 层调制 EfficientNet，既注入指令又不破坏 ImageNet 预训练。

### 推理时：11 个头各选一个 bin

![[90-Attachments/Handbook_VLA/Pasted image 20261022113606.png]]

*每个动作维度从自己的 256 个 bin 里选一个；11 个头并行 argmax，得到 11 个动作值。第 11 维是 mode，决定执行机械臂、底盘还是终止。*

## 📊 关键数据

| 维度 | 数值 |
|---|---|
| 参数量 | 35M（16M 视觉 + 19M Transformer） |
| 训练数据 | 130k episodes、744 个任务、13 台机器人、17 个月 |
| 控制频率 | 3 Hz（模型推理 <100 ms） |
| 主要结果 | Seen 97%、Unseen 76%、干扰物 83%、换背景 59% |

## 🧮 补充：FiLM（Feature-wise Linear Modulation）

### 基本公式

设视觉编码器得到特征：

$$
x \in \mathbb{R}^{C \times H \times W}
$$

语言指令经过文本编码器得到语言向量：

$$
l = \text{TextEncoder}(\text{instruction})
$$

FiLM 根据语言向量生成两个参数：

$$
\gamma(l), \beta(l) \in \mathbb{R}^{C}
$$

对每个视觉通道做调制：

$$
\text{FiLM}(x,l) = \gamma(l) \odot x + \beta(l)
$$

### 符号说明

| 符号 | 含义 |
|---|---|
| $x$ | 视觉特征图 |
| $l$ | 语言指令向量 |
| $\gamma(l)$ | 逐通道缩放参数 |
| $\beta(l)$ | 逐通道偏移参数 |
| $\odot$ | 逐元素乘法 |

### 直觉

FiLM 让语言在视觉编码的早期就参与特征选择：$\gamma$ 放大与任务相关的通道，$\beta$ 调整基线。相比把语言 token 拼到 Transformer 输入，这种早期融合更省 token，也让 CNN 提取的特征直接任务相关。

## 🧩 局限与失败模式

- **纯模仿学习**：策略性能受演示者水平限制，不能通过试错超过专家。
- **泛化是重组合**：新指令主要是已见概念的新组合；训练集中没有的新运动学不会。
- **单一本体**：11 维动作空间绑定 Everyday Robots；换机器人需重定义动作接口并重训。
- **无 action chunk**：每步独立推理，长程任务依赖 SayCan 等高层规划器拆解。

## 🔗 关联

- 后续：[[RT-2 — VLA 的诞生]]
- 相关概念：[[TokenLearner]]、[[FiLM]]、[[Action Tokenization]]
- 论文笔记：[[RT-1 — Robotics Transformer]]

## 📚 参考

- 论文：[RT-1: Robotics Transformer for Real-World Control at Scale](https://arxiv.org/abs/2212.06817)（RSS 2023）
- 项目页：[robotics-transformer1.github.io](https://robotics-transformer1.github.io/)
- 代码/数据：论文未发布官方代码
