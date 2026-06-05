---
title: "Asymmetric DMD"
tags: [note, concept, distillation, diffusion]
created: 2026-06-05
related:
  - "[[20-Papers/WM/2026-minWM-Real-Time-Video-World-Models]]"
---

# Asymmetric DMD（非对称分布匹配蒸馏）

## 定义
> DMD（Distribution Matching Distillation）的非对称变种 — **teacher 是双向扩散模型，student 是因果（AR）扩散模型**。用双向 teacher 的"全局信息优势"修正 AR student 的分布漂移。

## 为什么重要
- 把多步扩散（几十步）压到 **few-step（4 步）**，是实时推理的关键
- 标准 DMD 假设 teacher 和 student 同结构 → AR vs 双向场景不直接适用
- minWM 的核心创新：**结构非对称**也能蒸馏成功

## 核心要点
- **目标**：min KL(p_data ‖ p_AR_student)
- **梯度形式**：
  ```
  ∇θ L = E[ (s_real(x_t) - s_AR(x_t)) · ∇θ G_θ(noise) ]
  ```
- **非对称点**：
  - s_real 来自**双向 teacher**（信息更全）
  - s_AR 来自**因果 student**（推理更快）
  - 用 teacher 的梯度修 student 的偏差
- **训练效率**：minWM 只需 200 steps Stage 2c 就能收敛

## 与标准 DMD 对比
| 维度 | DMD | Asymmetric DMD |
|------|-----|----------------|
| Teacher | 多步扩散 | **双向**多步扩散 |
| Student | 少步扩散 | **因果**少步扩散 |
| 应用场景 | 图像 / 双向视频 | 实时 AR 视频 |

## 例子
- minWM Stage 2c：把 Stage 2b 初始化的 few-step AR student 微调到接近 teacher 质量
- 加速效果：HY1.5 从 81s → 3.4s（保持可控性）

## 关联
- 来源论文：[[20-Papers/WM/2026-minWM-Real-Time-Video-World-Models]]
- 引用 [30]：原始 DMD 论文（待读）
- 相关概念：[[30-Notes/concepts/Causal-Forcing]]、[[30-Notes/methods/Few-Step-AR-Distillation]]
