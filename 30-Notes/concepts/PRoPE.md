---
title: "PRoPE (Projective Rotary Position Embedding)"
tags: [note, concept, attention, camera-control]
created: 2026-06-05
related:
  - "[[20-Papers/WM/2026-minWM-Real-Time-Video-World-Models]]"
---

# PRoPE — Projective Rotary Position Embedding

## 定义
> 一种把**摄像机内外参 (K, T)** 注入 self-attention 的位置编码方法，让视频扩散模型理解"从哪个视角、什么相机看"。

## 为什么重要
- 文本 prompt 无法精确指定**相机轨迹** — "向右平移 30°"很难用文字约束
- PRoPE 把几何信息直接注入 attention → 模型可以学到**视角一致性**
- 是世界模型 / NeRF 风格生成 / 自动驾驶视频生成的关键技术

## 核心要点
- **几何洞察**：注意力交互只依赖**相对**摄像机位姿，不依赖绝对位置
- **构造**：用 PRoPE 把每个 token 的相机参数变成块对角变换矩阵 P_i
- **注入方式**：A_ij = (P_i Q_i)^T (P_j K_j)，类似 RoPE 但用相机投影矩阵
- **优势**：保留预训练模型生成质量，只通过 attention 注入控制信号

## 数学直觉
```
原 RoPE：A_ij 依赖位置差 (i - j)
PRoPE ：A_ij 依赖相对相机位姿 T_i^{-1} T_j（保持平移不变性）
```

## 例子
- minWM Stage 1：用 PRoPE 让 Wan2.1 / HY1.5 学会响应相机轨迹
- 训练数据：DL3DV（重建获得 ground-truth 相机）+ WorldPlay 生成

## 与其他方法对比
| 方法 | 控制粒度 | 生成质量保留 |
|------|---------|------------|
| Text prompt | 模糊 | ✓ |
| ControlNet | 精确 | 部分损失 |
| **PRoPE** | 精确（几何） | ✓ |

## 关联
- 来源论文：[[20-Papers/WM/2026-minWM-Real-Time-Video-World-Models]]
- 引用 [26]：原始 PRoPE 论文（NeurIPS 2024，待读）
- 相关：[[30-Notes/concepts/World-Model]]
