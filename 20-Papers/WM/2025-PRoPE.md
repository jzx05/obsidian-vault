---
title: Cameras as Relative Positional Encoding
authors: []
year: 2025
venue: NeurIPS 2025
tags:
  - paper
  - attention
  - camera-control
  - position-embedding
  - to-read
status: to-read
rating:
date-read:
url:
pdf:
priority: medium
---

# PRoPE（minWM 引用 [26]）

## TL;DR
> 把摄像机内外参注入 self-attention 的位置编码方法 — minWM Stage 1 的根基
**PRoPE 的 attention 只看两台相机的"相对关系"，绝对位置、绝对朝向对它毫无影响。**


## 为什么读
- minWM 摄像机控制的底层机制
- 理解后可推广到其他几何控制场景（多视角、3D 重建）

## 核心方法
SE(3) 等变性
[[PRoPE-SE3-equivariance.excalidraw]]

情景 A：相机 i 在 (0,0,0)，相机 j 在 (1,0,0)
情景 B：相机 i 在 (1000,0,0)，相机 j 在 (1001,0,0)

## 关联
- 概念笔记：[[30-Notes/concepts/PRoPE]]
- 应用：[[20-Papers/WM/2026-minWM-Real-Time-Video-World-Models]]
