---
title: "PRoPE (Projective Rotary Position Embedding)"
tags: [note, concept, attention, camera-control]
created: 2026-06-05
related:
  - "[[20-Papers/WM/2025-PRoPE.md]]"
  - "[[20-Papers/WM/2026-minWM-Real-Time-Video-World-Models]]"
---

# PRoPE — Projective Rotary Position Embedding

> 把相机内外参 (K, T) 注入 self-attention 的位置编码，让模型理解视角几何。

---

## 核心思路

每个 token 对应一帧相机，构造矩阵 P_i：

```
P_i = block-diag( K_i⁻ᵀ Rᵢ ,  K_i⁻ᵀ )
```

注入方式：只改 Q 和 K，V 不动：

```
A_ij = (P_i Q_i)ᵀ (P_j K_j) = Q_iᵀ · P_iᵀ P_j · K_j
                                        ↑
                                   只含相对位姿
```

**t_i（平移）不进入 P_i** → 对全局平移天然不变。
**旋转等变**：全局旋转 G 时，Gᵀ G = I 自动消去。

---

## 与 RoPE 的对应

| | 1D RoPE | PRoPE |
|---|---|---|
| 位置来源 | 序列下标 | 相机位姿 (K, R) |
| 相对性 | A_ij 依赖 (i−j) | A_ij 依赖 P_iᵀ P_j |
| V 改动 | 否 | 否 |

---

## AR 生成中的使用方式

视频按 chunk 生成（如每 4 帧一组）：

```
[噪声 fT1..fT4] → 多步去噪 → [干净 fT1..fT4]
                                      ↓ 作为条件
                              [噪声 fT5..fT8] → 去噪 → [干净 fT5..fT8]
                                                               ↓ ...
```

每个 token 分配对应帧的 P_i，去噪时相机位姿已知（来自输入条件或预设轨迹）。
PRoPE 不改 attention 结构，只在 Q/K 前插入矩阵乘，跨 chunk 的几何关系自动编码进注意力分数。

---

## 注意

- 只编码旋转，不编码平移（t_i 不在 P_i 里）→ 纯平移场景无法感知尺度
- 需要 ground-truth 相机轨迹训练（估计 pose 误差太大，信号被噪声淹没）

---

## 关联

- 论文详情：[[20-Papers/WM/2025-PRoPE.md]]
- 图解：[[Excalidraw/PRoPE-SE3-equivariance.excalidraw]] · [[Excalidraw/minWM-PRoPE-explanation.excalidraw]]
- 应用：[[20-Papers/WM/2026-minWM-Real-Time-Video-World-Models]]
