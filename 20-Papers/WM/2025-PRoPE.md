---
title: "Cameras as Relative Positional Encoding"
authors: []
year: 2025
venue: NeurIPS 2025
tags:
  - paper
  - attention
  - camera-control
  - position-embedding
  - prope
status: reading
rating: 
date-read: 2026-06-05
url: 
pdf: 
priority: high
---

# PRoPE — Cameras as Relative Positional Encoding

> minWM Section 2.1 的核心依赖，Stage 1 的根基。
> 相关图：![[Excalidraw/minWM-PRoPE-explanation.excalidraw]] · ![[Excalidraw/PRoPE-SE3-equivariance.excalidraw]]

## TL;DR

> 把相机内外参 (K, T) 注入 Self-Attention 的位置编码方法。
> 核心洞察：**两 token 的 attention 只依赖相对位姿，绝对位置/朝向无意义。**
> 把 RoPE 的"序列位置旋转"推广到"相机位姿旋转"，对 SE(3) 变换不变。

---

## 核心符号拆解

### 输入：每帧的相机参数

每帧 i 有两个参数矩阵：

| 符号 | 维度 | 含义 | 包含什么 |
|------|------|------|---------|
| `K_i` | 3×3 | **内参**（Intrinsic）| 焦距 fx,fy；主点 cx,cy |
| `T_i` | SE(3) | **外参**（Extrinsic）| 旋转 R_i + 平移 t_i |

```
        ┌  fx   0   cx ┐                  ┌ R_i | t_i ┐
K_i  =  │   0  fy   cy │       T_i  =    │─────┼──────│
        └   0   0    1 ┘                  └  0  |  1  ┘
```

内参 = 相机镜头特性（和场景无关）
外参 = 相机在世界里的位姿（和场景有关）

---

### P_i — 投影位置编码矩阵

PRoPE 把 (K_i, T_i) 合成一个**块对角矩阵** P_i：

```
P_i = block-diag( K_i⁻ᵀ R_i ,  K_i⁻ᵀ )

     = ┌ K_i⁻ᵀ R_i |    0    ┐
       │────────────┼─────────│
       └     0      |  K_i⁻ᵀ ┘
```

逐项含义：

| 符号 | 含义 | 为什么这样用 |
|------|------|------------|
| `R_i` | 相机旋转矩阵 | 编码相机朝向 |
| `K_i⁻ᵀ` | 内参逆转置 | 把像素坐标"解投影"回相机空间光线方向 |
| `K_i⁻ᵀ R_i` | 旋转后解投影 | 综合内外参，把 token 投影到世界方向 |
| `t_i` | 平移 | **不进入 P_i** → 平移不变性的来源 |
| `block-diag` | 块对角结构 | Q 和 K 各自独立旋转，对应 RoPE 分块设计 |

---

## 注入过程：只改 Q 和 K

### 标准 Self-Attention

```
Q_i = W_q · x_i
K_i = W_k · x_i
V_i = W_v · x_i

A_ij = Q_iᵀ K_j          ← 点积，无几何信息
O_i  = Σ softmax(A_ij) · V_j
```

### PRoPE 注入（仅在这一处加两行）

```
Q_i = W_q · x_i
K_i = W_k · x_i
V_i = W_v · x_i           ← V 完全不变

Q_i' = P_i · Q_i          ← ★ 唯一改动
K_i' = P_i · K_i          ← ★ 唯一改动

A_ij = (P_i Q_i)ᵀ (P_j K_j)
     = Q_iᵀ  P_iᵀ P_j  K_j
              ↑
         只含相对位姿，不含绝对位姿
```

**改动量极小**：仅在 Q/K 线性层后插两个矩阵乘法，其余结构（FFN、LayerNorm、残差）完全保留。

---

## SE(3) 等变性（不变性）

参见图：![[Excalidraw/PRoPE-SE3-equivariance.excalidraw]]

### 什么是 SE(3)

SE(3) = 三维空间的所有刚体运动 = 旋转 + 平移（不含缩放/变形）

**等变/不变的区别：**
- 不变（Invariant）：输入变了，输出不变
- PRoPE 的 A_ij 对全局 SE(3) 变换**不变**

### ① 平移不变性

平移整个场景 d：t_i → t_i + d，但 **t_i 根本不在 P_i 里**

```
P_i 不变  →  P_iᵀ P_j 不变  ✓
```

类比：把摄影棚搬到地球另一端，两台相机的相对夹角完全不变。

### ② 旋转不变性（关键）

旋转整个场景 G：R_i → G R_i，则 P_i → G P_i

```
(G P_i)ᵀ (G P_j)
= P_iᵀ  Gᵀ G  P_j
= P_iᵀ   I   P_j    ← G 是正交矩阵，Gᵀ G = I
= P_iᵀ P_j  ✓
```

**G 和 Gᵀ 互相抵消**（旋转出去再转回来 = 没变）。

具体到相对旋转：

```
全局旋转前：P_iᵀ P_j  ∝  R_iᵀ R_j
全局旋转后：(G R_i)ᵀ (G R_j) = R_iᵀ Gᵀ G R_j = R_iᵀ R_j  ✓
```

### 为什么这对神经网络重要

```
情景 A：相机 i 在 (0,0,0)，相机 j 在 (1,0,0)
情景 B：相机 i 在 (1000,0,0)，相机 j 在 (1001,0,0)
```

两个情景几何关系完全相同，attention 分数应该相同。
PRoPE 从**结构上保证**这一点，模型连学都不需要学。

---

## 和 1D RoPE 的对应关系

| 维度 | 1D RoPE | PRoPE |
|------|---------|-------|
| 位置来源 | 序列下标 i | 相机位姿 (K_i, R_i) |
| 注入方式 | R_i · Q（2D 旋转）| P_i · Q（投影旋转）|
| 相对性 | A_ij 只依赖 (i−j) | A_ij 只依赖 P_iᵀ P_j（相对位姿）|
| V 改动 | 否 | 否 |
| 平移不变 | 绝对位置 i 不重要 | 绝对平移 t_i 不进 P_i |

**PRoPE = 把 RoPE 里的"序列位置旋转"换成"相机位姿旋转"**，机制完全相同。

---

## 相机相同 vs 不同时的行为

### 相机完全相同（K_i = K_j, R_i = R_j）

```
P_i = P_j
P_iᵀ P_j = P_iᵀ P_i = 常数矩阵 C（对所有 token 都一样）

A_ij = Q_iᵀ · C · K_j  ← 等价于原始 attention + 一个缩放
```

**退化成普通 attention，几何信息不提供区分度。预训练知识完全保留。**

### 相机不同（R_i ≠ R_j）

```
P_iᵀ P_j  ∝  R_iᵀ R_j（含具体的相对旋转角度）

A_ij ≠ A_ik  （j 和 k 与 i 的视角差不同）
```

模型"感知"到不同帧之间的视角几何关系，自动学习视差、遮挡、多视角一致性。

**结论：PRoPE 是"按需激活"的** — 单目视频不干扰，多视角/相机运动时自动编码几何。

---

## 训练数据需求（来自 minWM 消融）

PRoPE 训练需要 **ground-truth 相机轨迹**（估计 pose 不够）：

| 数据来源 | 效果 | 原因 |
|---------|------|------|
| SpatialVid（估计 pose）| 失败 | 估计误差太大，信号被噪声淹没 |
| DL3DV（COLMAP 重建）| 成功 | Ground-truth 轨迹，信号准确 |
| WorldPlay（合成生成）| 成功 | 精确控制的合成轨迹 |

其他关键超参：

| 参数 | 值 | 说明 |
|------|---|------|
| 训练步数 | 8k+ steps | <5k 步相机控制无响应 |
| 批量大小 | bs ≥ 16 | bs<8 无法稳定训练 |

---

## 我的思考

- PRoPE 的精妙：**改动最小化**，只动 Q/K，保留 V 和所有其他结构，几乎不破坏预训练
- SE(3) 不变性是"免费午餐"，不需要数据增强，结构上直接保证
- 局限：只编码相机旋转，不编码平移（t_i 不在 P_i 里）→ 纯平移场景的几何关系无法编码
- 扩展可能：能否把 t_i 也编码进去？或许需要处理尺度歧义问题

---

## 关联

- 概念笔记：[[30-Notes/concepts/PRoPE]]
- 核心应用：[[20-Papers/WM/2026-minWM-Real-Time-Video-World-Models]]
- SE(3) 图解：[[Excalidraw/PRoPE-SE3-equivariance.excalidraw]]
- 数据流图：[[Excalidraw/minWM-PRoPE-explanation.excalidraw]]
- 相关概念：[[30-Notes/concepts/Asymmetric-DMD]] · [[30-Notes/concepts/Causal-Forcing]]
