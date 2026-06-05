---
title: "Causal Forcing"
tags: [note, concept, autoregressive, video-diffusion]
created: 2026-06-05
related:
  - "[[20-Papers/WM/2026-minWM-Real-Time-Video-World-Models]]"
  - "[[20-Papers/WM/2024-Causal-Forcing]]"
---

# Causal Forcing

## 定义
> 把**双向视频扩散模型**改造成**因果（自回归）模型**的训练策略：
> 训练时对 self-attention 加下三角 mask，同时把历史干净帧拼接为条件，保留完整多步去噪能力。

## 为什么重要
- 双向模型推理时所有帧必须**同时完成**才能输出 → 延迟高，不可流式
- AR 模型窗口完成后立即输出 → 实时交互的前提
- 直接用 next-frame prediction loss 会破坏预训练扩散能力 → Causal Forcing 是最小改动的过渡方案

## 核心操作（三步）

### 1. 换 attention mask
```
双向（原始）：f1 ↔ f2 ↔ f3 ↔ f4   全对全，无 mask
因果（CF）：  f1 → f2 → f3 → f4   下三角 mask，只看历史
```
代码层面：`attn_mask = None` → 下三角矩阵，**其余结构完全不变**。

### 2. 历史干净帧作为条件（Teacher Forcing）
```
训练时：
  去噪窗口 [f5,f6,f7,f8] 的条件 = GT 历史干净帧 [f1,f2,f3,f4]
  ↑ "喂正确答案"，即 Teacher Forcing

推理时：
  条件 = 自己生成的历史帧（有误差）← Exposure Bias 的来源
```

### 3. 条件帧的拼接方式
历史帧**以噪声水平=0 拼接进序列**，和待去噪帧组成同一个 Transformer 输入：
```
Window 2 输入序列：
[干净f1][干净f2][干净f3][干净f4] | [噪声f5][噪声f6][噪声f7][噪声f8]
←──── 条件区，frozen ────────→   ←──── 当前去噪区 ────────────────→
                                    ↑ 每步去噪只更新这4帧
```
每步去噪中，f5~f8 通过 causal mask 可以 attend 到所有条件帧（f1~f4）。

## 窗口内去噪：f1~f4 是同时去噪的，不是顺序的

**常见误解**：f1 生成完 → f2 用干净 f1 生成 → f3 ...（像 LLM 那样）

**实际**：
```
t=T:   [噪声f1, 噪声f2, 噪声f3, 噪声f4]  ← 4帧同时从纯噪声开始
  ↓  一步去噪（causal mask）
t=T-1: [半清f1, 半清f2, 半清f3, 半清f4]  ← 4帧同时更新
  ↓  ... 重复 50 步 ...
t=0:   [干净f1, 干净f2, 干净f3, 干净f4]  ← 4帧同时完成，整体输出
```

每步去噪里，f4 看到的是**当前噪声水平下**的 f1,f2,f3（不是干净帧），它们共同演化、互相参考。

## 滑动窗口推理

→ 详见 [[30-Notes/concepts/Sliding-Window-AR]]

```
Window 1: 条件=[]              → 生成 [f1 f2 f3 f4]
Window 2: 条件=[f1 f2 f3 f4]  → 生成 [f5 f6 f7 f8]
Window 3: 条件=[f5 f6 f7 f8]  → 生成 [f9 f10 f11 f12]
```

## 两个局限（Stage 2a 之后仍存在）

| 局限 | 原因 | 谁来解决 |
|------|------|---------|
| 仍需多步去噪（~50步） | CF 只改 mask，不压步数 | Stage 2b（ODE/CD 蒸馏）|
| Exposure Bias | 训练用 GT 历史，推理用自己生成的历史 | Stage 2c（Asymmetric DMD）缓解 |

## 变体
- **Causal Forcing**（Song et al., 2024）— 原始版本，minWM [23]
- **Causal Forcing++** — 扩展到 few-step 推理，minWM [24]，Stage 2b 的 ODE 路径基础
- **Self-Forcing** — 用生成的历史帧代替 GT 训练，正面解决 Exposure Bias，minWM [22]

## 关联
- 论文笔记：[[20-Papers/WM/2024-Causal-Forcing]]
- 蒸馏后续：[[30-Notes/concepts/Asymmetric-DMD]]
- 推理机制：[[30-Notes/concepts/Sliding-Window-AR]]
- KV Cache：[[30-Notes/concepts/KV-Cache]]
- 整体框架：[[20-Papers/WM/2026-minWM-Real-Time-Video-World-Models]]
