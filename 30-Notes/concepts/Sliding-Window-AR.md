---
title: "Sliding Window AR"
tags: [note, concept, autoregressive, inference, video-diffusion]
created: 2026-06-05
related:
  - "[[30-Notes/concepts/Causal-Forcing]]"
  - "[[30-Notes/concepts/KV-Cache]]"
---

# Sliding Window AR（滑动窗口自回归）

## 定义
> AR 视频生成的推理策略：把视频分成固定大小的窗口（W 帧），每次生成一个窗口，用上一个窗口的输出作为当前窗口的条件输入，逐窗口滚动生成任意长度视频。

---

## 和"帧级 AR"的区别

```
纯帧级 AR（LLM 风格）：        滑动窗口 AR（Causal Forcing）：
  f1 完全生成                   Window 1: [f1 f2 f3 f4] 同时去噪 50步
  f2 | 条件=[f1]               Window 2: [f5 f6 f7 f8] 同时去噪 50步
  f3 | 条件=[f1,f2]            Window 3: [f9 f10 f11 f12] ...
  ...                          条件 = 上一个窗口的输出（固定长度 W）
  fn | 条件=[f1...fn-1]
        ↑ 上下文无限增长              ↑ 上下文固定为 W 帧
```

---

## 窗口内部的去噪：不是顺序生成，是同时去噪

**常见误解**：
```
❌ f1 先生成完 → f2 用干净 f1 → f3 用干净 f1,f2 → f4 用干净 f1,f2,f3
```

**实际**：
```
t=T:  [噪声f1][噪声f2][噪声f3][噪声f4]  ← 4帧同时从纯噪声开始
  ↓  一步去噪（causal mask）
t=T-1:[半清f1][半清f2][半清f3][半清f4]  ← 4帧同时更新
  ↓  ... 重复 50 步 ...
t=0:  [干净f1][干净f2][干净f3][干净f4]  ← 整体输出
```

每步去噪里，**f4 看到的是当前噪声水平下的 f1,f2,f3**（不是干净帧）：

```
Causal mask（每步去噪内）：
       f1   f2   f3   f4
  f1 [  ✓    ✗    ✗    ✗ ]
  f2 [  ✓    ✓    ✗    ✗ ]
  f3 [  ✓    ✓    ✓    ✗ ]
  f4 [  ✓    ✓    ✓    ✓ ]
  ↑ 这里的 f1,f2,f3 是当前 t 时刻的噪声状态，不是干净帧！
```

f1~f4 共同演化、互相参考，所以窗口内质量比纯帧级 AR 好。

---

## 条件帧如何进入模型

上一个窗口的输出（干净帧）以**噪声水平=0 拼接进当前序列**：

```
Window 2 完整输入序列：

[干净f1][干净f2][干净f3][干净f4] | [噪声f5][噪声f6][噪声f7][噪声f8]
←────── 条件区，frozen ─────────→  ←────── 当前去噪区 ─────────────→
  不参与去噪更新，只提供 attention 上下文   每步去噪更新这 4 帧
```

完整 Causal mask（包含条件区）：

```
         条件区           当前去噪区
         f1  f2  f3  f4 | f5  f6  f7  f8
    f5 [  ✓   ✓   ✓   ✓ |  ✓   ✗   ✗   ✗ ]
    f6 [  ✓   ✓   ✓   ✓ |  ✓   ✓   ✗   ✗ ]
    f7 [  ✓   ✓   ✓   ✓ |  ✓   ✓   ✓   ✗ ]
    f8 [  ✓   ✓   ✓   ✓ |  ✓   ✓   ✓   ✓ ]
         ↑ 当前去噪帧可以 attend 到所有条件帧
```

f8 在每步去噪时能看到：干净 f1~f4（条件区，完美的） + 当前噪声水平的 f5,f6,f7（正在演化）。

---

## 为什么不用"每帧依赖所有历史"？

| 问题 | 原因 |
|------|------|
| Attention 是 O(N²)，生成越来越慢 | 第 N 帧要 attend 前 N-1 帧，无上界 |
| KV Cache 显存无限增长 | 每帧新增 K/V，长视频 OOM |
| 超出训练分布 | 基础模型在 ~16 帧片段上训练，1000 帧历史是 OOD |
| 时序依赖本来就是局部的 | 第 200 帧主要依赖第 196~199 帧，不需要第 1 帧 |

---

## 窗口大小 W 的取舍

```
W 小（如 2 帧）：              W 大（如 16 帧）：
  延迟低 ✓                      延迟高 ✗
  首帧快 ✓                      首帧慢 ✗
  窗口内质量差 ✗                窗口内质量好 ✓
  Exposure Bias 积累慢 ✓        更接近双向模型质量 ✓
  GPU 利用率低 ✗                GPU 利用率高 ✓
```

实践中 W 是根据延迟目标和显存限制调整的超参数，minWM 论文未明确给出。

---

## 与 LLM context window 的类比

```
GPT context window（128k tokens）：
  只记住最近 128k token，更早的被截断

Causal Forcing sliding window（W 帧）：
  只记住最近 W 帧，更早的被"遗忘"

本质相同：用有界上下文换取可控的计算开销
```

"长时一致性"（如何在不增加计算量的前提下记住更早的关键帧）是世界模型领域的开放问题。

---

## KV Cache 与滑动窗口

滑动窗口把 seq_len 固定为 W，配合 KV Cache：

```
显存占用 = num_layers × W × d_model × 2（K+V）× dtype_bytes
         = 固定值，不随视频长度增长
```

→ 详见 [[30-Notes/concepts/KV-Cache]]

---

## 关联
- 实现基础：[[30-Notes/concepts/Causal-Forcing]]
- 显存优化：[[30-Notes/concepts/KV-Cache]]
- 应用：[[20-Papers/WM/2024-Causal-Forcing]]
- 长时一致性问题：[[20-Papers/WM/2026-minWM-Real-Time-Video-World-Models]]（Future Work 部分）
