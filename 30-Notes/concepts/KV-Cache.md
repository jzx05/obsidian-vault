---
title: "KV Cache"
tags: [note, concept, transformer, inference, efficiency]
created: 2026-06-05
related:
  - "[[30-Notes/concepts/Sliding-Window-AR]]"
  - "[[30-Notes/concepts/Causal-Forcing]]"
---

# KV Cache（Key-Value 缓存）

## 定义
> 在 Transformer 的 AR 推理中，把已经计算过的历史 token 的 K（Key）和 V（Value）矩阵缓存到显存里，生成新 token 时直接复用，避免重复计算。

---

## 问题：AR 推理的重复计算

生成第 5 个 token 时：

```
f5 的 Q  ·  f1 的 K  → score   ← f1 的 K 需要计算
f5 的 Q  ·  f2 的 K  → score   ← f2 的 K 需要计算
f5 的 Q  ·  f3 的 K  → score   ← f3 的 K 需要计算
f5 的 Q  ·  f4 的 K  → score   ← f4 的 K 需要计算
```

生成第 6 个 token 时：

```
f6 的 Q  ·  f1 的 K  → score   ← f1 的 K 上一步已经算过！重复！
f6 的 Q  ·  f2 的 K  → score   ← 同上，重复计算
...
```

**每生成一个新 token，所有历史 token 的 K 和 V 都要重新算一遍**，是纯粹的浪费。

---

## 解法：缓存 K 和 V

```
生成 f4：
  计算 f4 的 Q, K, V
  cache 更新：{ f1:(K,V), f2:(K,V), f3:(K,V), f4:(K,V) }

生成 f5：
  只计算 f5 的 Q, K, V    ← 只算新 token
  直接读 cache 里 f1~f4 的 K, V
  不需要重新算历史 token
```

计算量从 O(N²) 降到 O(N)（每步只算一个新 token）。

---

## 代价：显存

每层、每个 token 都需要存 K 和 V：

```
KV cache 大小 = num_layers × seq_len × 2 × d_model × dtype_bytes

以 Wan2.1-1.3B 为例（估算，24层，d=2048，fp16=2bytes）：
  seq_len = 16 帧  → 约 3 MB
  seq_len = 100 帧 → 约 19 MB
  seq_len = 1000 帧→ 约 190 MB
  seq_len = 10000帧→ 约 1.9 GB   ← 开始吃紧

以 HY1.5-8B 为例（估算，32层，d=4096，fp16）：
  seq_len = 100 帧 → 约 50 MB
  seq_len = 1000 帧→ 约 500 MB
  seq_len = 10000帧→ 约 5 GB    ← 很快 OOM
```

**这就是为什么滑动窗口要把 seq_len 固定住** — 显存占用也固定了，不随视频长度无限增长。

---

## 在视频 AR 生成中的作用

```
不用 KV Cache（每窗口重算）：
  Window 2 生成时：重算 f1~f4 的所有 K,V（4 帧 × 50 去噪步）
  Window 100 生成时：重算前 W 帧 × 50 步 = 每次开销相同

用 KV Cache（缓存条件帧）：
  f1~f4 的 K,V 算一次就永久存在 cache 里
  每个去噪步只算当前窗口新 token 的 K,V
```

对视频扩散模型来说，KV cache 的收益和 LLM 略有不同：
- 每步去噪都是完整的 Transformer forward pass
- 条件帧（历史窗口）在整个去噪过程中固定不变 → 特别适合缓存

---

## 滑动窗口 + KV Cache 的显存公式

```
固定显存上界：
  = num_layers × W × 2 × d_model × bytes
  ↑ W 是窗口大小（固定），不随视频长度变化
```

| W（窗口帧数）| HY1.5-8B 估算 KV Cache | 实用性 |
|------------|----------------------|--------|
| 4 帧 | ~6 MB | 极低，但上下文少 |
| 16 帧 | ~25 MB | 常用配置 |
| 64 帧 | ~100 MB | 质量好，显存可接受 |
| 256 帧 | ~400 MB | 接近实用上限 |

---

## 和 LLM KV Cache 的相同与不同

| 维度 | LLM KV Cache | 视频扩散 KV Cache |
|------|-------------|-----------------|
| 缓存对象 | 历史 text token 的 K,V | 历史视频帧 token 的 K,V |
| 何时填充 | 生成每个新 token 时增量填充 | 窗口开始时一次性填充条件帧 |
| 固定方式 | context window 截断 | sliding window 固定长度 |
| 主要收益 | 避免重算 prompt K,V | 避免重算条件帧 K,V |

---

## 关联
- 使用场景：[[30-Notes/concepts/Sliding-Window-AR]]
- 实现基础：[[30-Notes/concepts/Causal-Forcing]]
- 相关模型：[[20-Papers/WM/2026-minWM-Real-Time-Video-World-Models]]
