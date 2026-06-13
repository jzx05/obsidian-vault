---
title: "Mixture-of-Transformers (MoT)"
tags: [concept, architecture, transformer, multi-modal, multi-expert]
created: 2026-06-13
related:
  - "[[30-Notes/concepts/Diffusion-Transformer]]"
  - "[[20-Papers/WAM/2026-Fast-WAM]]"
---

# Mixture-of-Transformers (MoT)

## 核心思想

多模态数据共享 **self-attention**（允许跨模态对话），但各自保留独立的 **FFN 和 cross-attention**（各自独立思考）。

这是一种介于"完全独立模型"和"完全共享 Transformer"之间的设计。

---

## 为什么需要 MoT

| 方案 | 跨模态交互 | 模态特化 | 问题 |
|------|:---:|:---:|------|
| 完全独立 | ✗ | ✓ | 无法互相利用信息 |
| 完全共享 | ✓ | ✗ | 强制统一处理差异极大的模态 |
| **MoT** | ✓ | ✓ | 平衡点：对话但不干扰 |

---

## 架构结构

```
Input: [video_tokens, action_tokens]

For each DiT Block:
  ┌──────────────────────────────────────────┐
  │            Shared Self-Attention          │
  │                                          │
  │  Q = cat([Q_video, Q_action])            │
  │  K = cat([K_video, K_action])            │
  │  V = cat([V_video, V_action])            │
  │  → attention → split back                │
  │  + attention mask 控制谁能看谁            │
  └──────────────────────────────────────────┘
           │                    │
           ▼                    ▼
  ┌─────────────────┐  ┌─────────────────┐
  │ Video Expert    │  │ Action Expert   │
  │ - Cross-Attn    │  │ - Cross-Attn    │
  │   (text+action) │  │   (text only)   │
  │ - FFN           │  │ - FFN           │
  │ - AdaLN (τ_v)   │  │ - AdaLN (τ_a)   │
  └─────────────────┘  └─────────────────┘
```

---

## 约束条件

| 参数 | 是否必须相同 | 原因 |
|------|:---:|------|
| `num_heads` | ✓ | Q/K/V 物理拼接后 reshape 需要一致 |
| `attn_head_dim` | ✓ | `Q @ K^T` 要求维度匹配 |
| `hidden_dim` | ✗ | Wq 输入维度可以不同，只要输出维度（H×D）一致 |
| `num_layers` | ✓ | 逐层共享 self-attention |

---

## Attention Mask 设计

MoT 的精髓在于**通过 mask 控制信息流方向**：

```
           Video KV    Action KV
          ┌──────────┬──────────┐
Video Q   │    ✓     │    ✗     │  video 看不到 action
          ├──────────┼──────────┤
Action Q  │ 首帧 only │    ✓     │  action 只看 video 第一帧
          └──────────┴──────────┘
```

这个不对称设计实现了：
- Video KV 不依赖 action → 可以 cache
- Action 从视觉获取信息 → grounded policy
- 因果方向正确：观察 → 决策

---

## 与其他多模态架构的对比

| 架构 | 跨模态方式 | 代表 |
|------|-----------|------|
| Concat + shared Transformer | 全部 token 拼接，共享所有层 | Flamingo, PaLM-E |
| Cross-attention only | 一个模态作为 context，另一个做 query | Stable Diffusion |
| **MoT** | 共享 self-attn，独立 FFN/cross-attn | **Fast-WAM** |
| Dual model + feature extraction | 独立模型，一个向另一个传 hidden states | DiT4DiT |

---

## MoT 的推理效率优势

由于 video 的 self-attention 不看 action：

```
Prefill 阶段（一次性）：
  video first frame → Video Expert forward → 产出 KV cache
  （不依赖 action，所以 KV 是固定的）

Denoising 阶段（多次）：
  action tokens + cached video KV → Action Expert forward × N steps
  （每步只跑 action 部分，video KV 直接复用）
```

这比"每步都跑完整模型"快很多倍。

---

## 在 Fast-WAM 中的具体实现

- Video Expert: 来自 Wan2.5-1B 预训练（1B params）
- Action Expert: 较小的 DiT（~100M params），从 Video Expert 初始化
- 共享: self-attention 的 Q/K/V projection + attention computation
- 独立: FFN, cross-attention, AdaLN modulation, encoder/head

---

## 相关概念
- [[30-Notes/concepts/Diffusion-Transformer]] — MoT 中的单个 Expert 就是一个 DiT
- [[30-Notes/concepts/Flow-Matching]] — MoT 中两个 Expert 各自用 flow matching 训练
- [[20-Papers/WAM/2026-Fast-WAM]] — MoT 的具体应用（Q1-Q4 详细解析）
- [[20-Papers/WAM/2026-DiT4DiT]] — 另一种多模态 DiT 联合方案（非 MoT）
