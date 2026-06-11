---
title: "Diffusion Transformer (DiT)"
tags: [concept, diffusion, transformer, architecture, video-generation]
created: 2026-06-11
related:
  - "[[30-Notes/concepts/Diffusion-Policy]]"
  - "[[30-Notes/concepts/World-Model]]"
  - "[[30-Notes/concepts/Generative-World-Model]]"
---

# Diffusion Transformer (DiT)

## 核心思想

用 **Transformer** 替代传统扩散模型中的 U-Net 作为去噪骨干网络。

传统扩散模型（如 DDPM、Stable Diffusion）用 U-Net 做 denoiser，但 U-Net 的归纳偏置（局部感受野、固定分辨率）限制了 scaling 能力。DiT 证明：把 denoiser 换成 Transformer，配合 patch 化的输入，可以获得更好的 scaling law 和生成质量。

---

## 架构

```
输入（图像/视频 latent）
     │
     ▼
[Patchify] → 将 latent 切成 patch → 每个 patch 变成一个 token
     │
     ▼
[Position Embedding] → 加入位置信息
     │
     ▼
[DiT Blocks × N]
  ├── Self-Attention（patch 间交互）
  ├── Cross-Attention（条件注入：文本 / 时间步 / 其他）
  └── FFN
     │
     ▼
[Unpatchify] → 还原为 latent 空间的噪声预测
```

### 关键设计

| 组件 | 作用 |
|------|------|
| **Patchify** | 将连续 latent map 切成离散 token 序列，类似 ViT |
| **Adaptive LayerNorm (adaLN-Zero)** | 条件注入方式：时间步 t 和类别标签通过 adaLN 调制每层的 scale/shift |
| **位置编码** | 2D/3D sinusoidal 或 RoPE（视频场景用 3D：空间+时间） |

---

## 条件注入方式

DiT 论文对比了多种条件注入机制：

| 方式 | 实现 | 效果 |
|------|------|------|
| In-context | 条件 token 拼接到输入序列 | 简单但占序列长度 |
| Cross-attention | 类似 Stable Diffusion 的交叉注意力 | 灵活，适合文本条件 |
| **adaLN-Zero** | 条件 → MLP → (γ, β, α) 调制 LN | 最优性能，DiT 推荐 |

---

## 为什么比 U-Net 好

1. **更好的 scaling**：参数量翻倍 → FID 持续下降（U-Net 有天花板）
2. **统一架构**：图像、视频、3D 都可以用同一套 Transformer
3. **灵活的序列建模**：支持变长输入、多模态 token 混合
4. **预训练迁移**：可复用大规模语言/视觉 Transformer 的权重

---

## 在视频生成中的应用

视频 DiT 将 3D latent（时间 × 高 × 宽）patch 化为 token 序列：

```
视频 latent: (T, H, W, C)
    │
    ▼
Patchify: patch_size = (t, h, w)
    │
    ▼
Token 序列: N = (T/t) × (H/h) × (W/w) 个 tokens
    │
    ▼
DiT Blocks (全局时空注意力)
```

代表模型：
- **Sora** (OpenAI) — 视频 DiT，支持变分辨率/变时长
- **Wan2.1/2.5** (阿里) — 开源视频 DiT，Fast-WAM 的 backbone
- **MovieGen** (Meta) — 长视频生成

---

## 在 Fast-WAM 中的角色

Fast-WAM 直接复用预训练的 **Wan2.5-1B** 视频 DiT 作为共享 backbone：

```
              ┌─── Video DiT (Wan2.5-1B) ───┐
              │                              │
              │  video tokens ← 视频去噪任务  │
              │  action tokens ← 动作预测任务 │
              │                              │
              │  + Action DiT (专家头)         │
              └──────────────────────────────┘
```

关键 insight：
- DiT 的预训练已经编码了丰富的时空动态知识
- 通过 video co-training，进一步强化 backbone 对物理世界的理解
- 推理时不需要跑完整的视频去噪 loop，只做一次前向传播出动作

---

## DiT vs U-Net 速查表

| | U-Net | DiT |
|---|---|---|
| 归纳偏置 | 局部（卷积） | 全局（注意力） |
| Scaling | 有限 | 持续提升 |
| 分辨率灵活性 | 需要 redesign | 天然支持 |
| 条件注入 | cross-attn + concat | adaLN-Zero（更优） |
| 计算模式 | 多尺度特征图 | 统一序列 |
| 代表作 | SD 1.5, SD 2.1 | SD3, Sora, Wan2.5 |

---

## 代表论文

| 论文 | 年份 | 贡献 |
|------|------|------|
| **DiT** (Peebles & Xie) | 2023 | 提出 DiT 架构，证明 Transformer 做 diffusion backbone 的可行性 |
| **U-ViT** (Bao et al.) | 2023 | 类似思路，加了 skip connections |
| **SD3** (Esser et al.) | 2024 | MMDiT，多模态双流 DiT |
| **Sora** (OpenAI) | 2024 | 视频 DiT 的大规模实现 |
| **Wan2.5** (阿里) | 2025 | 开源视频 DiT，Fast-WAM 的 base model |

---

## 相关概念
- [[30-Notes/concepts/Diffusion-Policy]] — 扩散模型做策略（action denoising）
- [[30-Notes/concepts/World-Model]] — DiT 作为 WM backbone
- [[30-Notes/concepts/Generative-World-Model]] — 生成式世界模型通常用 DiT
- [[20-Papers/WAM/2026-Fast-WAM]] — DiT 在 WAM 中的具体应用
