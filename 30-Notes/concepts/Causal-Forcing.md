---
title: "Causal Forcing"
tags: [note, concept, autoregressive, video-diffusion]
created: 2026-06-05
related:
  - "[[20-Papers/WM/2026-minWM-Real-Time-Video-World-Models]]"
---

# Causal Forcing

## 定义
> 一种把**双向视频扩散模型**改造成**因果（自回归）模型**的训练策略 — 训练时强制每帧只能"看到"之前的帧（mask 掉未来），同时保留扩散去噪能力。

## 为什么重要
- 双向模型生成必须等所有帧一起完成 → **延迟高、不可流式**
- AR 模型可以**逐帧生成、可流式输出** → 实时交互的前提
- 直接 fine-tune 用 next-token loss 会破坏预训练扩散能力 → Causal Forcing 是更温和的过渡方案

## 核心要点
- **训练时的 mask**：让 self-attention 只看历史帧
- **保留扩散过程**：每帧仍走完整去噪步骤（多步），不退化为 single-step 预测
- **滑动窗口推理**：生成第 t 帧时，把 t-W..t-1 帧作为 condition
- 与 **Teacher Forcing** 类似的思路，但用在视频扩散场景

## 变体
- **Causal Forcing**（Song et al., 2024）— 原始版本
- **Causal Forcing++** — 支持 few-step 推理的扩展版（minWM Stage 2b 用到）
- **Self-Forcing** — 缓解 train-test gap

## 例子
- minWM Stage 2a：把 Wan2.1-T2V / HY1.5-T2V 双向模型转为多步 AR 模型
- 后续可叠加 [[30-Notes/concepts/Asymmetric-DMD]] 进一步压成 few-step

## 关联
- 来源论文：[[20-Papers/WM/2026-minWM-Real-Time-Video-World-Models]]
- 相关概念：[[30-Notes/concepts/World-Model]]、[[30-Notes/concepts/Asymmetric-DMD]]
- 待读引用：[23] Causal Forcing 原文、[24] Causal Forcing++、[22] Self Forcing
