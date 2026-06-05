---
title: "World Model"
tags: [note, concept, world-model]
created: 2026-06-05
related:
  - "[[20-Papers/WM/2026-minWM-Real-Time-Video-World-Models]]"
---

# World Model（世界模型）

## 定义
> 一种能**预测环境未来状态**、并响应外部动作（agent action / camera / user input）的生成模型。本质是"可交互、可模拟"的环境。

## 为什么重要
- 是**具身智能 / 自主体（embodied AI）**的核心基础设施 — 在虚拟环境中训练智能体远比真实世界便宜
- 是**生成式游戏 / 神经渲染引擎**的底层 — Genie、GameNGen、minWM 都属此类
- 让 AI 从"理解世界"走向"模拟世界"

## 核心要点
- **可控性（Controllability）**：能响应输入信号（动作、相机、文本）
- **一致性（Consistency）**：长时间生成保持物理 / 视觉一致
- **实时性（Real-time）**：交互应用要求低延迟（< 5s 首帧）
- **预测性（Predictive）**：给定历史 → 预测下一帧/下一段

## 主要技术路线
1. **Latent Dynamics Model**（Dreamer 系列）— 学潜空间的状态转移
2. **Video Diffusion-based**（Sora、Genie、minWM）— 直接在像素 / latent video 上生成
3. **Autoregressive Token**（Genie、Game Engine LLM）— 把帧离散化为 token

## 例子
- **Genie 3**（DeepMind 2025）
- **Hunyuan-GameCraft**（腾讯）
- **minWM**（清华生数 2026）— 见 [[20-Papers/WM/2026-minWM-Real-Time-Video-World-Models]]
- **Sora**（OpenAI 2024）— 早期"模拟器"叙事

## 关联
- 上游能力：[[30-Notes/concepts/Video-Diffusion]]、[[30-Notes/concepts/Causal-Forcing]]
- 控制方式：[[30-Notes/concepts/PRoPE]]（相机控制）
- 加速技术：[[30-Notes/concepts/Asymmetric-DMD]]
