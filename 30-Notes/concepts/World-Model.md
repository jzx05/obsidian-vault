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

## 两大分类（综述视角）

WM 综述将所有方法分为两条根本不同的路线：

| | [[30-Notes/concepts/MBRL-World-Model\|Model-Based RL WM]] | [[30-Notes/concepts/Generative-World-Model\|大规模生成式 WM]] |
|---|---|---|
| **核心问题** | 下一步状态是什么？（为 RL 服务） | 下一帧画面是什么？（为感知/仿真服务） |
| **表示空间** | 压缩 latent vector | 像素 / 高维视频帧 |
| **训练数据** | 小量任务轨迹 | 互联网级视频 |
| **代表方法** | DreamerV1/V2/V3, TD-MPC2 | Genie 3, GameNGen, minWM, DIAMOND |
| **计算规模** | 单 GPU 可训 | 百~千 GPU 预训练 |

> 两者并非对立——生成式 WM 正在吸收 MBRL 的动作条件化思路，形成融合趋势。

## 主要技术路线
1. **Latent Dynamics Model**（Dreamer 系列）— 学潜空间状态转移，为 RL 服务 → [[30-Notes/concepts/MBRL-World-Model]]
2. **Video Diffusion-based**（Sora、Genie、minWM）— 像素级生成，高保真 → [[30-Notes/concepts/Generative-World-Model]]
3. **Autoregressive Token**（Genie、DIAMOND）— 帧离散化为 token，可与 LLM 统一

## 例子
- **DreamerV3**（Google DeepMind 2023）— MBRL 路线标杆
- **Genie 3**（DeepMind 2025）— 生成式路线，3D 感知
- **Hunyuan-GameCraft**（腾讯）— 游戏视频 WM
- **minWM**（清华生数 2026）— 实时生成式 WM，见 [[20-Papers/WM/2026-minWM-Real-Time-Video-World-Models]]
- **Sora**（OpenAI 2024）— 早期"模拟器"叙事

## 关联
- 路线详解：[[30-Notes/concepts/MBRL-World-Model]]、[[30-Notes/concepts/Generative-World-Model]]
- 上游能力：[[30-Notes/concepts/Causal-Forcing]]
- 控制方式：[[30-Notes/concepts/PRoPE]]（相机控制）
- 加速技术：[[30-Notes/concepts/Asymmetric-DMD]]
