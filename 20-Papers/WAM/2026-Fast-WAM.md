---
title: "Fast-WAM: Do World Action Models Need Test-time Future Imagination?"
authors: [Haoyuan Yuan, Zhilin Dong, Yucheng Lin, Hang Zhao]
year: 2026
venue: arXiv
tags: [paper, WAM, world-action-model, embodied-AI, video-prediction, policy-learning]
status: reading
rating: 
date-read: 
url: https://arxiv.org/abs/2603.16662
pdf: "C:\\Users\\Administrator\\Zotero\\storage\\QB328I9K\\Yuan 等 - 2026 - Fast-WAM Do World Action Models Need Test-time Future Imagination.pdf"
priority: high
---

# Fast-WAM: Do World Action Models Need Test-time Future Imagination?

## TL;DR
> WAM（World Action Models）的性能增益主要来自训练时的视频建模（video co-training），而非推理时的未来帧生成（test-time imagination）。Fast-WAM 保留训练时的视频目标，但在推理时完全跳过未来视频解码，直接从潜空间产生动作，大幅降低推理开销同时保持甚至超越完整 WAM 的性能。

## 核心问题
- 现有 WAM 遵循 imagine-then-execute 范式：先生成未来视频，再基于视频做决策
- 这带来巨大推理开销（视频解码 + 多步生成）
- **本文追问**：WAM 的收益到底来自哪里？是训练时的视频建模？还是推理时的未来想象？

## 相关工作 
VLA 在语义理解很强，但是 action-conditioned dynamics (给定当前状态和动作，预测下一个状态) 不一定好。Fast-WAM 在测试时也像 VLA 一样是直接输出动作的 policy interface，不需要先生成未来视频；但它在训练阶段额外加入了基于未来视觉预测的 world-modeling objective。

## 方法：Fast-WAM

### 架构设计
- 基于 Video DiT（Diffusion Transformer），构建在预训练 DiT（Wan2.5-1B）之上
- 输入分三组 token：
  1. **Clean first-frame tokens**：当前观察（清晰帧）
  2. **Noisy future video tokens**：用于视频重建训练
  3. **Action tokens**：动作预测（learnable tokens）
- **关键设计**：结构化注意力掩码（structured attention mask）
  - 训练时：video tokens 和 action tokens 共享视觉上下文，但 action tokens 不能看到 future video tokens
  - 推理时：直接丢弃 future video tokens，仅用 clean first-frame + action tokens

### 训练目标
三个损失的加权组合：
- $\mathcal{L}_{act}$：动作预测（flow matching）
- $\mathcal{L}_{vid}$：视频重建（flow matching on video）
- $\mathcal{L} = \mathcal{L}_{act} + \mathcal{L}_{vid}$

### 推理流程
- 不生成未来视频帧
- 仅输入当前 clean first-frame tokens → 直接输出动作
- 推理开销远低于标准 WAM

## 可控实验（Controlled Variants）

为解耦各因素贡献，设计了一组变体：
| 变体 | 视频 co-training | 推理时视频生成 | 说明 |
|------|:---:|:---:|------|
| Fast-WAM | ✓ | ✗ | 完整方案 |
| Fast-WAM-joint | ✓(joint attn) | ✗ | action 可看 video tokens |
| Fast-WAM-SIM | ✗ | ✗ | 无视频建模（纯 action） |
| Fast-WAM w/ video co-run | ✓ | ✓ | 完整生成+推理 |

## 实验结果

### RoboTwin（模拟）
- Fast-WAM 达到 **91.8%** 成功率（无 embedded pretraining）
- 超越多数使用 embedded pretraining 的 WAM 基线
- 关键发现：去掉 video co-training（SIM 变体）性能大幅下降 → 证明训练时视频建模是关键

### LIBERO（模拟）
- Fast-WAM 达到 **97.6%** 平均成功率
- 在 LIBERO-Spatial/Object/Goal/Long 四个子任务均表现竞争力强

### 真实机器人（Galaxea R1 Lite）
- 毛巾折叠任务，106 段演示，10K 步训练
- Fast-WAM 策略版本（$r_5$）成功率最高，同时完成时间最短
- 去掉 video co-running 同时降低成功率和增加完成时间

### 关键发现
1. **视频 co-training 是性能增益的主要来源**，而非推理时的未来生成
2. **去掉推理时视频生成对性能影响很小**，但推理效率大幅提升
3. Fast-WAM 推理延迟约 ~10ms，而完整 WAM 需要数百毫秒
4. 适合实时闭环控制场景

## 为什么读
- 直接挑战 WAM 的核心设计假设（imagine-then-execute）
- 提供了一个简洁高效的 WAM 设计范式
- 实验设计清晰，可控变体有效解耦各因素贡献
- 对实时机器人部署有直接参考价值

## 关联
- 同主题：[[20-Papers/WM/2026-minWM-Real-Time-Video-World-Models]]
- 概念：[[30-Notes/concepts/World-Model]]
- 相关方法：[[20-Papers/WM/2024-Self-Forcing]]
