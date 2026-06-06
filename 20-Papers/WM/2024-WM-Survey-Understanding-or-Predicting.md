---
title: "World Models: Understanding World or Predicting Future? A Comprehensive Survey"
authors: []
year: 2024
venue: 
tags:
  - paper
  - survey
  - world-model
status: reading
rating: 
date-read: 
url: 
pdf: "[[papers/world-models/WM_Survey_2024_Understanding_World_or_Predicting_Future.pdf]]"
---

## TL;DR
> 综述围绕一个核心二分法组织 World Model 全景：**Understanding（建模"世界是什么"）** vs **Predicting Future（建模"接下来发生什么"）**。

## 问题与动机
- World Model 概念被滥用，不同社区（RL / 视频生成 / 具身 / LLM）所指不同
- 需要一个统一分类去对齐这些路线

## 核心方法（分类骨架 · R1 待补）
- **Understanding 路线**：Dreamer 系、latent dynamics、state-space 模型
  - 关注：表征、状态转移、紧凑潜空间
- **Predicting Future 路线**：video diffusion、autoregressive token、game engine 风格
  - 关注：像素 / latent video 上的高保真生成
- 关联到我已读：
  - [[20-Papers/WM/2026-minWM-Real-Time-Video-World-Models]] → Predicting
  - [[20-Papers/WM/2025-Genie-3]] → Predicting
  - [[20-Papers/WM/2025-Hunyuan-GameCraft]] → Predicting
  - （Understanding 阵营我目前缺代表论文笔记）

## 实验与结论
- 综述无实验，关注**对比表 / 分类树 / 挑战清单**

## 我的思考（边读边填）
- 我现有的 [[30-Notes/concepts/World-Model]] 定义偏 Predicting 阵营，需要在 R1 后回写补充 Understanding 视角
- minWM 没解决的：长程一致性、动作可控性、跨场景泛化 — 综述挑战章节是否对齐？
- 关联笔记：[[30-Notes/concepts/World-Model]]、[[30-Notes/concepts/Causal-Forcing]]

## 阅读计划
- [ ] **R1 骨架（30 min）**：摘要 + 目录 + 图表 + 每章首段 → 画分类树
- [ ] **R2 血肉（1.5 h）**：分类章节 + 方法对比表 + 挑战 → 陌生方法建 stub
- [ ] **R3 机会（1-2 h）**：未来方向 → 写 2-3 条研究 hypothesis

## 引用与延伸阅读
- 同主题综述（vault 已有 PDF）：
  - `papers/world-models/WM_Survey_2024_Comprehensive_Embodied_AI.pdf`
  - `papers/world-models/WM_Survey_2025_Robot_Learning_Comprehensive.pdf`
- 通用方法论：[[WAM/05-资源库/如何高效阅读综述论文]]
