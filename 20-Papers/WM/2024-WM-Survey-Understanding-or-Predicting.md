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

## 核心方法（分类骨架 · R1）

> 综述定义 World Model = a unified model that can replicate its fundamental dynamics of the world
> 二分法依据：Implicit representation of the external world：偏 understanding
> 			Future predictions of the external world：偏 predicting

### 🌳 分类树

- **Understanding（理解世界）** — 学"世界是什么"
  - **子类 1**：?
    - 代表：?
    - § ?
  - **子类 2**：?
    - 代表：?
    - § ?

- **Predicting Future（预测未来）** — 学"接下来发生什么"
  - **子类 1**：?
    - 代表：?（已读：[[20-Papers/WM/2026-minWM-Real-Time-Video-World-Models|minWM]] / [[20-Papers/WM/2025-Genie-3|Genie 3]] / [[20-Papers/WM/2025-Hunyuan-GameCraft|GameCraft]] 落在哪个子类？）
    - § ?
  - **子类 2**：?
    - § ?

### 🔑 综述独有的关键概念（生词清单 · 留给 R2 深入）
- ? — 一行解释，§ ?
- ? — 一行解释，§ ?

### ❓ R1 阶段疑问（带到 R2 解决）
- 综述定义的 World Model 与 [[30-Notes/concepts/World-Model]] 是否一致？
- minWM / Genie / GameCraft 是否被综述提及？放在哪一节？
- LLM-as-WM 这条线综述如何处理？
- ?

### 📊 关键图表速记
- Fig ? （§ ?）：? — 记忆点 = ?
- Tab ? （§ ?）：方法对比表 — 列维度 = ?
- Fig ? （§ ?）：挑战 / 未来方向

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
