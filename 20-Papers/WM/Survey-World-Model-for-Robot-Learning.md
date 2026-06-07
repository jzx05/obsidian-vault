---
title: "World Model for Robot Learning: A Comprehensive Survey"
authors:
  - TBD
year: 2026
venue: TBD (draft outline)
tags:
  - paper
  - survey
  - world-model
  - robot-learning
  - embodied-ai
  - model-based-rl
status: drafting
rating: 0
date-read: 2026-06-07
url:
project:
pdf: "[[WM_Survey_2025_Robot_Learning_Comprehensive.pdf]]"
zotero-key:
---

# World Model for Robot Learning: A Comprehensive Survey

> 📝 综述写作大纲(草稿)
> 🎯 目标:系统梳理 **World Model (WM)** 在 **机器人学习 (Robot Learning)** 中的角色与应用
> 📂 索引:[[20-Papers/WM/_index]]

## TL;DR
> 本综述以 **机器人决策与控制** 为主线,围绕世界模型的 **表征 / 训练范式 / 使用方式 / 应用领域 / 评测体系** 五个维度建立统一分类法,梳理从 Dreamer (2018) 到 Genie 3 / minWM (2026) 的技术演进,聚焦 WM 作为"具身智能内部仿真器"的角色,与纯 video generation 综述形成差异化定位。

## 问题与动机
- 机器人学习面临的核心痛点
  - 真实数据稀缺 + Sim2Real gap
  - 长程规划需要可想象的"未来"
  - 安全约束下的反事实推理
- 现有综述的不足
  - Model-based RL 综述偏经典控制,缺少 foundation model 视角
  - Video generation 综述偏视觉质量,缺少决策有用性视角
  - **缺一个把 WM 视为机器人"内部仿真器"的系统综述**
- 本文定位
  - 以 **robot learning** 为锚点,统一 latent / pixel / 3D / multimodal WM
  - 强调 **生成保真度 ≠ 决策有用性** 的评估鸿沟

## 核心方法(综述结构总览)

### Section 1 — Introduction
- Motivation:为什么机器人需要世界模型
- Definition:从 Ha & Schmidhuber (2018) 到 foundation-WM 的演化
- Scope:聚焦机器人决策,区别于纯视频生成
- Contributions:统一 taxonomy + 时间线 + benchmark 汇总
- 与已有综述对比

### Section 2 — Preliminaries
- POMDP / MDP 形式化(状态、观测、动作、奖励)
- Forward / Inverse / Joint model
- 经典 MBRL:Dyna、PILCO、iLQR/MPC
- Deep WM 演化:World Models → PlaNet → Dreamer V1-V3 → MuZero
- 与 Foundation Model (Video Diffusion / VLA / 3DGS) 的关系

### Section 3 — Taxonomy(五维分类)
| 维度 | 类别 |
|------|------|
| Representation | Latent / Pixel-Video / 3D-Geometry / Multimodal-Symbolic |
| Generative Backbone | RSSM / Transformer-AR / Diffusion / Hybrid |
| Conditioning | Action / Goal-Language / Multi-agent / Multi-view |
| Usage | Imagination-RL / MPC / Data Engine / Evaluator |
| Embodiment | Manipulation / Locomotion / Driving / Navigation |

### Section 4 — Representations
- **4.1 Latent-State**:RSSM 系列、Dreamer V1-V3、TD-MPC2
- **4.2 Pixel/Video**:GAIA-1/2、Sora、Genie 1/2/3、V-JEPA、Cosmos、minWM、Self-Forcing、Causal Forcing
- **4.3 3D / Geometry-aware**:NeRF-WM、3DGS-WM、Occupancy-WM
- **4.4 Multimodal & Language-grounded**:VLM-as-WM、PDDL hybrid
- **4.5 Physics-informed**:可微物理、约束感知动力学
- **4.6 对比**:精度 vs 可控性 vs 计算成本

### Section 5 — Learning Paradigms
- Self-supervised pretraining (JEPA、masked prediction)
- Action-conditioned generative training (teacher forcing、diffusion forcing)
- Causal / Streaming training (Self-Forcing、DMD、minWM)
- Multi-task / Cross-embodiment (Open-X、DROID)
- Sim-to-Real & Domain Randomization
- Continual / Online adaptation

### Section 6 — World Models for Policy Learning ⭐
- Imagination-based RL (Dreamer-style)
- MPC with learned dynamics (Visual MPC、TD-MPC)
- Differentiable simulator (analytic policy gradient)
- Data Engine (synthetic demos for IL/VLA)
- Evaluator / Reward model / Safety screener
- WM + VLA co-design (UniSim、1X World Model、Genie-Robotics)

### Section 7 — Applications
- Manipulation / Bimanual / Dexterous
- Locomotion & Humanoid
- Autonomous Driving (DriveDreamer、Vista、GAIA、Cosmos)
- Indoor Navigation & Mobile Manipulation
- Multi-agent / HRI
- Surgical / Soft Robotics

### Section 8 — Datasets, Benchmarks & Evaluation
- 数据集:Open-X-Embodiment、DROID、RH20T、AgiBot World、Ego4D
- Benchmark:CALVIN、LIBERO、SimplerEnv、HumanoidBench、WorldModelBench
- 指标:FVD/PSNR/LPIPS、action consistency、physical plausibility、**downstream success rate**
- 评测开放问题:visual fidelity ≠ usefulness for control

### Section 9 — Challenges
- Long-horizon consistency & drift
- Action controllability & faithfulness
- Physical / causal reasoning
- Real-time inference (>10Hz on-robot)
- Cross-embodiment generalization
- Data scaling laws
- Safety, hallucination, OOD
- Evaluation gap

### Section 10 — Future Directions
- Unified foundation world models
- WM + Reasoning (CoT, RL post-training)
- Interactive & long-lived WM (persistent memory)
- Neuro-symbolic & physics-grounded
- On-device / edge WM
- WM as scientific tool
- Towards AGI-level embodied agents

### Section 11 — Conclusion + Appendix
- A. Timeline of representative works (2018–2026)
- B. Comparison tables (model × representation × data × task × metric)
- C. Notation summary
- D. Open-source codebase list

## 实验与结论(综述论证策略)
- 用 **timeline 图** + **taxonomy 树** + **comparison table** 三类视觉元素串联全文
- 每个分类至少 3 个代表性工作 + 1 个横向对比表
- Section 6 / 7 是核心卖点,需要最详细;其余章节 supporting

## 我的思考

### ✅ 优点
- **差异化定位明确**:从 robot learning 视角切入,而非 video generation
- **Taxonomy 在第 3 章前置**:后续叙述围绕固定坐标轴展开,降低读者认知负担
- **Evaluation 单独成章**:点出"生成质量 ≠ 决策有用性"的领域共识盲点

### ⚠️ 待解决
- 论文边界:Foundation model (Sora 类) 是否纳入?建议:**只纳入与机器人/决策直接相关的 demo**
- 与 model-based RL 综述的区分度需要在 Intro 显式说明
- 篇幅控制:11 章可能过长,可考虑把 Section 5 / 9 合并

### 🚀 可延伸方向
- 配套一份 awesome-world-model-for-robot-learning 仓库
- 制作交互式 taxonomy 网页(用 D3 / Observable)
- 针对 Section 6 单独发一篇 short position paper

## 引用与延伸阅读

### 已有相关笔记(可作为引用骨架)
- [[20-Papers/WM/2025-Genie-3]] — Section 4.2 / 6.6 引用
- [[20-Papers/WM/2026-minWM-Real-Time-Video-World-Models]] — Section 4.2 / 5.3 引用
- [[20-Papers/WM/2024-Self-Forcing]] — Section 5.3 引用
- [[20-Papers/WM/2024-Causal-Forcing]] — Section 5.3 引用
- [[20-Papers/WM/2024-Causal-ODE-CD-Init]] — Section 5.3 引用
- [[20-Papers/WM/2024-DMD]] — Section 5.3 引用
- [[20-Papers/WM/2025-Hunyuan-GameCraft]] — Section 4.2 引用
- [[20-Papers/WM/2025-PRoPE]] — Section 4.2 引用

### 待新建概念笔记
- [[30-Notes/concepts/World-Model]]
- [[30-Notes/concepts/Model-Based-RL]]
- [[30-Notes/concepts/Imagination-Learning]]
- [[30-Notes/concepts/Sim2Real]]
- [[30-Notes/methods/Differentiable-Simulator]]
- [[30-Notes/methods/Latent-Imagination]]

### 同主题索引
- [[20-Papers/WM/_index]]
- [[50-Areas/WAM]]

## 写作 TODO

- [ ] 收集 2024–2026 关键论文(目标 150+ refs)
- [ ] 制作 taxonomy 图(Excalidraw)
- [ ] 制作 timeline 图(2018 Ha&Schmidhuber → 2026)
- [ ] 每章草稿 → 1500–3000 字
- [ ] Section 6 / 7 优先详细化
- [ ] Section 8 benchmark 表格补全
- [ ] 与 model-based RL / video generation 综述对照表
