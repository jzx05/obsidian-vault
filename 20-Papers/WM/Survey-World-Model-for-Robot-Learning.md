---
title: "World Model for Robot Learning: A Comprehensive Survey"
type: survey-outline
tags:
  - world-model
  - robot-learning
  - survey
  - embodied-ai
created: 2026-06-07
status: drafting
---

# World Model for Robot Learning: A Comprehensive Survey

> 综述框架草稿。围绕 **World Model (WM)** 在 **机器人学习 (Robot Learning)** 中的角色,从基础理论、表征建模、训练范式、下游任务、评测基准到未来方向进行系统性梳理。

---

## 1. Introduction

- **1.1 Motivation**:为什么机器人需要世界模型?(Sim2Real gap、数据稀缺、长程规划、安全性、想象式 RL)
- **1.2 What is a World Model?**:从 Ha & Schmidhuber (2018) 到现代 video/3D/multimodal world models 的定义演化。
- **1.3 Why Now?**:基础模型 (VLM/Diffusion/Video Generation) 与具身智能 (Embodied AI) 的交汇。
- **1.4 Scope of this Survey**:聚焦 **机器人决策与控制** 中的 WM,区别于纯生成式视频模型。
- **1.5 Contributions & Taxonomy**:提出统一分类法 + 时间线 + benchmark 汇总。
- **1.6 Comparison with Related Surveys**:与 model-based RL、video generation、embodied AI 等综述的差异。

---

## 2. Preliminaries & Background

- **2.1 Formalism**
  - POMDP / MDP 设定
  - 状态、观测、动作、奖励的形式化定义
  - 前向模型 vs 逆向模型 vs 联合模型
- **2.2 Classical Model-Based RL**:Dyna、PILCO、iLQR/MPC 中的动力学模型。
- **2.3 Deep World Models 的发展脉络**
  - World Models (Ha & Schmidhuber, 2018)
  - PlaNet / Dreamer 系列 (V1/V2/V3)
  - MuZero & Latent Planning
- **2.4 Foundation Models 视角**:Video Diffusion、VLA、3D Generation 与 WM 的关系。
- **2.5 Robot Learning 范式**:IL、RL、Offline RL、VLA Policy。

---

## 3. Taxonomy of World Models for Robotics

> 提出一个多维分类体系,后续章节按此组织。

- **3.1 By Representation (表征空间)**
  - Latent-state WM (Dreamer-like)
  - Pixel/Video WM (Sora-like, Genie)
  - 3D / Geometry-aware WM (NeRF, 3DGS, Occupancy)
  - Multimodal / Symbolic WM (Language-grounded)
- **3.2 By Generative Backbone (生成范式)**
  - RSSM / Recurrent
  - Transformer / Autoregressive
  - Diffusion / Flow Matching
  - Hybrid (e.g., causal diffusion, self-forcing)
- **3.3 By Conditioning (条件输入)**
  - Action-conditioned
  - Goal/Language-conditioned
  - Multi-agent / Multi-view
- **3.4 By Usage (使用方式)**
  - Imagination for Policy Learning (Dreamer-style)
  - Planner / MPC backbone
  - Data Engine (Synthetic data generation)
  - Evaluator / Critic / Reward Model
- **3.5 By Embodiment Scale**
  - Manipulation
  - Locomotion / Humanoid
  - Autonomous Driving
  - Mobile / Navigation

---

## 4. Representations: How to Model the World

- **4.1 Latent-State Models**
  - RSSM 及其变体 (Dreamer V1-V3, DayDreamer, TD-MPC2)
  - 离散 vs 连续 latent
  - 信息瓶颈与 disentanglement
- **4.2 Pixel-Space / Video World Models**
  - Action-conditioned video prediction (GameGAN, IRIS, GAIA-1/2)
  - 大规模 video foundation WM:Sora、Genie 1/2/3、V-JEPA、Cosmos
  - Real-time / streaming WM:Self-Forcing, Causal Forcing, minWM
- **4.3 3D & Geometry-Aware Models**
  - Neural Radiance / Gaussian Splatting WM
  - Occupancy / Voxel-based WM(自动驾驶)
  - 4D dynamic scene models
- **4.4 Multimodal & Language-Grounded WM**
  - VLM-as-WM (RoboFlamingo, RT-2 衍生)
  - Symbolic / PDDL hybrid
- **4.5 Physics-Informed WM**
  - 可微物理 + 神经网络
  - Constraint-aware dynamics
- **4.6 Discussion**:不同表征的精度/可控性/计算成本权衡。

---

## 5. Learning Paradigms

- **5.1 Self-Supervised Pretraining**:masked prediction、JEPA、对比学习。
- **5.2 Action-Conditioned Generative Training**:teacher forcing、scheduled sampling、diffusion forcing。
- **5.3 Causal / Streaming Training**:Self-Forcing、DMD、minWM 等长程一致性方法。
- **5.4 Multi-task / Cross-Embodiment Training**:Open-X、DROID 等大规模数据混训。
- **5.5 Sim-to-Real & Domain Randomization**:在 WM 中显式建模 domain gap。
- **5.6 Continual / Online World Model Adaptation**:on-robot fine-tuning、test-time adaptation。

---

## 6. World Models for Policy Learning

- **6.1 Imagination-Based RL**
  - Dreamer-style on-policy in latent
  - 长 horizon rollout 的稳定性
- **6.2 Model Predictive Control (MPC) with Learned Dynamics**
  - Visual MPC、TD-MPC、DiffPhysics
- **6.3 World Model as Differentiable Simulator**
  - 反向传播至策略 (analytic policy gradient)
- **6.4 World Model as Data Engine**
  - 合成 demonstrations 训练 IL/VLA
  - Counterfactual rollouts
- **6.5 World Model as Evaluator / Reward Model**
  - Generative reward、preference 评分
  - 用 WM 做 policy verification & safety screening
- **6.6 World Model + VLA Co-design**
  - 共享 backbone:Genie-Robotics、UniSim、1X World Model

---

## 7. Application Domains

- **7.1 Robotic Manipulation**:tabletop、双臂、灵巧手。
- **7.2 Locomotion & Humanoid**:四足、人形全身控制。
- **7.3 Autonomous Driving**:GAIA、DriveDreamer、Vista、Cosmos。
- **7.4 Indoor Navigation & Mobile Manipulation**。
- **7.5 Multi-Agent / Human-Robot Interaction**。
- **7.6 Surgical / Soft Robotics**(变形体、流体)。

---

## 8. Datasets, Benchmarks & Evaluation

- **8.1 Datasets**
  - Robot:Open-X-Embodiment、DROID、RH20T、AgiBot World
  - Driving:nuScenes、NAVSIM、Waymo
  - Egocentric:Ego4D、EgoExo4D
- **8.2 Benchmarks**
  - Manipulation:CALVIN、LIBERO、RoboCasa、SimplerEnv
  - Locomotion:Isaac Lab、HumanoidBench
  - WM-specific:WorldModelBench、EvalCrafter、VBench
- **8.3 Evaluation Metrics**
  - Pixel quality (FVD, PSNR, LPIPS)
  - Action-conditional consistency
  - Physical plausibility (object permanence、collision)
  - Downstream policy success rate(最关键)
- **8.4 Open Problems in Evaluation**:visual fidelity ≠ usefulness for control。

---

## 9. Key Challenges

- **9.1 Long-Horizon Consistency & Drift**
- **9.2 Action Controllability & Faithfulness**(动作真的有效?)
- **9.3 Physical & Causal Reasoning**(刚体、接触、因果)
- **9.4 Real-Time Inference**(>10Hz on-robot)
- **9.5 Generalization to Novel Embodiments / Scenes**
- **9.6 Data Scaling Laws for WM**
- **9.7 Safety, Hallucination & Out-of-Distribution**
- **9.8 Evaluation Gap**:生成质量 vs 决策有用性

---

## 10. Future Directions

- **10.1 Unified Foundation World Models**:跨任务、跨本体、跨模态。
- **10.2 World Models + Reasoning (CoT, RL post-training)**
- **10.3 Interactive & Long-Lived WM**(memory、persistent world state)
- **10.4 Neuro-Symbolic & Physics-Grounded WM**
- **10.5 On-Device / Edge WM**(模型压缩、蒸馏)
- **10.6 World Model as Scientific Tool**(发现物理规律、做实验设计)
- **10.7 Towards AGI-Level Embodied Agents**

---

## 11. Conclusion

- 总结 taxonomy 与核心 insight
- 强调 WM 是 **具身智能的 "internal simulator"**
- Call to action:统一 benchmark、可复现 pipeline、跨学科合作

---

## Appendix

- **A. Timeline of Representative Works**(2018–2026)
- **B. Comparison Tables**(模型 × 表征 × 数据 × 任务 × 指标)
- **C. Notation Summary**
- **D. Open-Source Codebase List**

---

## 写作建议 / TODO

- [ ] 收集 2024–2026 关键论文(已有 [[20-Papers/WM/2025-Genie-3]]、[[20-Papers/WM/2026-minWM-Real-Time-Video-World-Models]]、[[20-Papers/WM/2024-Self-Forcing]]、[[20-Papers/WM/2024-Causal-Forcing]] 等可引用)
- [ ] 制作 taxonomy 图(可用 Excalidraw)
- [ ] 制作时间线(2018 Ha&Schmidhuber → 2026 当前)
- [ ] 每个分类至少 3 个代表性工作 + 1 个对比表
- [ ] Section 6 与 Section 7 是核心卖点,需要最详细
- [ ] 与现有综述 diff:强调 **robot learning** 视角而非纯 video generation
