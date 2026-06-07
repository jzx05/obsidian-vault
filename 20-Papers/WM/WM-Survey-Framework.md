---
title: "WM Survey — 信息提取框架"
type: reading-framework
source: "[[Survey-World-Model-for-Robot-Learning]]"
pdf: "[[WM_Survey_2025_Robot_Learning_Comprehensive.pdf]]"
status: in-progress
date: 2026-06-07
---

# 🗺️ World Model for Robot Learning — 阅读框架

> **论文结构速览**：本综述以"WM 在机器人学习中扮演什么角色"为主线，分三大使用场景展开，共 8 个主章节。

---

## 📐 论文骨架（章节地图）

```
1. Introduction        → 为什么需要 WM？
2. Background          → WM 是什么？Robot Policy 是什么？
3. WM for Policy       → WM 如何增强策略学习
4. WM as Simulator     → WM 如何替代真实环境训练
5. WM for Video Gen    → WM 如何用于机器人视频生成
6. WM for Other Apps   → 导航 / 自动驾驶
7. Benchmarks          → 评测体系 + 数据集 + 实验结果
8. Challenges          → 当前瓶颈与未来方向
```

---

## 🚪 Section 1 — Introduction（引言）

> 综述的"立论部分"——为什么写这篇综述？为什么 WM 对机器人重要？

1. VLA的缺陷：long-horizon reasoning, temporal credit assignment, and robustness
2. What is world model ?: a state-transition model that predicts the next state or a sequence of future states from the current state and action
3. WM的分类：

|               | **Model-Based RL（Dreamer 系列）**         | **大规模生成式模型**                                 |     |     |
| ------------- | -------------------------------------- | -------------------------------------------- | --- | --- |
| **核心目标**      | 学习环境动力学，提升 RL 样本效率                     | 生成逼真可控的视频/观测                                 |     |     |
| **表示空间**      | 压缩 latent vector（抽象状态）                 | 像素 / 高维视频帧                                   |     |     |
| **动作条件化**     | 显式紧密耦合 $z'=f(z,a)$                     | 灵活注入（attention / adapter）                    |     |     |
| **输出**        | 抽象状态向量（不一定可视）                          | 光真实感视频帧                                      |     |     |
| **主要用途**      | 在"想象"中做 RL rollout                     | 仿真器 / 数据增强 / 策略先验                            |     |     |
| **计算规模**      | 轻量，单 GPU 可训                            | 重，需百~千 GPU 预训练                               |     |     |
| **代表方法**      | DreamerV1/V2/V3, TD-MPC2               | Genie 3, GameNGen, DIAMOND, minWM            |     |     |
| **concept笔记** | [[30-Notes/concepts/MBRL-World-Model]] | [[30-Notes/concepts/Generative-World-Model]] |     |     |

4. WM for robot learning : robotic policy learning,   planning, simulation, evaluation, data generation.
5. WM in robot area 的分类：1. tighter coupling between predictive modeling and action generation(内化为policy) 2. world models as simulators for *validation*, *post-training*, and *reinforcement learning*  ---》 integrated together

**章节地图（基于 Figure 1）：**
```
§2 Background ─┬─ World Model 定义
               └─ Robot Policy 定义
                     │
                     ▼
              ┌──────┴──────┬──────────┐
              │             │          │
        §3 Policy 增强  §4 Simulator  §5 Video Gen
              │             │          │
              └──────┬──────┴──────────┘
                     │
              §6 其他应用 (导航/驾驶)
                     │
              §7 Benchmarks & Datasets
                     │
              §8 Challenges & Future
```


---

## 📚 Section 2 — Background（背景）

> 建立基础概念：WM 和 Robot Policy 各是什么

### 2.1 World Model vs. Video Generation Model

**需要找的信息：**
- [x] 综述对 "World Model" 的定义是什么？a predictive model of agent-environment dynamics that captures how a robotic or embodied system evolves under actions.
- [x] WM 的核心数学表达是什么？p(xt+1:t+H | xt, at:t+H−1, l)， 状态向量x的形式很多，可以是视频，潜空间甚至式符号，l表示语言指令，例如把杯子移到左前方45度0.5m远处。
- [x] Video Generation Model 的定义是什么？p(vt+1:t+H | ot, at:t+H−1, l),
- [ ] 两者的**核心区别**在哪里？（提示：policy-centric vs. generative）

**关键概念提取：**
| 概念 | 定义 | 页码 |
|------|------|------|
| World Model | | |
| Video Generation Model | | |
| 两者区别 | | |

---

### 2.2 Robot Policy 的两大类型

**需要找的信息：**
- [ ] **Visuomotor Policy** 是什么？代表方法？
- [ ] **VLA (Vision-Language-Action)** 是什么？代表方法？（如 RT-2, OpenVLA）
- [ ] 两类 Policy 的输入输出格式有何不同？
- [ ] 当前 Policy 的主要局限是什么？（WM 要解决的问题）

**关键方法列表：**
| Policy 类型 | 代表方法 | 特点 |
|------------|---------|------|
| Visuomotor | | |
| VLA | | |

---

## 🤖 Section 3 — World Model for Policy（WM 辅助策略学习）

> **核心问题**：WM 如何帮助 robot policy 学得更好？

### 3.1 为什么 WM 能帮助 Policy 学习？

**需要找的信息：**
- [ ] WM 给 policy 带来的核心优势是什么？（预测未来 / 提供监督信号 / 想象轨迹）
- [ ] 论文用什么理论框架解释这种帮助？

---

### 3.2 逆动力学模型 (Inverse Dynamics Models)

**需要找的信息：**
- [ ] 逆动力学方法的基本思路是什么？
- [ ] 代表方法有哪些？
- [ ] 优势和局限？

---

### 3.3 单骨干网络 (Single World Model Backbone)

> WM 和 Policy 共享同一个网络骨干

**需要找的信息：**
- [ ] "单骨干"的核心思想是什么？
- [ ] 代表方法：Unified VLA 有哪些？（如 DreamVLA, UniVLA, CoWVLA）
- [ ] 与解耦方法相比有什么优劣？

**代表方法：**
| 方法名 | 年份 | 核心创新 | WM 类型 |
|--------|------|---------|--------|
| | | | |

---

### 3.4 MoE/MoT 混合专家策略

**需要找的信息：**
- [ ] MoE（Mixture of Experts）在这里如何与 WM 结合？
- [ ] 代表方法？（如 Motus, LingBot-VLA）
- [ ] 相比单骨干的优势？

---

### 3.5 潜在空间世界模型策略 (Latent-Space WM)

**需要找的信息：**
- [ ] 潜在空间 WM 和视频 WM 的核心区别？
- [ ] 代表方法？（如 VLA-JEPA, JEPA-VLA）
- [ ] 计算效率上的优势？

---

### 3.6 统一 VLA 模型

**需要找的信息：**
- [ ] 这与 3.3 的区别是什么？
- [ ] 代表方法？
- [ ] 是否有 Table 对比各方法性能？（→ Table 5, 6）

---

## 🎮 Section 4 — World Model as Simulator（WM 作为仿真器）

> **核心问题**：如何用 WM 代替真实环境进行强化学习训练？

### 4.1 WM 用于强化学习

**需要找的信息：**
- [ ] WM 作为仿真器的基本范式是什么？（找图示，Figure 9）
- [ ] 训练流程：如何在 WM 内部进行 RL？
- [ ] 代表方法？（如 Dreamer 系列, DreamerV1/V2/V3）
- [ ] WM-RL 的核心挑战：模型偏差如何影响 policy？

**关键公式：**
- [ ] WM 的预测目标函数是什么？
- [ ] RL 的优化目标如何在 WM 内定义？

---

### 4.2 规划与模型预测控制

**需要找的信息：**
- [ ] 如何用 WM 做 planning？（MPC 范式）
- [ ] 代表方法？
- [ ] 规划 horizon 对性能的影响？

---

### 4.3 WM 用于数据增强与验证

**需要找的信息：**
- [ ] WM 如何生成合成训练数据？
- [ ] 用于 policy 评估的方式？
- [ ] 相关方法？（如 WorldFomo, WoW）

---

### 4.4 WM 作为评估器

**需要找的信息：**
- [ ] 用 WM 评估 policy 的可行性？
- [ ] 与真实环境评估的对比？

---

## 🎬 Section 5 — World Model for Robotic Video Generation

> **核心问题**：视频生成型 WM 如何服务于具身机器人？

### 5.1 问题设定与范围

**需要找的信息：**
- [ ] 本节的视频 WM 和前几节有何不同定位？
- [ ] 机器人视频生成的特殊要求？（图示 Figure —— 找 Section 5 的 figure）

---

### 5.2 视频生成作为策略学习的想象空间

**需要找的信息：**
- [ ] "Imagination-based" 的核心思路？
- [ ] 代表方法？（如 Dreamer, UniPi, ManipDreamer）
- [ ] Table 2 的对比——找该表格

---

### 5.3 动作可控视频世界模型

**需要找的信息：**
- [ ] 从"视觉真实"到"动作可控"的转变是什么？
- [ ] 代表方法？（如 IRASim, RoboEnvision, RoboMaster, Ctrl-World）
- [ ] 如何实现动作条件化生成？

---

### 5.4 带交互与几何先验的结构化生成

**需要找的信息：**
- [ ] 引入结构先验（接触、几何、多视角）的动机？
- [ ] 代表方法？（如 Mask2IV, TesserAct, RoboVIP）

---

### 5.5 从视频骨干到基础世界模型

**需要找的信息：**
- [ ] "Foundation World Model"的含义是什么？
- [ ] 代表方法？（如 Vid2World, DreamDojo, UnifoLM-WMA-0, Cosmos Predict 2.5）
- [ ] 与前几类方法的核心区别？

---

### 5.6 技术演进与开放挑战

**需要找的信息：**
- [ ] 论文总结的演进路径是什么？（早期→中期→当前前沿）
- [ ] 当前最核心的开放挑战？

---

## 🌐 Section 6 — World Model for Other Applications

### 6.1 WM for Navigation（导航）

**需要找的信息：**
- [ ] 导航与操作的 WM 需求有何不同？
- [ ] 代表方法？（如 Pathdreamer, VISTA, NWM）
- [ ] WM 在导航中的核心价值？

---

### 6.2 WM for Autonomous Driving（自动驾驶）

**需要找的信息：**
- [ ] 自动驾驶对 WM 的特殊需求？（多智能体、长时预测）
- [ ] 代表方法？（如 GAIA-1, DriveDreamer, Drive-WM）
- [ ] 与机器人操作的 WM 有何异同？

---

## 📊 Section 7 — Benchmarks, Datasets, Results

### 7.1 评测体系（三类 Benchmark）

**需要找的信息：**
> 论文将 Benchmark 分为三类，找出每类的定义和代表：

| 类别 | 评测目标 | 代表 Benchmark |
|------|---------|---------------|
| Open-loop 预测质量 | | RBench, EWMBench... |
| Closed-loop 任务效用 | | WorldArena, WorldGym... |
| 物理一致性诊断 | | WorldSimBench, WoW-World-Eval... |

**关键问题：**
- [ ] 为什么视觉真实度（visual realism）不等于 WM 质量？
- [ ] 什么是"action-grounded"评测？

---

### 7.2 训练数据集

**需要找的信息：**
> 论文用多维度分类数据集（见 Table 3 & 4），找出：

- [ ] 数据集的核心维度有哪些？（X-Emb, Action, Obs/3D, Language, M/C）
- [ ] 最重要的数据集有哪些？（2025-2026年）
- [ ] 缺乏什么类型的数据？

---

### 7.3 代表性实验结果

**需要找的信息：**
> 论文按 WM 集成方式分组对比（见 Table 5 & 6）：

| 组别 | 代表方法 | LIBERO 均分 |
|------|---------|-----------|
| Decoupled | UniPi, VidMan | |
| Single-backbone | UVA, VideoPolicyy | |
| MoE/MoT | Motus, LingBot-VLA | |
| Unified VLA | DreamVLA, CoWVLA | |
| Latent-space WM | VLA-JEPA, JEPA-VLA | |

**关键发现：**
- [ ] 哪种集成方式性能最优？
- [ ] Long-horizon 任务上的主要瓶颈？

---

## ⚠️ Section 8 — Challenges & Future Directions

**需要找的信息（逐一填写）：**

| 挑战 | 问题描述 | 现有解决方案 | 未来方向 |
|------|---------|------------|---------|
| 8.1 因果条件缺口 | WM 预测与机器人动作的因果脱节 | | |
| 8.2 效率瓶颈 | 推理/训练计算代价高 | | |
| 8.3 多模态感知瓶颈 | 过度依赖视觉，忽略触觉等 | | |
| 8.4 ??? | | | |
| 8.5 ??? | | | |

> 提示：继续阅读 Section 8 补全后几项挑战

---

## 🔗 关键方法速查表

> 随阅读进展填写，按首次提出时间排序

| 方法名 | 年份 | 所在章节 | 核心贡献一句话 | WM 类型 |
|--------|------|---------|-------------|--------|
| Dreamer | 2018 | §4.1 | 首个在潜在空间做完整 MBRL 的框架 | Latent |
| DreamerV3 | 2023 | §4.1 | | |
| UniPi | 2023 | §3/§5 | | |
| VLA-JEPA | 2025 | §3.5 | | |
| DreamVLA | 2025 | §3.3 | | |
| Genie 3 | 2026 | | | |
| minWM | 2026 | | | |

---

## 📝 阅读进度追踪

- [x] 摘要 & Introduction（了解整体定位）
- [x] Section 2（背景概念）
- [ ] Section 3（WM for Policy）→ **当前位置**
- [ ] Section 4（WM as Simulator）
- [ ] Section 5（WM for Video）
- [ ] Section 6（其他应用）
- [ ] Section 7（Benchmarks）
- [ ] Section 8（挑战与未来）

---

## 💡 核心问题导航

> 带着这些问题阅读，效率更高：

1. **统一问题**：WM 在这三大场景中的作用有何本质区别？
2. **对比问题**：Latent-space WM vs. Video WM，分别适合什么场景？
3. **评测问题**：为什么 "视觉真实 ≠ 具身有用"？论文如何解决这个矛盾？
4. **效率问题**：WM 最大的计算瓶颈在哪里？目前有什么缓解方案？
5. **趋势问题**：Foundation World Model 和之前的方法有何本质突破？
