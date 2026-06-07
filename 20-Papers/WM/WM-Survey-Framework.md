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

### 1.1 研究背景与动机

**需要找的信息：**
- [ ] 当前机器人学习的主流范式是什么？（提示：foundation models, large-scale pre-training）
- [ ] VLA 模型的现状和局限是什么？（如 RT-2, OpenVLA 等）
- [ ] 为什么单纯的"模仿学习"不够？需要 WM 的根本原因？

**核心论点提取：**
- [ ] 论文如何论证 "WM = 具身智能的内部仿真器" 这个定位？
- [ ] WM 区别于 video generation 的本质是什么？

---

### 1.2 World Model 的角色定位

**需要找的信息：**
- [ ] 论文给出的 WM 三大角色是什么？（提示对应 Section 3/4/5）
  - 角色 1：作为 Policy 的辅助 → §3
  - 角色 2：作为 Simulator → §4
  - 角色 3：作为 Video Generator → §5
- [ ] 这三种角色的边界在哪里？是否有重叠？

**关键图示：**
- [ ] **Figure 1**：综述的整体组织图——找出三大主线的视觉呈现
- [ ] **Figure 2**：representation 类型的概览（visuomotor / VLA 两条主线）

---

### 1.3 与已有综述的差异化定位

**需要找的信息：**
- [ ] 论文如何区分自己与其他 WM 综述？
- [ ] 为什么聚焦 "robot learning" 而非 "video generation"？
- [ ] 该综述独有的视角是什么？（提示：policy-centric, fine-grained taxonomy）

**对比表：**
| 已有综述 | 焦点 | 本综述的差异 |
|---------|------|------------|
| | | |

---

### 1.4 核心贡献（Contributions）

**需要找的信息（论文一般会列 3-4 条）：**
- [ ] 贡献 1：分类法（taxonomy）层面 →
- [ ] 贡献 2：覆盖范围层面 →
- [ ] 贡献 3：评测/数据集层面 →
- [ ] 贡献 4：挑战与未来方向 →

> 提示：在 Introduction 末尾或 §1.5 处通常有 "Our contributions are:" 的列表

---

### 1.5 论文组织（Roadmap）

**需要找的信息：**
- [ ] 论文如何引导读者按顺序阅读？
- [ ] 各章节之间的逻辑关系？

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

### 📌 Introduction 阶段的关键产出

读完 Introduction 后，你应该能回答：

1. **WHY**：为什么 WM 对 robot learning 重要？（一句话总结）
2. **WHAT**：WM 在该综述中的精确定义是什么？
3. **HOW**：综述用什么分类法组织所有方法？
4. **NEW**：与同主题综述相比，新增了什么视角？

---

## 📚 Section 2 — Background（背景）

> 建立基础概念：WM 和 Robot Policy 各是什么

### 2.1 World Model vs. Video Generation Model

**需要找的信息：**
- [ ] 综述对 "World Model" 的定义是什么？（区别于纯视频生成）
- [ ] WM 的核心数学表达是什么？（找公式）
- [ ] Video Generation Model 的定义是什么？
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
