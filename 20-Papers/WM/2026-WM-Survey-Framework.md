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

1. **VLA 的缺陷**：long-horizon reasoning, temporal credit assignment, and robustness
2. **What is world model?** — a state-transition model that predicts the next state or a sequence of future states from the current state and action
3. **WM 的分类**：

|               | **Model-Based RL（Dreamer 系列）**          | **大规模生成式模型**                                  |
| ------------- | --------------------------------------- | --------------------------------------------- |
| **核心目标**      | 学习环境动力学，提升 RL 样本效率                      | 生成逼真可控的视频/观测                                  |
| **表示空间**      | 压缩 latent vector（抽象状态）                  | 像素 / 高维视频帧                                    |
| **动作条件化**     | 显式紧密耦合 $z'=f(z,a)$                      | 灵活注入（attention / adapter）                     |
| **输出**        | 抽象状态向量（不一定可视）                           | 光真实感视频帧                                       |
| **主要用途**      | 在"想象"中做 RL rollout                      | 仿真器 / 数据增强 / 策略先验                             |
| **计算规模**      | 轻量，单 GPU 可训                             | 重，需百~千 GPU 预训练                                |
| **代表方法**      | DreamerV1/V2/V3, TD-MPC2                | Genie 3, GameNGen, DIAMOND, minWM             |
| **concept笔记** | [[30-Notes/concepts/MBRL-World-Model]]  | [[30-Notes/concepts/Generative-World-Model]]  |

4. **WM for robot learning**：robotic policy learning, planning, simulation, evaluation, data generation.
5. **WM in robot area 的分类**：
   1. Tighter coupling between predictive modeling and action generation（内化为 policy）
   2. World models as simulators for *validation*, *post-training*, and *reinforcement learning*

   → 两者最终 integrated together

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

- [x] 综述对 "World Model" 的定义是什么？
  - A predictive model of agent-environment dynamics that captures how a robotic or embodied system evolves under actions.
- [x] WM 的核心数学表达是什么？
  - ==p(x~t+1:t+H~ | x~t~, a~t:t+H-1~, l)==
  - 状态向量 x 的形式很多，可以是视频、潜空间甚至符号；l 表示语言指令，例如"把杯子移到左前方 45° 0.5m 远处"。
- [x] Video Generation Model 的定义是什么？
  - p(v~t+1:t+H~ | o~t~, a~t:t+H-1~, l)
- [x] 两者的**核心区别**在哪里？
  - WM in robot = ==预测未来== + ==这个预测能被机器人用来做决策==

**Action-conditioned video generation model**（动作条件化的视频生成模型）— WM 和 video generation model 交叉的子类 [[action-conditioned video generation model]]

```
World Model（广义）
│
├─ 非视频的 world model（latent / symbolic / physical）
│   └─ 本文提及但不重点讨论
│
└─ Video-based World Model ← 本文主线
    │
    ├─ 非 action-conditioned（纯视频预测）
    │   └─ 对决策价值有限
    │
    └─ Action-conditioned ← 本文核心关注
        ├─ 低层控制作为条件（控制指令）
        └─ 高层语言作为条件（任务描述）
```

### 2.2 Robot Policy 的两大类型

> p(a~t+1:t+k~ | o~t~, l)

**需要找的信息：**

- [x] **Visuomotor Policy** 是什么？代表方法？
  - 专用视觉运动策略：一种任务专用的端到端网络，直接从视觉观测映射到动作轨迹，中间不经过语言环节。
  - Diffusion Policy 是其代表方法。
- [x] **VLA (Vision-Language-Action)** 是什么？代表方法？
  - 在大规模视觉-语言模型（VLM）上微调，融入机器人轨迹数据，使模型同时理解图像、语言，并输出动作（如 RT-2, OpenVLA）。
- [x] 两类 Policy 的输入输出格式有何不同？

```
┌───────────────────────────────────────────────────────┐
│  Visuomotor Policy                                    │
│                                                       │
│  [Image o_t] ──→ [ Encoder + Policy Net ] ──→ [a_t]  │
│                   (Diffusion / ACT / ...)             │
│                                                       │
│  特点: 端到端, 无语言, 任务专用                         │
└───────────────────────────────────────────────────────┘

┌───────────────────────────────────────────────────────┐
│  VLA (Vision-Language-Action)                         │
│                                                       │
│  [Image o_t] ─┐                                      │
│               ├─→ [ VLM Backbone ] ─→ [Action Head]  │
│  [Language l] ─┘   (RT-2/OpenVLA/π0)      ──→ [a_t]  │
│                                                       │
│  特点: 预训练VLM, 支持语言指令, 通用性强                │
└───────────────────────────────────────────────────────┘
```
- [x] 当前 Policy 的主要局限是什么？

**关键方法列表：**

| Policy 类型 | 代表方法 | 特点 |
|------------|---------|------|
| Visuomotor | [[30-Notes/concepts/Diffusion-Policy\|Diffusion Policy]], ACT, RDT-1B | 端到端图像→动作；建模多模态动作分布；Action Chunking |
| VLA | RT-2, OpenVLA, π0 | 预训练 VLM backbone + action head；支持语言指令 |

---

## 🤖 Section 3 — World Model for Policy（WM 辅助策略学习）

> **核心问题**：WM 如何帮助 robot policy 学得更好？

### 3.1 为什么 WM 能帮助 Policy 学习？

**需要找的信息：**

- [x] WM 给 policy 带来的核心优势是什么？

| 优势 | 含义 | 价值 |
|------|------|------|
| Foresight | 执行前预判后果 | 避免盲目动作，减少真实环境错误 |
| Imagination-driven Planning | 想象多条未来轨迹并比较 | 在脑内试错，选出最优方案 |
| Data Amplification | 合成额外训练轨迹 | 用想象数据训练更鲁棒的策略 |

- [x] 论文用什么理论框架解释这种帮助？
  - 概率统一视角（Probabilistic Unification）：p(o~t+1:t+k~, a~t+1:t+k~ | o~t~, l)
  - 把 Policy、Passive WM、Controllable WM、Inverse Dynamics 统一在同一个框架下

---

### 3.2 逆动力学模型 (Inverse Dynamics Models) [[IDM-pipline]]

**需要找的信息：**

- [x] 逆动力学方法的基本思路是什么？
  - 看到"当前画面"和"预测的未来画面"，推断出"这中间发生了什么动作"。
- [x] 代表方法有哪些？

| 方法                                | 未来预测方式                 | 特点                   |
| --------------------------------- | ---------------------- | -------------------- |
| ==UniPi (Du et al., 2023)==       | 显式视频 rollout           | 最早的代表性工作之一           |
| VidMan (Wen et al., 2024)         | 显式视频 rollout           | 视频生成 + IDM 分离        |
| Vidar (Feng et al., 2025)         | 显式视频 rollout           | —                    |
| Gen2Act (Bharadhwaj et al., 2025) | 显式 human-video rollout | 人类演示视频作为条件           |
| VPP (Hu et al., 2025)             | 隐式预测特征                 | 不生成像素，生成 latent 预测特征 |
| Video2Act (Jia et al., 2025b)     | 隐式预测特征                 | —                    |
| MimicVideo (Pai et al., 2025)     | 隐式视觉规划                 | —                    |

- [x] 优势和局限？
  - **优势**：① 模块化 ② 可解释性强 ③ 可以利用大规模视频预训练
  - **缺点**：① 两阶段训练，不是端到端优化，WM 不知道"什么样的预测对 IDM 最有用" ② 推理时开销大

---

### 3.3 单骨干网络 (Single World Model Backbone)

> **一句话理解**：不再把"世界模型"和"策略"分成两个模型，而是让一个统一 backbone 同时学"未来会怎样"和"我该怎么做"。

#### 核心思想

从 3.2 的 **predict first, then act**（先预测，再行动）
走向 3.3 的 **predict and act in one backbone**（在一个骨干里同时预测和行动）

**统一形式**：
- 令 x = [z~v~; z~a~]，其中 z~v~ 为未来视觉表示，z~a~ 为动作表示
- 统一 backbone f~θ~ 在当前观测 o~t~ 和指令 l 条件下，对被扰动的 x 进行统一去噪/生成
- 训练目标 L~unified~ 可以是：diffusion 噪声预测 / flow matching 速度场预测 / 离散 masked token 预测

```
┌─────────────────────────────────────────────────────┐
│  3.2 解耦式 IDM-style                               │
│                                                     │
│  o_t → [World Model] → 未来视频 → [IDM] → 动作     │
│         (模块 1)                    (模块 2)         │
└─────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────┐
│  3.3 单 backbone 统一式                              │
│                                                     │
│  o_t + l → [ 统一生成模型 f_θ ] → [z_v; z_a]       │
│             同时建模未来视觉和动作                    │
└─────────────────────────────────────────────────────┘
```

#### 与 3.2 的核心区别

| 维度 | 3.2 解耦式 IDM | 3.3 单骨干统一式 |
|------|---------------|-----------------|
| 架构 | WM 模块 + IDM 模块串联 | 单一 backbone 联合生成 |
| 关系 | 视频预测 → 动作反推（串行） | 视觉预测 & 动作生成（并行共生） |
| 目标对齐 | WM 不知道什么预测对控制有用 | 预测和控制共同优化，目标一致 |
| 先验利用 | 视频预训练只服务于"想象" | 视频 backbone 的时空先验直接注入控制 |
| WM 角色 | 上游工具模块 | 就是 policy backbone 本身 |

#### 设计合理性：为什么视频 backbone 适合做控制？

视频生成 backbone 在预训练时已学到三类对控制有价值的先验：

| 先验类型 | 含义 | 对控制的价值 |
|---------|------|------------|
| Motion continuity | 运动连续性 | 动作轨迹平滑，避免突变 |
| Temporal causality | 时间因果性 | 当前动作决定未来状态 |
| Approximate physical dynamics | 近似物理动力学 | 隐式理解物体交互 |

> **对比**：VLM backbone 擅长图文对齐、语义对应；Video backbone 擅长"一帧怎么变成下一帧"、"动作怎么在时间上产生后果"。

> **作者的谨慎判断**：视频预训练 backbone 是否一定比同等规模的 VLM backbone 更适合机器人控制，目前仍是开放问题。现有结果更像是"有前景的归纳偏置证据"，而非最终定论。

#### 想解决 3.2 的什么痛点？

1. **减少模块割裂** — 生成未来的模块不一定知道"什么样的未来最有利于控制"；统一训练让预测和控制目标对齐
2. **让视觉预测直接服务动作生成** — 动作生成嵌入同一个 denoising 过程，未来表征天然更贴近控制需求
3. **更好吸收视频预训练的时空先验** — policy 直接建立在"会时序推演"的模型上

#### 代表方法

| 方法 | 核心特点 |
|------|---------|
| UVA | 联合 video-action latent space；推理时可通过轻量 decoder 绕过显式视频生成 |
| UWA | 视频和动作在同一 transformer 里扩散；用 modality-specific timesteps 区分模态；可边缘化视觉分支 |
| VideoVLA / VideoPolicy | 直接把预训练视频生成器改造成控制模型；视频模型本体就是 backbone |
| Cosmos Policy | 收紧视觉预测与动作控制的距离；视觉分支更像隐藏推理流 |
| DreamZero | 未来视觉当作 latent predictive process；推理时只取动作输出 |
| UD-VLA | 统一去噪过程 |
| GigaWorld-Policy | 大规模统一世界模型策略 |

**发展趋势**：视觉分支越来越不像"必须输出的视频结果"，而更像"指导动作的隐藏推理流"。重点不是生成更长更清晰的视频，而是让视频世界建模更深地内化进控制本身。

#### 三个关键词总结

1. **从解耦到统一** — 从"WM 先预测 → policy 再出动作"变成"同一 backbone 同时学未来和动作"
2. **从显式视频输出到隐式时空先验** — 重点不再是完整画出未来视频，而是让视频 backbone 的时空知识进入策略学习
3. **从"WM 是工具"到"WM 就是骨干"** — 3.2 里 world model 是上游工具；3.3 里 world model backbone 本身就成了 policy backbone

---

### 3.4 MoE/MoT 混合专家策略

> **一句话理解**：别把视频预测和动作控制彻底融成一个模型，也别完全拆成两个模型，而是让"视频专家"和"动作专家"各司其职、反复协作。

#### 定位：三种范式的光谱

```
3.2 解耦式          3.4 多专家协作          3.3 单骨干统一
(两个独立部门)    (多个专业小组协作)      (一个大部门共脑)
松 ─────────────────── 中 ─────────────────── 紧
```

#### 核心思想

- MoE = Mixture of Experts / MoT = Mixture of Transformers
- 保留多个功能不同的 expert stream（video / action / language），通过 attention 机制在层间持续耦合
- 高层形式：h~v~^l+1^, h~a~^l+1^ = F~mix~^l^(h~v~^l^, h~a~^l^; o~t~, l)
- 交互方式可以是：joint attention / cross-attention / shared-attention fusion

#### 为什么从 3.3 退一步？

3.3 假设"全参数共享最优"，但视频预测和动作生成存在本质差异：

| 差异维度 | 视频预测 | 动作生成 |
|---------|---------|---------|
| 时间频率 | 较稠密的连续时序变化 | 高频、精确、低延迟信号 |
| 表示尺度 | 高维视觉 token / latent | 低维控制命令 |
| 优化目标 | 时空一致性 | 可执行性和控制精度 |

> **作者原话**：full parameter sharing is not always optimal.

#### 与 3.3 的核心区别

| 维度 | 3.3 Single Backbone | 3.4 MoE/MoT Expert |
|------|--------------------|--------------------|
| 架构 | 一个共享 backbone | 多个专门 expert |
| 参数共享 | 高 | 部分共享、部分分工 |
| 视觉/动作关系 | 同一 backbone 内统一建模 | 保留独立流，attention 反复交互 |
| 假设 | 统一骨干最有效 | 专门化 + 交互可能更优 |
| 重点 | Fully unified generation | Deeply interacting specialization |

#### 关键转变：视频分支从"输出"变成"指导流"

- 早期：视频分支 = 真要生成完整未来视频 → 再根据视频做动作
- 3.4 趋势：视频分支 = **predictive latent process**（预测性潜在流）
  - 不一定要渲染完整视频
  - hidden states 用来指导动作分支
  - 重点从"生成视频结果"转向"利用视频分支内部的时空表征"

#### 代表方法

**第一类：并行 expert coupling**

| 方法 | 特点 |
|------|------|
| GE-Act | 预训练 video diffusion backbone + 轻量 action branch；通过 deep cross-attention 注入视频 latent；不需在线完整渲染视频 |

**第二类：Mixture-of-Transformers 式深交互**

| 方法 | 特点 |
|------|------|
| Motus | understanding expert + video generation expert + action expert，标准 MoT |
| LingBot-VA | video token 和 action token 交织进共享 autoregressive 序列；dual-stream MoT + shared attention |
| BagelVLA | 语言规划 + 视觉预测 + 动作生成放入同一执行循环，适合长时程 manipulation |
| DiT4DiT | 用视频分支中间去噪特征来指导动作预测 |
| Fast-WAM | 认为收益更多来自训练时视频协同训练，而非推理时显式想象未来 |

**第三类：Latent-space expertization**

| 方法 | 特点 |
|------|------|
| LDA-1B | 在 DINO latent space 里做视觉预测；视觉 expert 和动作 expert 通过 shared self-attention 耦合 |
| FRAPPE | 不直接重建未来图像，学习 future-aware latent representations；与视觉基础模型做 latent 对齐 |

#### 通俗比喻

```
视频专家：如果这么做，杯子会滑过去
动作专家：那我需要多大力？
语言专家：目标是放进碗里，不是推到桌边
视频专家：那未来轨迹要再往右一些
动作专家：好，我调整抓取路径
```

→ 持续交互的专家协作，而不是一个人包办，也不是两个人串联。

#### 小结

- 位于"完全解耦的模块化 pipeline"和"完全统一的单 backbone 生成器"之间
- 把 world modeling 深嵌进 policy，但仍保留架构专门化
- 视频分支逐渐内化为"指导动作的 foresight stream"

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
