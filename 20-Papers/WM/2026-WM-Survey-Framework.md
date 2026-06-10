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

#### Section 3 范式递进总览

| 子节 | 范式 | WM 与 Policy 关系 | 代表 |
|------|------|------------------|------|
| 3.2 | Decoupled IDM | WM 先预测未来 → IDM 反推动作（串联） | UniPi, VidMan |
| 3.3 | Single Backbone | 同一 backbone 联合生成未来视觉 + 动作 | UVA, VideoVLA |
| 3.4 | MoE/MoT Expert | 多 expert 保留分工，层间持续交互 | Motus, DiT4DiT |
| 3.5 | Latent-Space WM | 不做像素预测，在表示空间构造 future target | VLA-JEPA, FLARE |
| 3.6 | Unified VLA | VLA 自身内化 predictive structure | DreamVLA, WorldVLA |

> 递进主线：从"外挂 WM 模块"逐步走向"WM 能力内生于 policy backbone"。

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

> **开放问题**：视频预训练 backbone 是否一定比同等规模的 VLM backbone 更适合机器人控制，目前仍是开放问题。现有结果更像是"有前景的归纳偏置证据"，而非最终定论。

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

---

### 3.4 MoE/MoT 混合专家策略

> 保留多个功能专门化的 expert stream（video / action / language），通过 attention 在层间持续耦合。介于完全解耦（3.2）和完全统一（3.3）之间。

#### 核心思想

- 高层形式：h~v~^l+1^, h~a~^l+1^ = F~mix~^l^(h~v~^l^, h~a~^l^; o~t~, l)
- 交互方式：joint attention / cross-attention / shared-attention fusion
- 视频分支从"必须输出的视频"转变为 **predictive latent process**（hidden states 指导动作分支）

#### 动机：为什么不完全共享参数？

视频预测和动作生成在三个维度上存在本质差异：

| 差异维度 | 视频预测 | 动作生成 |
|---------|---------|---------|
| 时间频率 | 较稠密的连续时序变化 | 高频、精确、低延迟信号 |
| 表示尺度 | 高维视觉 token / latent | 低维控制命令 |
| 优化目标 | 时空一致性 | 可执行性和控制精度 |

> full parameter sharing is not always optimal.

#### 代表方法

**第一类：并行 expert coupling**

| 方法 | 特点 |
|------|------|
| GE-Act | 预训练 video diffusion backbone + 轻量 action branch；deep cross-attention 注入视频 latent |

**第二类：Mixture-of-Transformers 式深交互**

| 方法 | 特点 |
|------|------|
| Motus | understanding + video generation + action expert，标准 MoT |
| LingBot-VA | video/action token 交织进共享 autoregressive 序列；dual-stream MoT + shared attention |
| BagelVLA | 语言规划 + 视觉预测 + 动作生成放入同一执行循环 |
| DiT4DiT | 用视频分支中间去噪特征指导动作预测 |
| Fast-WAM | 收益更多来自训练时视频协同训练，而非推理时显式想象 |

**第三类：Latent-space expertization**

| 方法 | 特点 |
|------|------|
| LDA-1B | DINO latent space 视觉预测；视觉/动作 expert 通过 shared self-attention 耦合 |
| FRAPPE | 学习 future-aware latent representations；与视觉基础模型做 latent 对齐 |

---

### 3.5 潜在空间世界模型策略 (Latent-Space WM)

> Future prediction entirely in representation space — 不做像素级视频解码，在 latent space 构造 predictive targets 与动作生成耦合。Backbone 通常是 MLLM-based 而非 video-DiT-based。

#### 动机

| 显式视频预测的问题 | Latent-space 的优势 |
|------------------|-------------------|
| 计算开销大（要解码像素） | 避免显式生成解码的开销 |
| 信息冗余（背景纹理等控制无关细节） | 只提取对动作有用的未来信息 |
| 像素级重建不等于对控制有帮助 | 直接学"对控制有用的未来结构" |

#### 代表方法

| 方法 | 核心做法 | 关键词 |
|------|---------|--------|
| FLARE | action denoising network 的 hidden feature 对齐 future observation 的 latent embedding | Future Latent Repr. Alignment |
| VLA-JEPA | Leakage-free state prediction：未来帧只产生 latent supervision target，逼模型学习动作相关的状态转移 | JEPA-style, leakage-free |
| JEPA-VLA | 直接用 V-JEPA 2 的 predictive embeddings 做 VLA backbone，而非额外加 future head | Predictive pretraining as backbone |
| WoG | 直接预测对精确控制最有用的 compact future-oriented conditions | Compact control conditions |
| DIAL | 在 VLM feature space 中用 latent visual foresight 做结构化 bottleneck，解耦高层意图与低层动作 | Intent-action decoupling |

#### 补充：Symbolic / Planner-facing World Models

- 把 world modeling 外化为 predicate / object relation / affordance / operator / causal process
- 供 symbolic planner 或 task-and-motion planner 查询，生成高层技能序列
- 说明：有用的 WM 不依赖像素，也不必是 neural latent — 只要能表达对控制有用的未来结构
- 与 JEPA family 概念关联：都在 embedding space 做预测而非像素空间
- 核心立场：**比起把整个未来画出来，不如直接学对动作有用的未来表示**

---

### 3.6 统一 VLA 模型 (Unified VLA Models)

> 不依赖外部视频 WM，在 VLA backbone 内部内化 future-oriented predictive modeling。区分标准：不是"有没有独立 WM 模块"，而是预测结构是否被内化在同一个 multimodal policy backbone 里。

#### 与 reactive VLA 的区别

| 类型 | 纯 reactive VLA | Unified VLA |
|------|----------------|-------------|
| 输出 | 直接出动作 | 动作 + 某种未来预测目标 |
| 内在结构 | 当前→动作的直映射 | 动作生成与内部预测目标联合训练 |

#### 三类实现方式

**第一类：显式 future-image prediction**

预测未来图像作为辅助训练信号（推理时不一定渲染），迫使模型学习动态结构。

| 方法 | 特点 |
|------|------|
| GR-1 | 单个 GPT-style transformer 联合预测动作和未来图像 |
| UP-VLA | future-image prediction 同时提升动作生成和视觉泛化 |
| WorldVLA | autoregressive 统一动作+图像理解+图像生成；future-image 主要是训练信号 |

**第二类：隐式 / latent future modeling**

不预测像素，预测 compact future-aware representations。

| 方法 | 特点 |
|------|------|
| DreamVLA | 预测 structured world knowledge（dynamic / spatial / semantic cues） |
| UniVLA | 后训练阶段吸收大规模视频中的 causal dynamics，不外挂独立 WM |
| CoWVLA | 压缩到 latent motion + compact future visual targets |

**第三类：Multi-expert unified VLA**

主框架仍是统一 VLA，但保留功能专门化的 predictive branch（visual foresight / subgoal generation）。

| 方法 | 特点 |
|------|------|
| F1 | MoT 架构中预测 future visual states 作为 planning targets |
| InternVLA-A1 | lightweight latent visual foresight + action generation 联合优化 |
| HALO | visual subgoal prediction + embodied reasoning |
| TriVLA | grounding + episodic dynamics perception + control 协调子系统 |

#### 小结

- 共同原则：动作生成不再是纯 reactive mapping，而是与内部预测目标联合训练
- 预测目标可以是：未来状态本身 / latent surrogate / semantic surrogate

---

## 🎮 Section 4 — World Model as Simulator（WM 作为仿真器）

> **核心问题**：WM 直接充当交互式仿真环境，供 RL 训练和 policy 评估使用。

### 4.1 WM 用于强化学习 (World Model for Reinforcement Learning)

#### 核心范式

与 Section 3 的区别：Section 3 中 WM 提供预测/监督/先验来增强 policy；Section 4 中 WM 直接替代真实环境，policy 在 WM 内部做 rollout、收集奖励、迭代优化。

```
┌───────────────────────────────────────────────────┐
│  World Model for RL                               │
│                                                   │
│  [World Model] ←── action ──── [Policy Model]     │
│       │                              ↑            │
│       ├─→ imagined observation ──────┘            │
│       ├─→ reward signal                           │
│       └─→ termination signal                      │
│                                                   │
│  Policy 在 WM 内部做 rollout 并优化                │
└───────────────────────────────────────────────────┘
```

WM 作为 learned simulator 提供：imagined transitions + reward signals + termination signals。Policy 通过最大化 imagined rollout 的期望回报来优化。

#### 两个关键设计问题

1. **WM 必须与 policy 共同演化** — learned simulator 需要和 policy 一起迭代改进（dual function）
2. **环境忠实度（fidelity）** — 模型偏差会累积，影响 sim-to-real transfer

#### 代表方法

**Latent-space RL：**

| 方法              | 核心特点                                              |
| --------------- | ------------------------------------------------- |
| DreamerV1/V2/V3 | 在 latent space 做完整 MBRL；V3 通用化到多种任务               |
| TD-MPC2         | temporal difference + model-predictive control 混合 |
| GRPO / PanguRL  | 把 RL 更新适配到 flow-based action heads                |
| WorldRL-Flex    | 灵活的 RL-WM 耦合框架                                    |
| GreenRobotLM    | world-model-based RL 接入预训练 VLA                    |
|                 |                                                   |

**Video/pixel-space RL：**

| 方法 | 核心特点 |
|------|---------|
| DWA (Chandak et al., 2026) | 大规模数据训练 online WM，达到 RL-level policy |
| WorldRL (Ding et al., 2025) | 高保真操作任务 WM-RL |
| WorldVLA-R | 视频 WM 内做 RL fine-tuning |
| VLAW (Guo et al., 2026) | 交替 WM 数据收集与 RL 优化 |
| WMPO (Rao et al., 2026) | joint state-space imagination + reward prediction |

#### 核心挑战

| 挑战 | 描述 |
|------|------|
| 模型偏差累积 | 长 horizon rollout 时误差复合，policy 在真实环境退化 |
| Reward 建模 | WM 需准确估计 reward 和 termination，不只是状态转移 |
| WM-Policy 共演化 | 两者需交替改进，单独优化任一方都不够 |
| Sim-to-real gap | 即使 WM 高保真，与物理世界仍有差距 |

---

### 4.2 WM 用于评估 (World Model for Evaluation)

#### 核心思想

WM 不仅用于训练 policy，还可作为 **safety evaluator / policy validator**：在部署前或运行时，用 WM 模拟候选动作的后果，进行筛选和验证。

```
┌───────────────────────────────────────────────────┐
│  World Model for Evaluation                       │
│                                                   │
│  Scene → [World Model] → imagined outcomes        │
│           multiple candidate actions              │
│                   ↓                               │
│  [Policy Model] ← select best / validate safety   │
└───────────────────────────────────────────────────┘
```

#### 与 4.1 的区别

| 维度 | 4.1 WM for RL | 4.2 WM for Evaluation |
|------|--------------|----------------------|
| 目标 | 优化 policy 参数 | 评估/排序已有 policy |
| 交互方式 | 完整训练循环 | 一次性 rollout + 打分 |
| 关键输出 | 梯度更新 | 安全判断 / 质量评分 |

#### 评估模式

1. **Policy ranking / selection** — 在 WM 中让多个候选 policy rollout，选择表现最优者
2. **Safety validation** — 模拟极端场景，检测 policy 是否产生危险动作
3. **Runtime monitoring** — 部署时用 WM 预判下一步后果，拦截潜在错误
4. **Offline policy evaluation** — 无需真实交互，用 WM 估计 policy 的期望表现

#### 代表方法

| 方法 | 核心特点 |
|------|---------|
| WorldFomo | WM-based offline policy evaluation；用世界模型估计 policy 效用 |
| VLA-RST (Storr et al., 2025) | video reward 用于策略验证和反馈 |
| WorldGym | 探索 WM 能否完全替代真实环境进行评估 |
| BISE (Wang et al., 2025) | 通过 appearance yields 增强评估真实性 |
| HEPA | 基于 WM 的 policy competition / benchmark |
| GigaSim | WM-based simulation 用于大规模策略评估 |

#### 关键洞察

- 评估者不仅是 scorer，更是 **judge** — 需判断动作是否安全、合理、可执行
- 核心价值：**无需真实世界交互即可大规模测试 policy**
- 当前瓶颈：WM 的 fidelity 直接决定评估可信度；低保真 WM 可能给出误导性评分

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

- [x] 导航与操作的 WM 需求有何不同？
- [x] 代表方法？（如 Pathdreamer, VISTA, NWM）
- [x] WM 在导航中的核心价值？

---

### 6.2 WM for Autonomous Driving（自动驾驶）

**需要找的信息：**

- [x] 自动驾驶对 WM 的特殊需求？（多智能体、长时预测）
- [x] 代表方法？（如 GAIA-1, DriveDreamer, Drive-WM）
- [x] 与机器人操作的 WM 有何异同？

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
