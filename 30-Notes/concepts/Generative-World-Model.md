---
title: "Generative World Model（大规模生成式世界模型）"
tags: [concept, world-model, generative-model, video-diffusion, foundation-model]
created: 2026-06-07
related:
  - "[[30-Notes/concepts/World-Model]]"
  - "[[30-Notes/concepts/MBRL-World-Model]]"
---

# Generative World Model（大规模生成式世界模型）

## 定义
> 以**大规模视频/图像数据**预训练，能在**像素或高维视觉空间**中生成逼真、动作可控的未来帧序列。本质是"可条件化的视频生成器"，兼具环境仿真能力。

核心范式：
$$v_{t+1:t+H} = G_\theta(v_{1:t},\ a_{1:t},\ c)$$
其中 $v$ 为视频帧，$a$ 为动作/控制信号，$c$ 为语言/任务条件。

---

## 与 MBRL-WM 的根本差异

MBRL-WM 问的是：**"环境下一步状态是什么？"**（为 RL 服务）  
生成式 WM 问的是：**"下一帧画面看起来是什么样？"**（为感知、仿真、数据增强服务）

| 维度 | [[30-Notes/concepts/MBRL-World-Model\|MBRL-WM]] | 生成式 WM |
|------|---------|----------|
| 表示空间 | 压缩 latent vector | 像素 / latent video |
| 视觉保真度 | 低（可选） | 高（光真实感） |
| 训练数据规模 | 小量任务轨迹 | 互联网级视频 |
| 动作条件化 | 显式紧密耦合 | 灵活注入（attention/adapter） |
| 主要应用 | in-imagination RL | 仿真器、数据增强、策略先验 |
| 计算规模 | 单 GPU 可训 | 数百~数千 GPU |
| 代表方法 | DreamerV3, TD-MPC2 | Genie 3, GameNGen, DIAMOND, UniSim |

---

## 核心技术路线

### 路线 A — Video Diffusion Based
```
噪声帧 → UNet/DiT 反复去噪 → 清晰视频帧
         ↑ 条件注入
    (动作 a, 语言 c, 历史帧 v_{1:t})
```
- 代表：**Genie 3**、**GameNGen**、**IRASim**、**Cosmos**
- 优势：生成质量最高，支持复杂场景
- 劣势：推理慢（多步去噪），实时性差

### 路线 B — Autoregressive Token Based
```
帧 → VQ-VAE → 离散 token → Transformer next-token prediction
```
- 代表：**Genie 1/2**、**DIAMOND（IRIS）**
- 优势：可与 LLM 统一架构，支持长序列推理
- 劣势：量化损失精度，token 序列长

### 路线 C — Causal Video Transformer（Flow/Consistency）
```
历史帧 + 动作 → Causal Attention → 下一帧（单步/少步）
```
- 代表：**minWM**（清华生数 2026）、**Causal-Forcing**
- 优势：兼顾质量与速度，支持实时交互
- 劣势：长期一致性仍是挑战

---

## 关键能力维度

```
生成式 WM
    │
    ├── 可控性 ──── 动作条件化（joystick/robot action）
    │              语言条件化（"open the door"）
    │              相机条件化（PRoPE 等）
    │
    ├── 保真度 ──── 视觉真实感（photorealism）
    │              物理一致性（rigid body, contact）
    │              时序一致性（identity, texture）
    │
    └── 效率性 ──── 首帧延迟（< 5s 目标）
                   吞吐量（FPS）
                   内存占用
```

---

## 在机器人学习中的三种用法

```
用法1：作为仿真器（Simulator）
  WM 生成机器人操作视频 → policy 在 WM 内做 RL/IL
  代表：UniSim, RoboDreamer

用法2：作为数据增强（Data Augmentation）
  真实轨迹 → WM 生成变体/扩展 → 扩充训练集
  代表：WorldFomo, WoW

用法3：作为策略的感知骨干（Perceptual Backbone）
  预训练 WM 特征 → 迁移到下游 policy
  代表：DreamVLA, UniVLA (单骨干架构)
```

---

## 代表方法速查

| 方法 | 年份 | 技术路线 | 核心贡献 |
|------|------|---------|---------|
| **GameNGen** (Google) | 2024 | Video Diffusion | 首个神经网络实时运行 DOOM |
| **DIAMOND / IRIS** | 2024 | AR Token | Diffusion WM 在 Atari 超越 DreamerV3 |
| **Genie 1** (DeepMind) | 2024 | AR Token | 无动作标注，从视频自监督学控制 |
| **Genie 3** (DeepMind) | 2025 | Video Diffusion | 3D 感知 + 物理一致；基础 WM |
| **IRASim** | 2024 | Video Diffusion | 机器人动作可控视频生成 |
| **UniSim** | 2023 | Video Diffusion | 统一动作空间的机器人仿真器 |
| **minWM** (清华生数) | 2026 | Causal Flow | 实时交互；见 [[20-Papers/WM/2026-minWM-Real-Time-Video-World-Models]] |
| **Hunyuan-GameCraft** (腾讯) | 2025 | Video Diffusion | 高保真游戏视频 WM |

---

## 核心挑战

- **动作因果性（Causal Grounding）**：生成视频≠正确响应动作，视觉真实≠物理正确
- **长时一致性**：超过 ~100 帧后 identity / 场景布局漂移
- **实时性**：Video Diffusion 的多步去噪难以达到交互帧率
- **数据需求**：需要带动作标注的机器人视频，采集成本高
- **评测困境**：视觉指标（FID/FVD）≠任务有用性，见综述 §7

---

## 相关概念
- [[30-Notes/concepts/World-Model]] — 上位概念
- [[30-Notes/concepts/MBRL-World-Model]] — 对比路线（紧凑潜在空间 RL）
- [[30-Notes/concepts/Causal-Forcing]] — 生成式 WM 的训练技术
- [[30-Notes/concepts/Asymmetric-DMD]] — 生成式 WM 的加速技术
- 在综述中对应位置：§5（WM for Video Generation）、§4.1 后半段
