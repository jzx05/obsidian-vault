---
title: "DiT4DiT: Jointly Modeling Video Dynamics and Actions for Generalizable Robot Control"
authors: [Teli Ma, Jie Zhong, Zifan Wang, Chundi Jiang, Andy Cui, Junwei Liang, Shua Yang]
year: 2026
venue: arXiv
tags: [paper, WAM, world-action-model, DiT, video-generation, policy-learning, flow-matching]
status: reading
rating: 
date-read: 
url: https://arxiv.org/abs/2603.10448
pdf: "C:\\Users\\Administrator\\Zotero\\storage\\5EU8P4TP\\Ma 等 - 2026 - DiT4DiT Jointly Modeling Video Dynamics and Actions for Generalizable Robot Control.pdf"
priority: high
---

# DiT4DiT: Jointly Modeling Video Dynamics and Actions for Generalizable Robot Control

## TL;DR
> 用两个 Diffusion Transformer（Video DiT + Action DiT）联合建模视频动态和动作预测。Action DiT 从 Video DiT 的中间层抽取 hidden states 作为条件，实现"视频生成能力→动作预测能力"的知识迁移。关键创新：asymmetric timestep 采样 + dual flow matching + video generation as scaling proxy。

## 核心问题
- VLA 模型语义强但缺乏物理动态理解（action-conditioned dynamics 差）
- 现有 WAM 要么推理时太慢（需要先生成视频），要么不能有效利用视频生成的 representation
- **本文追问**：如何让 video generation 的 backbone 表示直接增强 action prediction，同时保留视频生成能力作为 scaling 指标？

## 与 Fast-WAM 的对比

|                    | Fast-WAM                       | DiT4DiT                                   |
| ------------------ | ------------------------------ | ----------------------------------------- |
| 架构                 | MoT（共享 self-attention）         | 独立双 DiT + feature extraction              |
| Video→Action 信息流   | Action attend to video 第一帧 KV  | Action DiT 读取 Video DiT 中间层 hidden states |
| 推理时 video 分支       | 完全移除                           | 保留（可选 joint inference）                    |
| 核心 insight         | 训练时 co-training 够了，推理不需要 video | 视频生成是 action quality 的 scaling proxy      |
| Attention coupling | 共享 self-attention（MoT）         | 单向 feature conditioning                   |
| 预训练基础              | Wan2.5-5B                      | Cosmos-Tokenizer + OpenSora2.2            |

## 方法：DiT4DiT

### 核心假设
**Video generation as scaling proxy**：视频生成质量和动作预测性能正相关。验证方式：
1. 用 VLM + auxiliary direction 配置（视频+动作联合训练 vs 纯 VLA）
2. 用 hybrid world model（FLAME 范式，先生成视频再出动作）
→ 两种方式都证明 video generation 能力提升 action success rate

### 架构设计

```
观测 o_t + 语言指令 l
     │
     ▼
[VAE Encoder] → first frame latent z_0
     │
     ▼
┌─────────────────────────────┐
│   Video DiT (Cosmos-2.2B)   │ ← 生成未来帧 latent
│                             │
│   Layer k: hidden state h_k │──────→ [Feature Extraction]
│                             │              │
└─────────────────────────────┘              │
     │                                       ▼
     ▼                              ┌────────────────────┐
  Video Latents                     │   Action DiT       │
     │                              │   条件: h_k + text  │
     ▼                              └────────────────────┘
[VAE Decoder]                                │
     │                                       ▼
  Future Frames                         Action a_1...H
```

### 关键设计细节

**1. Asymmetric Timestep Sampling**
- Video DiT 使用时间步 $\tau_v$，平均分布采样
- Action DiT 使用独立时间步 $\tau_a$，偏
- **不对称**：不同于强制同步的 joint diffusion，允许两个模态以不同节奏去噪
- 代码中用 `Beta(α, α)` 分布采样 $\tau_v$

**2. Dual Flow Matching**
两个独立的 flow matching 目标：
$$\mathcal{L}_{FM} = \lambda \cdot \mathcal{L}_{video} + \mathcal{L}_{action}$$

- Video: 预测 velocity field $v_\theta(x_t, t)$，target = $\epsilon - x_0$
- Action: 同样的 flow matching，但条件是 Video DiT 的 hidden states

**3. Video DiT → Action DiT 的信息传递**

```python
# 不是简单的最后一层输出
# 而是从 Video DiT 某一层（或多层）抽取 hidden states
h_k = video_dit.layers[k].output  # 中间层特征
# h_k 包含了视频生成过程中学到的时空动态信息
# Action DiT 通过 cross-attention 读取 h_k
```

关键：Video DiT 的 hidden states 在**生成过程中**编码了物理动态知识（运动趋势、物体交互），这些信息比最终输出的 pixel-level latent 更抽象、更 action-relevant。

**4. Decoupled Training**
- Video backbone 可以独立用大规模视频数据预训练
- Action DiT 在 robot data 上微调时，video backbone frozen 或 low-lr
- 允许 video backbone 的 scaling（更大模型、更多数据）直接 benefit action

### 推理流程

**Action-only inference（部署）：**
1. 输入当前帧 → Video DiT forward（不做完整去噪，只做一次 forward 获取 hidden states）
2. Hidden states → Action DiT 做 N 步去噪 → 输出 action chunk

**Joint inference（可选，用于调试/可视化）：**
1. Video DiT 完整去噪 → 生成未来帧
2. 同时 Action DiT 用 Video DiT 的 hidden states → 预测动作

### 训练细节
- Video DiT base: Cosmos-Pretrained-2.2B（也支持 OpenSora2.2）
- Action DiT: 较小的 DiT（~128M params）
- Tokenizer: Cosmos video tokenizer（4×spatial, 4×temporal compression）
- 训练数据: FLAME-style pretraining on large video + robot-specific finetuning
- Action representation: delta action + proprioception

## 实验结果

### LIBERO（模拟）
- DiT4DiT 平均成功率 **90.4%**（4 个子任务）
- 优于 OpenVLA, Octo 等纯 VLA baseline
- 在 LIBERO-Long（长 horizon 任务）上优势最明显

### RoboSea-GR1（模拟）
- 7-DoF + 2 DoF gripper，29 维 action space
- 比 baseline 高出 5-10% 成功率

### 真实机器人
- Unitree G1 humanoid robot
- 7 类桌面操作任务
- 证明视频生成质量与 action 性能正相关

### 关键消融
1. **Video loss 权重 λ**：λ=0（纯 action）性能显著下降 → video generation 确实帮助 action
2. **Feature extraction 层选择**：中间层（~层 12/30）比最后一层或第一层效果好
3. **Video generation quality vs action success**：正相关（R² ≈ 0.7+）

## 与 Fast-WAM 的深度对比

### 信息流方式不同
- **Fast-WAM**：MoT 共享 self-attention，action tokens 直接在 attention 中读 video 第一帧的 KV
- **DiT4DiT**：Action DiT 通过 cross-attention 读取 Video DiT 的中间层 hidden states

→ DiT4DiT 的信息更丰富（中间层 hidden states 包含多层抽象），但计算量更大

### 推理效率
- **Fast-WAM**：推理时完全去掉 video 分支，~10ms
- **DiT4DiT**：推理时仍需 Video DiT forward（至少一次），~50-100ms（估计）

→ Fast-WAM 更适合高频实时控制，DiT4DiT 更适合需要丰富视觉条件的复杂任务

### Scaling 故事不同
- **Fast-WAM**："video co-training 已经足够，推理时不需要"
- **DiT4DiT**："video generation 是 action 的 scaling proxy，越好的视频生成→越好的动作"

→ 两者其实不矛盾：Fast-WAM 强调推理效率，DiT4DiT 强调训练时 video 的价值

## 为什么读
- 与 Fast-WAM 形成互补对照：同样是 Video DiT + Action 联合建模，但设计哲学不同
- "Video as scaling proxy" 假设对综述 §6 有重要参考价值
- Dual flow matching 的独立 scheduler 设计与 Fast-WAM 类似，但解耦方式不同
- 为综述提供"显式 feature extraction" vs "shared attention" 两种 video→action 信息流的对比案例

## 关联
- 对比论文：[[20-Papers/WAM/2026-Fast-WAM]] — 同时期同主题，设计哲学不同
- 同主题：[[20-Papers/WM/2026-minWM-Real-Time-Video-World-Models]]
- 概念：[[30-Notes/concepts/Diffusion-Transformer]] — 核心架构
- 概念：[[30-Notes/concepts/World-Model]]
- 综述位置：§6 WM for Policy Learning（video backbone → action head）
