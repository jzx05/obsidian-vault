---
title: "Model-Based RL World Model (MBRL-WM)"
tags: [concept, world-model, reinforcement-learning, dreamer, MBRL]
created: 2026-06-07
related:
  - "[[30-Notes/concepts/World-Model]]"
  - "[[30-Notes/concepts/Generative-World-Model]]"
---

# Model-Based RL World Model（基于模型的强化学习世界模型）

## 定义
> 在**压缩潜在空间**中学习环境的状态转移动力学，使 RL 智能体无需与真实环境交互——直接在 WM 的"想象"中优化策略。

核心公式：
$$z_{t+1}, \hat{r}_t = f_\theta(z_t, a_t)$$
其中 $z_t$ 是潜在状态，$a_t$ 是动作，$\hat{r}_t$ 是预测奖励。

---

## 核心组件（以 Dreamer 为原型）

```
观测 o_t
    │
    ▼ Encoder
潜在状态 z_t ──────────────┐
    │                      │
    │ + 动作 a_t            │ 解码（可选，仅训练时）
    ▼                      ▼
RSSM（循环状态空间模型）    重建 ô_t
    ├── 确定性隐状态 h_t
    └── 随机潜在状态 z_t
    │
    ├── 奖励预测头 r̂_t
    └── 终止预测头 d̂_t
```

**RSSM（Recurrent State Space Model）** 是 Dreamer 系列的核心架构：
- **确定性路径**：GRU/Transformer 维护历史记忆
- **随机路径**：显式建模环境随机性，用于不确定性估计

---

## 训练与使用流程

```
阶段1 — 世界模型训练（真实数据）
  真实轨迹 (o, a, r) → 训练 encoder + RSSM + 解码头

阶段2 — 策略在想象中优化（无真实交互）
  z_0 → WM rollout → z_1, z_2, ..., z_H
        ↓
  在潜在轨迹上计算 return → 反向传播更新 π

阶段3 — 策略在真实环境执行
  收集新数据 → 加入 replay buffer → 回到阶段1
```

---

## 演进路线

| 版本 | 年份 | 核心改进 |
|------|------|---------|
| **World Models** (Ha & Schmidhuber) | 2018 | 首次将 WM + Controller 解耦；MDN-RNN 潜动力学 |
| **DreamerV1** (Hafner et al.) | 2019 | RSSM + 端到端反向传播穿过 WM；Atari/DMControl |
| **DreamerV2** (Hafner et al.) | 2020 | 离散化潜在状态（categorical latent）；Atari 超越 MFRL |
| **DreamerV3** (Hafner et al.) | 2023 | 跨域无调参（Minecraft、Atari、Proprio）；symlog 变换；KL 均衡 |
| **TD-MPC2** | 2023 | 隐式 WM + 时序差分规划；无需解码器 |

---

## 与生成式 WM 的核心区别

| 特性 | MBRL-WM | [[30-Notes/concepts/Generative-World-Model\|生成式 WM]] |
|------|---------|----------------|
| 输出 | 抽象 latent vector | 可视化视频帧 |
| 可视保真度 | 低（可选重建） | 高（光真实感） |
| 计算代价 | 低，单 GPU 可训 | 高，需大规模预训练 |
| 训练数据 | 任务专属小量轨迹 | 互联网级视频 |
| 主要用途 | in-imagination RL | 仿真器/数据增强 |
| 动作耦合 | 显式紧密 | 可松散（prompt注入） |

---

## 局限性
- **模型偏差（Model Bias）**：WM 与真实环境的误差在长 horizon 累积，导致策略在真实环境失效
- **分布外泛化弱**：RSSM 在训练数据覆盖外的状态预测不可靠
- **视觉保真度低**：潜在空间难以捕捉精细外观细节
- **数据效率仍受限**：需要足够的真实交互数据才能学到可信的动力学

---

## 相关概念
- [[30-Notes/concepts/World-Model]] — 上位概念
- [[30-Notes/concepts/Generative-World-Model]] — 对比路线
- 在综述中对应位置：§4.1（WM for RL）
