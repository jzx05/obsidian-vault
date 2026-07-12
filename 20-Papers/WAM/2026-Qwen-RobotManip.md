---
title: "Qwen-RobotManip: Alignment Unlocks Scale for Robotic Manipulation Foundation Models"
authors: [Qwen Team, Haoyuan Yuan, Zhuoran Liang, Analu Chen, Yu Wang, Haoyang Li, Fan Liu]
year: 2026
venue: arXiv
tags: [paper, VLA, robotic-manipulation, embodied-AI, foundation-model, alignment, cross-embodiment]
status: reading
rating: 
date-read: 
url: https://arxiv.org/abs/2506.13846
pdf: "C:\\Users\\Administrator\\Zotero\\storage\\Z7YI9TVG\\Yuan 等 - 2026 - Qwen-RobotManip Technical Report Alignment Unlocks Scale for Robotic Manipulation Foundation Models.pdf"
priority: high
---

# Qwen-RobotManip: Alignment Unlocks Scale for Robotic Manipulation Foundation Models

## TL;DR
> 通过设计统一的对齐框架（动作对齐 + 视觉对齐），将异构的人类数据、合成数据与机器人数据对齐到统一表示空间，解锁了大规模数据对机器人操作基础模型的 scaling 效应。基于 Qwen2.5-VL 构建的 VLA 模型在标准基准和 OOD 泛化任务上达到 SOTA，并在多个真实平台（ALOHA、Franka、UR、ARX）验证了部署能力。

## 核心问题与动机
- 语言和多模态领域的 Foundation Model 通过大规模数据实现了 scaling，机器人操作能否复制这一路径？
- 核心瓶颈：机器人数据来源异构（不同 embodiment、不同形态），难以直接 scale
- 已有方法的不足：
  - 单纯增加数据量不够——embodiment 差异（手形态、动作空间）和视觉域差异（背景、视角）导致负迁移
  - 现有 VLA 缺乏有效的 OOD 泛化评估标准
- **本文核心洞察**：Alignment（对齐）是解锁 Scale 的关键。通过 morphology alignment + visual alignment，将异构数据统一到共享表示，才能让 scaling recipe 生效

## 架构设计

### 整体框架
- **视觉-语言骨干**：Qwen2.5-VL (7B)，作为 vision-language backbone
- **动作专家**：Diffusion Transformer (DiT)，预测 unified end-effector motion
- **两大统一表示**：
  1. Unified State and Action Representation（跨 embodiment 的状态-动作统一）
  2. Unified End-Effector Motion Prediction（末端执行器运动预测）

### 跨 Embodiment 状态与动作表征
- 所有 embodiment 统一到 **末端执行器空间**：
  - 关节位置 (7-dim)
  - End-effector pose: Cartesian position (3) + 6D continuous rotation (6)
  - Gripper state: 抓爪位置
  - Dexterous hand (12-dim): 灵巧手指位置
- 用 64 维 per-action block（7 个 per-action 维度 + padding），前 22 维存实际动作

### 统一末端执行器运动预测
- **相机坐标系下的 delta action**：将 action 表示为相对于参考帧的 SE(3) delta
  - $\Delta R = R_t \cdot R_{\text{ref}}^{-1}$（旋转矩阵差）
  - $\Delta p = R_{\text{ref}}^{-1} \cdot (p_t - p_{\text{ref}})$（平移差，在参考帧坐标系下）
- **Camera-aware positional encoding**：将 3D positional info 注入 action tokens，处理不同相机内外参
- **End-effector-aware conditioning**：DiT 根据 embodiment type、优先表示、数据来源等条件化

### In-Context Policy Adaptation
- 支持 few-shot 上下文学习：将少量演示作为 context chunks 注入
- 每个 chunk 包含两条互补路径：
  1. Proprioceptive states + action chunks → action decoder MLP → 离散 action tokens
  2. Video frames → VLM visual encoder → 视觉表征
- Stochastic context sampling：随机采样上下文窗口，训练时模型学会从部分上下文中泛化

## 数据来源

### 规模
- 总计 ~58 million trajectories（机器人 + 人类 + 合成）
- ~98,000 hours 的操作视频数据

### 机器人数据集
| 数据集 | 规模 | 特点 |
|--------|------|------|
| Open X-Embodiment (OXE) | 聚合多源 | 多 embodiment |
| AgiBot-World (fBeta) | ~60 万条 | 大规模双臂 |
| RoboMIND 2.0 | 大规模真实世界 | 多场景 |
| Galaxea Open-World | ~500 hours | 工业+家庭 |
| DROID | ~76,000 episodes | 多相机+自然语言 |
| RH20T | 跨 14 个任务大类 | 多步骤 |

### 人类数据（Egocentric）
- Ego4D, EPIC-KITCHENS, VISOR, SEEDs, RoboPoint, Hand pose datasets
- 通过 Human-to-Robot Synthesis Pipeline 对齐到机器人动作空间

### Human-to-Robot 数据合成管线
1. **Action Alignment（动作对齐）**：
   - 手形态重映射（retargeting）：将人手关键点映射到 gripper 的 end-effector position
   - 利用 Saliency-Guided Similarity Matching + Gaussian-weighted SLERP 平滑插值
2. **Visual Alignment（视觉对齐）**：
   - 替换背景：inpainting + optical flow 引导，生成干净背景序列
   - Embodiment 注入：将机器人手臂渲染到人类视频中（通过优化 base frame 位置最小化 embodiment distance）

### VL Co-training 数据
4 类共 ~250 万样本：
1. General Visual Understanding（视觉问答、图像推理）
2. Spatial Perception & Reasoning（2D/3D 空间关系）
3. OCR & Document Understanding
4. Instruction Following / MultiRound / Pure Text

### Embodied Chain-of-Thought (ECoT) 数据
- 用强 VLM 生成结构化推理过程：
  - Scene description → Object detection → Task progress assessment → Atomic action
- 在 LIBERO 和 RoboTwin 上合成 ECoT 数据

## 训练流程

### Pre-training（Dual-Stream Co-Training）
- **VLA stream**：Flow matching loss 预测动作 + VLM next-token prediction
- **训练目标**：$\mathcal{L} = \lambda_{\text{video}} \cdot \mathcal{L}_{\text{flow}} + \mathcal{L}_{\text{VLM}}$
- Flow matching target：velocity field $v = \varepsilon - x_0$（Rectified Flow）
- 独立 scheduler：不同 shift 适配 action 的去噪难度曲线
- LLR regularization：防止 VLM 灾难性遗忘

### Post-training（Domain-specific SFT）
- 针对特定 embodiment/task fine-tune
- 保持 SFT 仅优化 flow matching loss（不含 VLM loss），视觉编码器冻结
- 可选 ECoT 模式（结构化推理 + 动作生成）

### Co-Training in Post-Training
- 混合通用 VL 数据防止过拟合
- 实验发现：混入 VL 数据在 OOD 泛化上一致提升
- 最优配比：VL 数据占 post-training 数据的 ~30-50%

## 数据筛选与对齐管线

### 5 阶段数据清洗
1. **Sudden Change Detection**：过滤突变帧（传感器故障、遮挡）
2. **State-Action Temporal Alignment**：纠正时间错位（相机帧和控制帧的延迟）
3. **Extreme Value Filtering**：过滤异常动作值（超出物理合理范围）
4. **Joint End-Effector FK Consistency**：关节角度和 FK 解算末端位置的一致性
5. **Base Frame & EEF Orientation Alignment**：统一不同数据集的坐标系约定

### 3 项质量检查
1. **Instruction Consistency**：文本描述和动作的一致性
2. **Video-State Consistency**：视觉帧和状态变量的同步性
3. **Video Quality Filtering**：去除黑帧、模糊帧、遮挡帧

## 实验结果

### 标准基准（In-Distribution）
| 基准 | Qwen-RobotManip | 对比 SOTA |
|------|:---:|:---:|
| LIBERO (avg) | **99.1** | π₀ (Black): 95.4 |
| RoboTwin-Easy | **86.7** | GR00T N1.5: 85.7 |
| RoboTwinClean | **95.2** | — |

- 在所有标准基准上 match 或 exceed SOTA
- 关键发现：从头训练（scratch）也能达到 comparable 性能 → 说明 alignment + scale 本身就足够

### OOD 泛化（核心贡献）
| 基准 | Qwen-RobotManip | 对比 |
|------|:---:|:---:|
| LIBERO-Plus (7 维扰动) | **75.4** (Master) | π₀: 23.4 |
| RoboTwin-CleanRand | **86.5** (Easy) | LIBERO-Plus 上 3x |
| RoboCasa365 (atomic) | **46.0** | GR00T: 26.6 |
| EBench (Overall) | **46.7** | — |

- OOD 评估维度：Camera、Robot、Language、Light、Background、Noise、Layout
- **关键发现**：QWEN-ROBOTMANIP 的 OOD 泛化与 in-context history 前缀（few-shot context）结合效果最佳

### 真实机器人部署
| 平台 | In-Domain | OOD | 特点 |
|------|:---------:|:---:|------|
| ColoRMagic ALOHA | 88.6% | 84% | 双臂，5-6 个任务 |
| ARX ALOHA | 达到 ~88% few-shot | — | 少样本适应 |
| Franka | 验证通过 | — | 单臂 |
| UR | 验证通过 | — | 工业 |
| ADX | 验证通过 | — | — |

### Few-shot Adaptation（ARX ALOHA）
- 仅 10 次演示即可达到 ~30-50% 成功率（从 0 开始）
- Full frotate（3 子步）：52.2%
- 展示了 in-context policy adaptation 的有效性

### Cross-Embodiment Transfer
- 在 RoboTwin-XL 上 zero-shot 迁移到未见 embodiment
- ARX-X7, UR5-WSG, Franka Panda 三个 unseen embodiment 均有效

### RoboChallenge TableSet-V1（Generalist Track）
- 强双手协调能力：在 ALOHA 平台上完成需要双臂同步的 6 个挑战任务
- 关键行为：bimanual 协调（左手稳定、右手操作），error recovery（自纠错、重新抓取）
- 超越 purely 数据中学到的策略——展现了组合式操作技能

## 关键消融实验

### Alignment 的贡献
- **Camera-frame delta action** vs global-frame：camera-frame 一致性提升显著（尤其多相机场景）
- **Human2Robot 合成数据**：加入 H2R 对齐数据后，LIBERO-Plus 从 ~57% → 90%+
- **去掉对齐只加数据**：性能反而下降 → 证明 alignment 是 scale 的前提

### VL Co-training 的贡献
- 加入 VL 数据对 OOD 泛化一致正向：LIBERO-Plus language 扰动从 86% → 95%
- In-distribution 性能基本持平或略升
- 最优 VL 数据占比：~30-50%

### ECoT 的贡献
- 在 RoboTwin-II 上 +32.2% 成功率提升
- 在强 perturbation 场景（instruction following）尤其有效
- 代价：推理延迟增加（需要先生成 reasoning tokens）

### 数据 Scaling
- Action 预测 MSE 随数据量增加持续下降（log-linear）
- Camera-frame + alignment 的 scaling curve 比 global-frame 更陡——alignment 放大了 scale 的收益
- EEF（end-effector frame）比 joint-space 的 scaling 效率更高

### 架构消融
- 3 层 self-attention 变体最优（vs 1 层或 full-layer cross attention）
- In-context tokens 的注入方式：last-layer self-attention 最佳
- Camera-frame EEF 的优势在数据 scaling 后更加显著（Table 20）

## 核心洞察与创新

1. **Alignment > Scale alone**：单纯增加数据不够，必须先对齐（morphology + visual + coordinate frame），才能让 scale 发挥作用
2. **Camera-frame delta action 是关键设计选择**：让不同相机配置的数据可以共享动作表示
3. **Human data 通过对齐管线变得有用**：raw human video 对机器人几乎无用，但经过 action alignment + visual alignment 后变成高价值训练数据
4. **VL co-training 防止视觉退化**：在 post-training 中混入 VL 数据维持 VLM 的泛化能力
5. **In-context policy adaptation 实现 few-shot deployment**：无需 fine-tune，通过少量演示上下文即可适配新 embodiment/task

## 与 Fast-WAM 的关系

| 维度 | Qwen-RobotManip | [[20-Papers/WAM/2026-Fast-WAM|Fast-WAM]] |
|------|:---:|:---:|
| 范式 | VLA（直接输出动作） | WAM（视频 co-training，推理时不生成视频） |
| 视频角色 | VL co-training 作为辅助（不生成视频） | Video loss 作为训练信号 |
| 动作预测 | DiT action expert + flow matching | DiT action expert + flow matching |
| 视觉骨干 | Qwen2.5-VL (7B) | Wan2.5-1B |
| 核心创新 | 对齐框架 + 数据 scaling | 证明 test-time imagination 不必要 |
| 数据规模 | ~58M trajectories | 较小规模实验 |
| 部署验证 | 多平台真实部署 | 单平台 |

共同点：
- 都用 DiT 做 action expert + flow matching
- 都认为 video/visual 信息对 action 预测有间接帮助（通过训练信号）
- 都不需要推理时生成视频

## 我的思考

### 优点
- 系统性工程：从数据清洗、对齐、训练到部署的完整 pipeline
- OOD 评估框架有价值——填补了 VLA 领域的评估空白
- 真实平台验证充分（5+ 个平台）
- 数据 scaling curve 提供了有意义的 insight（alignment 是 scale 的前提）

### 局限
- 仍然依赖 end-effector space → 对高度灵巧操作（如灵巧手精细操作）可能不够
- Human2Robot 对齐管线复杂度高，泛化到新 embodiment 需要重新设计
- ECoT 的推理开销在实时控制场景中可能不可接受
- 论文未详细讨论失败模式和 failure analysis

### 可延伸方向
- 与 [[20-Papers/WAM/2026-Fast-WAM|Fast-WAM]] 的 MoT 架构结合：用 video expert 提升视觉表示质量
- 对齐框架能否自动化（learnable alignment 而非手工管线）
- 将 in-context adaptation 扩展到更复杂的 multi-step reasoning 场景
- OOD 泛化基准可以标准化为社区共享的 benchmark

## 关联
- 同方向：[[20-Papers/WAM/2026-Fast-WAM]]（WAM 视角的 action prediction）
- 概念：[[30-Notes/concepts/Diffusion-Policy]]、[[30-Notes/concepts/Flow-Matching]]
- 基础：[[30-Notes/concepts/Diffusion-Transformer]]

## 引用与延伸阅读
- π₀ (Black et al., 2024): 开创性的 VLA foundation model
- GR00T N1.5 (NVIDIA): 大规模 robot foundation model
- Open X-Embodiment (2024): 跨 embodiment 数据集联盟
- Octo (Ghosh et al., 2024): 早期 generalist robot policy
- AgiBot-World (2025): 大规模双臂操作数据集
