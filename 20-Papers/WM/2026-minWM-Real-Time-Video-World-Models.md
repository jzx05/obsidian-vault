---
title: "minWM: A Full-Stack Open-Source Framework for Real-Time Interactive Video World Models"
authors:
  - Min Zhao
  - Hongzhou Zhu
  - Kaiwen Zheng
  - Zihan Zhou
  - Yimin Chen
  - et al.
year: 2026
venue: arXiv:2605.3026v1 (May 2026)
tags:
  - paper
  - world-model
  - video-generation
  - real-time
  - autoregressive
  - diffusion-distillation
  - camera-control
  - open-source
status: read
rating: 4
date-read: 2026-06-05
url: https://arxiv.org/abs/2605.3026
project: https://github.com/shengshu-ai/minWM
pdf: "file:///C:/Users/Administrator/Zotero/storage/AJF8YS54/Zhao%20%E7%AD%89%20-%202026%20-%20minWM%20A%20Full-Stack%20Open-Source%20Framework%20for%20Real-Time%20Interactive%20Video%20World%20Models.pdf"
zotero-key: AJF8YS54
---

# minWM: A Full-Stack Open-Source Framework for Real-Time Interactive Video World Models

> 📎 [打开原文 PDF](file:///C:/Users/Administrator/Zotero/storage/AJF8YS54/Zhao%20%E7%AD%89%20-%202026%20-%20minWM%20A%20Full-Stack%20Open-Source%20Framework%20for%20Real-Time%20Interactive%20Video%20World%20Models.pdf)
> 🔗 项目主页：https://github.com/shengshu-ai/minWM
> 🏛️ 单位：清华生数（ShengShu）/ 清华 / 人大 / HKUST / UT-Austin

## TL;DR
> 一个**全栈、开源、可复现**的视频世界模型框架。把现成的 T2V/TI2V 视频扩散大模型（Wan2.1、HY1.5）改造成**摄像机可控、低延迟、实时交互**的自回归世界模型。首帧延迟从 ~270s / 770s 压到 **1.1s / 3.4s**（最高 236× 加速）。

## 问题与动机
- 现有视频生成模型多为**离线、双向、文本驱动**，无法满足世界模型对实时交互的需求
- 三个核心 gap：
  - 控制：text prompt → camera + user actions
  - 推理：双向多步去噪 → 因果 AR + few-step
  - 数据：缺乏可控、可复现的训练数据 pipeline
- 已有零散方案（Genie、Hunyuan-GameCraft、Yume、Vidar 等），但**完整链路无开源版本**
- 定位：类似"Stable Diffusion 之于扩散模型"，做世界模型的社区基建

## 核心方法（两阶段流水线）

### Stage 1：双向扩散 + 摄像机控制
- 用 **PRoPE**（Projective Rotary Position Embedding）把摄像机内外参 (K, T) 注入 self-attention
- 关键洞察：注意力交互只依赖**相对**摄像机位姿，保留原模型生成能力的同时获得相机可控性
- 适配两种主流 backbone：
  - **Wan2.1-T2V-1.3B**（cross-attention 架构）
  - **HY1.5-T2V-8B**（MMDiT 架构）

### Stage 2：AR 扩散蒸馏（三步走）
1. **Stage 2a — Causal Forcing 训练**：把双向模型变成因果 AR 模型，仍保留多步去噪
2. **Stage 2b — 初始化（二选一）**：
   - **Causal ODE 初始化**（更高质量，但需离线生成 ODE 数据）
   - **Causal CD（Consistency Distillation）初始化**（无需额外存储，更省资源）
3. **Stage 2c — Asymmetric DMD 后训练**：
   - 用**双向 teacher** 校正 **AR student** 的分布漂移
   - 边际分布对齐：min KL(p_data ‖ p_AR) + DMD 梯度修正

## 实验与结论

### 训练设置
- 模型：Wan2.1-T2V-1.3B、HY1.5-T2V-8B
- 数据：DL3DV（重建+渲染获得 ground-truth 相机轨迹）+ WorldPlay 生成 + OpenVid
- 双向训练：32 batch × 4k–8k steps；蒸馏：bs 4–16，4k–6k steps（Stage 2a）+ 2k（Stage 2b）+ 200（Stage 2c）

### 关键结果（Table 1：A800 单卡首帧延迟）

| Base Model | Multi-step Bidir | Multi-step AR | **Few-step AR (minWM)** | **Speedup** |
|-----------|-----------------|--------------|------------------------|-------------|
| HY1.5     | 771 s           | 81 s         | **3.4 s**              | **223×**    |
| Wan2.1    | 269 s           | 28.6 s       | **1.1 s**              | **236×**    |

- 摄像机可控性基本保留（Figure 2 多场景案例）
- AR 模型可"无限延展"，能在 sliding window 中保留前帧条件持续生成

### 消融研究（Section 3.3）

| 因素 | 结论 |
|------|------|
| **训练数据** | 直接用 SpatialVid（pose 是估计的）效果差 → 需要 ground-truth 相机轨迹（DL3DV 重建 / WorldPlay 生成） |
| **训练步数（HY1.5）** | 1–2k 步：完全不可控；5k 步：开始响应；8k 步：稳定可控 |
| **批量大小（Wan2.1）** | bs<4 失败；bs=8 不稳定；**bs=16 才能稳定训练** |

## 我的思考

### ✅ 优点
- **工程价值高**：第一份完整的"数据→训练→蒸馏→推理"开源 recipe
- **架构无关性**：同时验证了 cross-attention 和 MMDiT 两类架构
- **延迟数字硬核**：1.1s 首帧已经接近真正的实时交互门槛
- **蒸馏路径灵活**：CD 初始化对低预算用户友好

### ⚠️ 局限
- 当前只支持**摄像机控制**，未实现 pose / 键盘 / 手柄等离散 action（Future Work）
- SpatialVid 这类大规模真实数据**暂时用不起来**，依赖合成轨迹，scaling 受限
- 蒸馏后**质量损失**未给出量化指标（FVD、CLIP score 都没报）
- bs=16 的训练成本对小团队仍然偏高

### 🚀 可延伸方向
- 加入离散 action token（参考 Genie 3、Hunyuan-GameCraft）
- 关联到 [[50-Areas/WAM]] 中的 World-Action Model 研究
- 探索**长时一致性**（当前 sliding window 可能漂移）
- 把 minWM 接到 RL / 具身智能 pipeline 当 simulator

## 引用与延伸阅读

### 同主题论文（按论文 references）
- [9] **Genie 3** — DeepMind 新一代世界模型（2025）
- [10] **Hunyuan-GameCraft** — 腾讯交互式游戏世界模型
- [11] **Yume-1.5** — 文本可控交互式视频生成
- [17] **Pan: A world model for general, interactable, and long-horizon world simulation**
- [22] **Self Forcing** — 桥接训练-测试 gap 的 AR 视频扩散
- [23] **Causal Forcing** — minWM Stage 2a 的核心技术
- [24] **Causal Forcing++** — 同作者前作，可扩展 few-step AR 蒸馏
- [26] **PRoPE** — 摄像机注入 attention 的位置编码（NeurIPS 2024）
- [30] **DMD（Distribution Matching Distillation）** — 蒸馏核心算法

### 待新建概念笔记
- [[30-Notes/concepts/World-Model]]
- [[30-Notes/concepts/Causal-Forcing]]
- [[30-Notes/concepts/PRoPE]]
- [[30-Notes/concepts/Asymmetric-DMD]]
- [[30-Notes/methods/Few-Step-AR-Distillation]]

### 同主题索引
- [[20-Papers/WM/_index]]
