---
title: "ImageWAM: Do World Action Models Really Need Video Generation, or Just Image Editing?"
authors: [Yuyang Zhang, Wenyan Zhang, Zekun Qi, He Zhang, Hailuo Lin, Jingbo Zhang, Yue Ma, Xiaolong Yang, Wenjia Zeng, Xin Jin]
year: 2026
venue: arXiv
tags: [paper, WAM, world-action-model, embodied-AI, image-editing, policy-learning, robot-manipulation]
status: completed
rating: 
date-read: 2026-07-12
url: https://arxiv.org/abs/2606.19551
pdf: "C:\\Users\\Administrator\\Zotero\\storage\\8Z6SF3WG\\Zhang 等 - 2026 - ImageWAM Do World Action Models Really Need Video Generation, or Just Image Editing.pdf"
priority: high
---

# ImageWAM: Do World Action Models Really Need Video Generation, or Just Image Editing?

## TL;DR
> 现有 WAM 依赖视频生成模型预测未来视觉轨迹，计算开销巨大且对长时间推理不稳定。ImageWAM 提出用预训练的**图像编辑模型**替代视频生成作为 WAM 的 backbone，将未来预测重新定义为逐帧的局部视觉变化（image editing），而非完整时空视频生成。实验表明，图像编辑模型能够提供紧凑有效的世界建模表示，在多个基准上超越视频生成式 WAM。

## 核心问题
- 视频生成 WAM 需要预测完整的未来视频帧序列，计算昂贵且容易产生时间不一致的 artifacts
- 视频生成模型的许多能力（长期时序建模、多帧一致性）对于机器人短 horizon 控制可能是冗余的
- **本文追问**：WAM 真的需要视频生成能力吗？还是只需要图像编辑（理解局部视觉变化）就够了？

## 动机：从视频生成到图像编辑

图像编辑模型相比视频生成有三大优势：
1. **强指令跟随能力** — 能精确理解"什么应该变、什么应该不变"
2. **聚焦局部变化** — 不需要重建整个场景，只关注 action-relevant 的视觉变化
3. **预训练充分** — 大规模图像编辑预训练提供了丰富的视觉变换先验（颜色、位移、遮挡等）

关键 insight：机器人操作中每步的视觉变化本质上就是一次"编辑"——物体被移动、手臂位置改变、gripper 状态变化。这与 image editing 的 formulation 天然对齐。

## 方法：ImageWAM

### 问题建模

将 WAM 的预测分解为：
- 在每个时间步 $t$，给定当前观察 $o_t$ 和任务指令 $l$，预测动作 chunk $a_{t:t+H}$
- 策略学习目标：$\pi_\theta = \arg\max_\theta \{a_t\}$
- **视频生成式 WAM**：先预测完整未来帧序列 $\{o_1, ..., o_T\}$，再从中提取动作
- **ImageWAM**：将未来预测转化为逐帧图像编辑，聚焦帧间局部变化

### 架构设计

```
Text Instruction → LLM → Text Hidden/State
                              ↓
Current Obs (o_t) → VAE Enc → Image Editing Backbone ← Noise, Time, Action/Noise
                              ↓                              ↓
                     Edited Image (o_{t+1})           Action Expert → a_t, a_{t+1}, ...
                              ↓
                         VAE Dec → Next Frame
```

核心组件：
1. **Image Editing Backbone** — 基于预训练编辑模型（OmniGen2/FLUX.1 Fill），接收当前帧 + 文本指令，输出编辑后的下一帧
2. **Action Expert** — 从 editing backbone 的中间 transformer 层提取 KV cache 作为 action context，通过 flow matching 预测动作
3. **LLM** — 处理文本指令，提供 language conditioning

### Image Editing Backbone

- 将当前观察帧作为 source image，下一帧作为 editing target
- Diffusion-based editing model 通过 text instruction 引导编辑方向
- **关键设计**：action expert 从 editing backbone 的 transformer 中间层收集 KV，这些 KV 编码了"当前帧向下一帧转变"的 editing-aware 表示

### Action Prediction 与 Training

**Image editing objective:**
- Diffusion image branch 预测 velocity field，将当前帧"编辑"为下一帧
- 优化目标：使编辑后的 latent 逼近真实下一帧的 latent

**Action flow matching:**
- 使用 flow matching 目标预测动作
- 采样 action noise $\epsilon$，加噪：$a_t = (1-\sigma) \cdot a_0 + \sigma \cdot \epsilon$
- Action expert 预测 velocity field：$\hat{v}_{edit} = f(o_t, l, \sigma)$

**联合训练损失：**
$$\mathcal{L} = \mathcal{L}_{edit} + \lambda \cdot \mathcal{L}_{action}$$

### Efficient Inference

两种推理模式：
1. **Joint inference** — 同时生成下一帧 + 预测动作（用于可视化/调试）
2. **Action-only inference (Fast-WAM style)** — 跳过图像解码，仅用 editing backbone 的 KV 直接预测动作
   - 只保留 editing backbone 的 understanding component（前向 + KV cache）
   - 跳过 diffusion generation branch 和 VAE decode
   - 推理延迟大幅降低

此外实现了 video-WAM 兼容接口：将 editing backbone 串联为逐帧生成，模拟 video WAM 的 future-token interface。

## 实验结果

### RoboTwin 2.0
| Method | P.T. | Conv | Total | Avg. |
|--------|:---:|:---:|:---:|:---:|
| π₀ [1] | ✓ | 74.24 | 92.00 | 91.3 |
| Gr-2 [7] | ✓ | 70.76 | 79.75 | — |
| ARefer [12] | ✗ | — | 87.09 | — |
| FanGS+VLA [5] | ✗ | 91.03 | 91.88 | — |
| **ImageWAM** | ✗ | **83.20** | **83.50** | **85.38** |

### LIBERO
| Method | P.T. | Spatial | Object | Goal | Long | Avg. |
|--------|:---:|:---:|:---:|:---:|:---:|:---:|
| HPT [4] | ✓ | 64.4 | 78.6 | 77.2 | 48.4 | 67.2 |
| GR-MG [3] | ✓ | 96.0 | 94.6 | 90.4 | 79.0 | 90.0 |
| Diffusion Policy | ✗ | 78.8 | 92.2 | 68.4 | 50.4 | 72.5 |
| **ImageWAM** | ✗ | **97.0** | **95.0** | **88.8** | **86.6** | **91.9** |

### LIBERO-Plus（新增复杂指令基准）
- 涵盖 8 个维度：P.I., Camera, Robot, Language, Light, Background, Size, Layout
- ImageWAM 在大多数维度上超越 VLA baselines（OpenVLA, EasyR1）和 video-based WAMs（FanWAM, YanWAM）
- ImageWAM(FLUX.1 fill) 平均达到 **60.5** 的成功率

### Real-world 实验
- 4 个真实桌面操作任务：Stack Three Boxes, Fold Towel, Open Drawer & Store Marker, Hang Cup On Rack
- ImageWAM 超越 FanWAM 等视频生成 baseline
- 最大增益在需要精确空间控制的任务（如 Hang Cup On Rack）

### 效率对比
| Method | Lat. | TFLOPs | Inference |
|--------|:---:|:---:|:---:|
| Fast-WAM(Vid) | 18 ms | 14.62 | Video |
| EasyWAM(1-step) | 180 ms | 11.91 | Video |
| **ImageWAM(Oura)** | **223 ms** | **8.72** | Image |

### 关键发现
1. **图像编辑模型是有效的 WAM backbone** — 不需要完整视频生成能力
2. **编辑式表示聚焦帧间变化** — 避免了视频 WAM 的时间不一致 artifacts
3. **ImageWAM 在 VLA 和 video backbone 上均有提升** — 增益不仅来自更强的图像识别或语言对齐
4. **Editing backbone 的 unified understanding-generation 能力很关键** — 纯理解或纯生成模型效果不如兼具两者的编辑模型
5. **更大的 editing backbone 进一步提升性能** — 但收益随任务复杂度变化

## 消融实验

### Q4: 不同 editing model 的影响
| Editing Backbone | LIBERO | RoboTwin |
|---|:---:|:---:|
| OmniGen2 w/ CLIP | ✗ | 72.5 |
| FLUX.1 fill | ✓ | 83.5 |

FLUX.1 fill 的 fill-in-the-middle 能力比 OmniGen2 更适合"给定当前帧，编辑为下一帧"的 formulation。

### Q5: Unified understanding + generation 的重要性
- 仅 understanding（CLIP encoder）：性能下降
- 仅 generation（pure diffusion）：缺乏对指令的精确理解
- Editing model 天然兼具两者：理解"什么该变" + 生成"变成什么样"

### Q6: Editing backbone size 的影响
- FLUX.1 (12B) 在 LIBERO-Plus 上明显优于较小模型
- T2 任务（需要理解复杂指令）增益最大 → 大模型的语言理解更强
- 但对于简单任务（LIBERO 标准），较小模型已足够

## Qualitative Analysis

### Future-video artifacts 对比
- Video-based WAM 容易产生：幻觉新物体、忽视 task-relevant 物体、混合不同任务的视觉元素
- ImageWAM：避免这些 artifacts，因为 editing 天然聚焦于局部变化
- 但 ImageWAM 也有局限：对需要长时间规划的任务，逐帧编辑可能缺乏全局视角

## 与 Fast-WAM 的对比

| 维度 | Fast-WAM | ImageWAM |
|------|----------|----------|
| Backbone | Video DiT (Wan2.5) | Image Editing (FLUX.1/OmniGen2) |
| 核心思路 | 保留 video co-training，跳过推理时视频生成 | 直接用 image editing 替代 video generation |
| 未来建模 | 训练时有 video loss，推理时不生成 | 训练时有 editing loss，推理时可选 |
| 表示优势 | 时空视频预训练的丰富 KV | 编辑式变化感知的紧凑表示 |
| 推理效率 | ~10ms (action-only) | ~223ms (editing) / 可 action-only 加速 |
| 适用场景 | 实时闭环控制 | 需要精确空间理解的操作 |

两者的共同 insight：**WAM 的性能增益主要来自训练时的视觉建模目标（video/editing loss），而非推理时的完整视觉生成。**

## 为什么读
- 从全新角度审视 WAM 的设计空间：video generation 不是唯一选择
- 提出了 image editing 作为世界建模的轻量替代
- 实验覆盖全面（模拟 + 真实 + 多基准 + 消融）
- 对理解"WAM 到底需要什么视觉能力"有启发价值

## 关联
- 同主题：[[20-Papers/WAM/2026-Fast-WAM]]
- 同主题：[[20-Papers/WAM/2026-DiT4DiT]]
- 概念：[[30-Notes/concepts/World-Model]]
- 相关：Image Editing Models (OmniGen2, FLUX.1 Fill)

## 追问
- ImageWAM 的 action-only 推理延迟如何？论文只给了 joint inference 的 223ms，跳过 decode 后能否接近 Fast-WAM 的 10ms 级别？
- 对于需要长 horizon 规划的任务（如 LIBERO-Long），逐帧编辑是否会累积误差？
- Editing backbone 的预训练数据分布（网页图像编辑）与机器人场景差距大，fine-tuning 数据量需要多少？
- 能否结合 Fast-WAM 的 MoT 架构与 ImageWAM 的 editing backbone，取两者之长？
