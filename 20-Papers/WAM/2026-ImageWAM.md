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

### 架构与表示
1. **ImageWAM 真正送给 Action Expert 的是什么？** 最终编辑图、图像 latent，还是逐层 KV cache？为什么作者不直接使用最终图？
   - → 逐层 KV cache。相比最终图，KV cache 保留了 editing backbone 各层的中间表示，信息更密集，编码了"变化过程"而非仅"变化结果"。
2. **论文中的"未来目标图"是整项任务完成后的图像，还是固定 $H$ 帧后的图像？** 这个区别为什么重要？
   - → 待确认。如果是任务完成后的全局目标图，则编辑难度大、跨度长；如果是固定 $H$ 帧后的短期目标图，则更适合局部编辑 formulation。
3. **图像编辑 flow time 和动作 flow time 是同一个变量吗？** 它们分别控制什么？
   - → 不是同一个变量。图像编辑 flow time 控制 editing diffusion 的去噪程度（图像生成质量），动作 flow time 控制 action flow matching 的插值步骤（动作精度）。

### 训练与推理
4. **ImageWAM 在训练时和推理时对图像编辑分支的使用有什么不同？** 这为什么能降低推理成本？
   - → 训练时会监督图像生成质量（$\mathcal{L}_{edit}$），推理时跳过图像解码，仅利用 editing backbone 的 KV cache 预测动作。省去 VAE decode 和 diffusion 采样循环，大幅降低延迟。
5. **为什么作者冻结 VLM，却更新图像生成分支和 Action Expert？** 如果全部一起更新，可能出现什么问题？
   - → 冻结 VLM 保留预训练语言理解能力，避免灾难性遗忘；若全部更新，小规模 robot 数据可能破坏 VLM 的通用表示，导致语言条件退化。
6. **ImageWAM 声称"不需要额外 policy pretraining"，这是否意味着它没有使用大规模预训练？**
   - → 不是。它复用了 image editing model 的大规模预训练（OmniGen2/FLUX.1），只是不需要像 π₀ 那样额外做 policy-specific 预训练阶段。

### 局限与鲁棒性
7. **如果两个动作序列最终得到几乎一样的目标图，但其中一个会碰撞，ImageWAM 仅靠编辑 cache 能否区分？** 你认为还应增加什么表示？
   - → 纯视觉编辑 cache 难以区分几何上不同但视觉外观相似的轨迹。可能需要增加 3D 几何表示、接触预测或碰撞约束模块。
8. **ImageWAM 相对视频 WAM 丢掉了哪些信息？** 这些信息在哪些任务中可能非常重要？
   - → 丢掉了多帧时序一致性和长期运动动态。在需要长 horizon 规划、物理交互预测（如推倒多米诺骨牌链）或多物体协调的任务中可能关键。
9. **如果编辑分支预测出的未来图存在严重伪影，但 KV cache 仍能产生正确动作，这说明最终图像质量和控制表示之间是什么关系？**
   - → 说明 KV cache 中的控制信息与像素级重建质量是解耦的。中间层表示捕捉的是"变化语义"而非"像素精确度"，图像质量是副产品而非动作预测的必要条件。

### 进一步研究方向
10. **如果把预测目标从 RGB 未来图换成深度图、光流、3D scene flow、对象位姿变化或接触图，哪一种最适合作为动作引导？**
    - → 待思考。各有 trade-off：深度图提供几何但丢颜色；光流直接编码运动但缺乏语义；接触图对操作最直接但标注昂贵。
11. **推理时固定编辑 timestep $\tau^*$ 是否合理？** 能否根据任务阶段或不确定性动态选择 $\tau$？
    - → 固定 $\tau^*$ 是简化假设。动态选择 $\tau$ 可能在任务初期（需要粗粒度规划）用小 $\tau$，任务末期（需要精确控制）用大 $\tau$，但增加了超参搜索复杂度。
12. **单个编辑 cache 是否会压缩掉多种可行策略？** 能否用多个噪声种子生成多个 cache，再由 critic 选择动作？
    - → 可能。单 cache 倾向于均值化。多种子 + critic 选择类似于 diffusion policy 中的多模态采样思路，但推理成本线性增加。
13. **ImageWAM 能否与 SimpleVLA-RL 结合？** 考虑到 ImageWAM 使用连续 flow-matching action，而 SimpleVLA-RL 使用离散 action token，算法上需要改哪些地方？
    - → 需要将 flow-matching action head 替换为离散 token 预测，或在 RL 端适配连续动作空间（如 PPO/SAC）。核心挑战是 editing KV cache 与 RL value function 的兼容性。
14. **如果只允许增加一个模块，你会选择接触预测、3D 几何、碰撞约束、价值模型还是语言子任务规划？为什么？**
    - → 待决定。直觉上价值模型收益最大（提供 reward signal 选择策略），但接触预测对操作任务的物理对齐最直接。
15. **怎样设计实验，才能证明性能提升确实来自"图像编辑预训练"，而不只是模型更大、数据更多或 Action Expert 更强？**
    - → 需要控制变量实验：固定 Action Expert 和模型大小，对比 (a) 有 editing 预训练 vs (b) 随机初始化 editing backbone vs (c) 用同等参数的 pure understanding model。

