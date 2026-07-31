---
title: "TurboVLA: Real-Time Vision-Language-Action Model at 32 Hz on an RTX 4090 with <1 GB VRAM"
authors:
  - Hengyi Xie
  - Chenfei Yao
  - Xianjin Wu
  - Xuanyang Xi
  - Yiping Tang
  - Di Xu
  - Yingying Zhu
  - Dingkang Liang
  - Xiang Bai
  - Han Ding
year: 2026
venue: arXiv
tags:
  - paper
  - VLA
  - robot-manipulation
  - efficient-VLA
  - action-chunking
  - vision-language-interaction
status: read
rating:
date-read: 2026-07-31
url: https://github.com/H-EmbodVis/TurboVLA
pdf: "C:/Users/huawei/Zotero/storage/29W9NIRH/Xie et al. - 2026 - TurboVLA Real-Time Vision-Language-Action Model at 32 Hz on an RTX 4090 with 1 GB VRAM.pdf"
related:
  - "[[2026-Hi-VLA-Deep-Read]]"
  - "[[Diffusion-Policy]]"
---

## TL;DR

TurboVLA 的核心不是提出更强的动作解码器，而是把 VLA 的执行路径从 **LLM-centric V -> L -> A** 改成轻量的 **V + L -> A**：视觉用 DINOv3，语言用 BERT，二者通过双向 cross-attention 交互，然后用 ACT-style decoder 一次性预测连续 action chunk。论文的主张是：执行级 manipulation 不一定需要每一步都经过大语言模型，只要指令已经是具体任务，轻量语言编码 + 直接视觉语言交互就足够强，并且能显著降低延迟和显存。

一句话抓住它：

> TurboVLA 不是让 VLA 更会思考，而是让 VLA 更快、更小、更适合实时执行。

## 论文信息

- 题目：TurboVLA: Real-Time Vision-Language-Action Model at 32 Hz on an RTX 4090 with <1 GB VRAM
- 作者：Hengyi Xie, Chenfei Yao, Xianjin Wu, Xuanyang Xi, Yiping Tang, Di Xu, Yingying Zhu, Dingkang Liang, Xiang Bai, Han Ding
- 单位：华中科技大学，华为技术有限公司
- arXiv：2607.27205v1, 2026-07-29
- 代码：<https://github.com/H-EmbodVis/TurboVLA>
- 任务类型：execution-level language-conditioned robotic manipulation
- 主要 benchmark：LIBERO, RoboTwin 2.0, AgileX Piper real-world tasks

## 问题与动机

现有 VLA 通常把 LLM / VLM 放在执行路径中心：

```text
image -> visual tokens -> LLM hidden space -> action decoder
instruction -> text tokens -> LLM hidden space -> action decoder
```

论文把这种范式概括成：

```text
V -> L -> A
```

这里的 `L` 不是简单的 language encoding，而是一个大语言模型中心的 multimodal interface。问题在于：

- 自回归 action-token VLA 会继承 language generation 的串行解码成本。
- action-expert VLA 虽然可以并行输出连续动作，但视觉和指令仍然要经过 multi-billion parameter language backbone。
- 每个控制周期都经过大模型，导致 latency、VRAM、部署成本高。
- 对很多执行级任务来说，指令已经足够具体，例如 "stack the three bowls"，低层控制并不需要每一步开放式推理。

作者的关键判断：

> 语言对 instruction-conditioned manipulation 是必要的，但执行级控制不一定要以大语言模型为中心。

因此他们提出：

```text
V + L -> A
```

即视觉和语言分别编码，再直接交互并预测动作。

## 核心方法

TurboVLA 包含四个主要部分：

```text
multi-view observation
        |
        v
DINOv3 visual encoder
        |
        v
visual tokens

instruction
        |
        v
BERT text encoder
        |
        v
text tokens

visual tokens + text tokens
        |
        v
bidirectional V-L interaction
        |
        v
action-ready multimodal features + robot state
        |
        v
ACT-style action chunk decoder
        |
        v
continuous action chunk
```

### 1. Multimodal Feature Encoding

视觉：

- 使用 DINOv3 作为 visual backbone。
- 多视角图像分别编码。
- 视觉特征投影到共享 hidden dimension `d = 256`。
- 加入 positional embedding 和 camera-view embedding。
- 多个 camera stream 直接 concat。

语言：

- 使用轻量 instruction encoder，例如 BERT。
- 保留完整 token sequence，而不是 pooled embedding。
- 原因是 object、attribute、spatial relation 这些执行相关语义需要保留给细粒度视觉 conditioning。

机器人状态：

- 单独用轻量 MLP 投影成 state tokens。
- 不参与前面的 V-L interaction。
- 在 action decoder 阶段再加入。

这个设计的意图很明确：让 cross-modal interaction 专注于 task-conditioned scene understanding，而 robot state 只在动作生成时介入。

### 2. Vision-Language Interaction Module

这是论文最核心的结构模块。

给定：

```text
visual features: Z_v
instruction features: Z_l
```

每一层做双向 cross-attention：

```text
instruction-to-visual:
language conditions visual features

visual-to-instruction:
scene context refines instruction features
```

论文强调这个模块从 Grounding DINO 的 direct cross-modal interaction 得到启发，但用途不同：

- Grounding DINO 用它做 object localization。
- TurboVLA 用它构造 control-oriented representation。

最后得到：

```text
Z_vl = concat(V_N, L_N)
```

也就是 task-conditioned visual features 和 scene-aware instruction features。

### 3. Continuous Action Chunk Prediction

动作解码器是 ACT-style transformer decoder。

输入：

```text
fused V-L features
robot state tokens
learnable action queries
```

输出：

```text
H-step continuous action chunk
```

特点：

- 不做 action tokenization。
- 不做 autoregressive action generation。
- 所有 action query 并行解码。
- 训练目标就是 behavior cloning 的 L1 loss。
- LIBERO 主实验使用 `H = 12`。

## 关键公式理解

LLM-centric VLA：

```text
Z_env = P_v(E_v(O_n))
H_L = F_L([Z_env; Tok(x)])
A_hat = D_act(H_L, s_n)
```

其中 `F_L` 是大语言模型中心接口。

TurboVLA：

```text
Z_l = P_l(f_text(x))
Z_v = P_v(f_img(I_n))
V_N, L_N = Fusion(V_0, L_0)
Z_vl = [V_N; L_N]
A_hat = D_action(Q_a, [Z_vl; Z_s])
```

关键变化：

```text
LLM as central bridge
被替换成
direct bidirectional V-L interaction
```

## 实验设置

### LIBERO

- 四个 suite：LIBERO-Spatial, LIBERO-Object, LIBERO-Goal, LIBERO-Long
- 每个 suite 10 个 language-conditioned manipulation tasks
- 使用 OpenVLA 发布的 modified no-noops RLDS datasets
- 联合训练一个 mixed-suite model
- DINOv3 ViT-B
- 连续 7-DoF actions
- action chunk horizon `H = 12`
- 训练 80k steps，10k warm-up
- effective batch size 256
- 每个任务 50 rollouts，总计 2,000 trials

### RoboTwin 2.0

- 50 个双臂 language-conditioned manipulation tasks
- 只使用 official clean demonstrations，不加 randomized-scene data
- 联合训练一个 multi-task model
- DINOv3 ViT-L
- 输出 50-step chunks
- action 是 14-dimensional absolute joint-position actions
- 训练 55k steps，1k warm-up
- effective batch size 192
- 每个任务 100 clean-setting rollouts

### Real-world

- 平台：AgileX Piper
- 任务：
  - grab roller
  - move playing card away
  - press stapler
  - stack three bowls
- 从 LIBERO pretrained TurboVLA checkpoint 初始化
- 每个任务 65 条遥操作 demos，共 `4 x 65`
- fine-tune 12.5k steps
- 每个任务 40 trials

## 主要结果

### LIBERO

TurboVLA 在 LIBERO 上达到：

```text
Spatial: 99.2
Object: 99.8
Goal: 97.4
Long: 94.2
Average: 97.7
```

部署效率：

```text
Params: 0.2B
Inference VRAM: 0.9 GB
Latency: 31.2 ms
Frequency: about 32 Hz
GPU: RTX 4090
```

和代表性方法对比：

| Method | Params | VRAM | Latency | LIBERO Avg |
|---|---:|---:|---:|---:|
| OpenVLA | 7.5B | 14.9 GB | 202.9 ms | 76.5 |
| pi0 | 3.2B | 12.3 GB | 84.2 ms | 94.2 |
| pi0.5 | 3.4B | 12.8 GB | 93.6 ms | 96.9 |
| CogVLA | 8.3B | 16.1 GB | 115.5 ms | 97.4 |
| VLA-Adapter | 1.5B | 4.3 GB | 87.3 ms | 97.3 |
| TurboVLA | 0.2B | 0.9 GB | 31.2 ms | 97.7 |

最重要的对比是 pi0.5：

```text
TurboVLA: 97.7 avg, 0.2B params, 31.2 ms
pi0.5:    96.9 avg, 3.4B params, 93.6 ms
```

也就是 TurboVLA 用约 6% 参数量，取得略高成功率和约 3x 更低延迟。

### RoboTwin 2.0

TurboVLA 在 RoboTwin 2.0 上：

```text
Params: 0.4B
Latency: 43.4 ms
Average Success: 60.2
```

对比：

| Method | Params | Latency | Avg Success |
|---|---:|---:|---:|
| ACT | 0.1B | 20.4 ms | 29.7 |
| DP3 | 0.3B | 78.4 ms | 55.2 |
| pi0.5 | 3.4B | 95.6 ms | 57.0 |
| StarVLA-alpha | 3.8B | 74.9 ms | 50.3 |
| TurboVLA | 0.4B | 43.4 ms | 60.2 |

这里说明它不是只在单臂 LIBERO 上有效，也能扩展到双臂多任务设置。

### Real-world

AgileX Piper 四个真实任务：

```text
grab roller: 92.5
move playing card away: 80.0
press stapler: 90.0
stack three bowls: 87.5
```

论文称 TurboVLA 在四个真实任务上均超过 pi0.5。

## 消融实验

### 1. Language conditioning 是否必要

| Condition | Spatial | Object | Goal | Long | Avg |
|---|---:|---:|---:|---:|---:|
| w/o Language | 87.0 | 99.4 | 11.6 | 85.0 | 70.8 |
| Task-ID Embedding | 95.6 | 98.6 | 95.8 | 91.6 | 95.4 |
| Semantic Instruction | 99.2 | 99.8 | 97.4 | 94.2 | 97.7 |

结论：

- 没语言时 Goal suite 几乎崩掉，说明同一视觉场景下有多个可能行为，语言不可省。
- task-id embedding 能恢复大部分性能，但比自然语言少 2.3 pp。
- 这说明自然语言不只是 closed-set task identity，还携带 object / attribute / spatial relation 等细粒度信息。

### 2. Text encoder 选择

| Text Encoder | Params | Spatial | Object | Goal | Long | Avg |
|---|---:|---:|---:|---:|---:|---:|
| SigLIP-Base | 216.9M | 98.6 | 99.6 | 94.8 | 89.0 | 95.5 |
| T5-Small | 141.9M | 98.8 | 99.8 | 96.8 | 92.8 | 97.1 |
| BERT | 216.1M | 99.2 | 99.8 | 97.4 | 94.2 | 97.7 |

结论：

- 不依赖某个特定 text encoder。
- 轻量 text encoder 足以支持执行级 instruction following。
- BERT 最好，但 T5-small 也很接近。

### 3. Vision-language interaction 方向

| Interaction Design | Spatial | Object | Goal | Long | Avg |
|---|---:|---:|---:|---:|---:|
| w/o Interaction | 97.4 | 99.8 | 90.8 | 92.8 | 95.2 |
| Language Queries Visual | 98.4 | 99.4 | 94.2 | 92.4 | 96.1 |
| Visual Queries Language | 98.6 | 100.0 | 94.4 | 93.0 | 96.5 |
| Bidirectional Interaction | 99.2 | 99.8 | 97.4 | 94.2 | 97.7 |

结论：

- 简单 concat 已经强，但不够。
- 单向 cross-attention 有提升。
- 双向交互最好，说明 scene-aware instruction 和 instruction-conditioned vision 是互补的。

### 4. Interaction depth

| N | Params | Spatial | Object | Goal | Long | Avg |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | 206.6M | 96.6 | 99.6 | 88.4 | 89.4 | 93.5 |
| 4 | 211.3M | 98.0 | 99.4 | 93.2 | 92.2 | 95.7 |
| 6 | 216.1M | 99.2 | 99.8 | 97.4 | 94.2 | 97.7 |
| 8 | 220.8M | 98.2 | 99.6 | 95.8 | 92.8 | 96.6 |

结论：

- `N = 6` 最优。
- 过浅不够，过深反而轻微下降。
- 这个结果支持“轻量但充分”的 cross-modal interaction，而不是越深越好。

### 5. Action horizon

论文报告：

```text
H = 8:  96.4
H = 10: 96.9
H = 12: 97.7
H = 15: 95.6
```

结论：

- 太短，时序表达不够。
- 太长，chunk prediction 难度增加。
- `H = 12` 是主实验选择。

## 论文真正的贡献

这篇论文的贡献可以分成两层。

第一层是工程贡献：

```text
0.2B params
0.9 GB inference VRAM
31.2 ms latency
97.7 LIBERO average success
```

这是非常强的部署指标。

第二层是范式贡献：

```text
execution-level VLA does not need to be LLM-centric
```

它实际上在挑战一个默认假设：

```text
大 VLM/LLM 是 VLA 的必要中心接口
```

TurboVLA 的回答是：

```text
如果任务已经是具体执行指令，那么轻量 text encoder + direct V-L interaction + action chunk decoder 就够了。
```

## 我怎么看

### 优点

1. 论文很干净，主张集中。

它没有堆一堆新模块，而是围绕一个问题：

```text
LLM-centered execution pathway 是否必要？
```

然后给出一个简单替代：

```text
V + L -> A
```

2. 实验指标抓得准。

它不只报 success rate，还同时报：

```text
params
latency
VRAM
```

这对机器人部署很重要，因为一个 98% 成功率但 15 GB 显存 / 200 ms 延迟的模型，未必比 97% 成功率但 1 GB 显存 / 30 ms 延迟的模型更有价值。

3. 消融很支持核心 claim。

尤其是：

- w/o language 大幅下降。
- task-id 不如 semantic instruction。
- bidirectional V-L interaction 优于 concat 和单向 attention。
- N=6、H=12 的选择有实证支撑。

这让方法看起来不是“随便搭了个轻量模型”，而是说明轻量语义编码和直接跨模态交互确实有作用。

### 局限

1. 它主要针对 execution-level concrete instruction。

论文自己也承认：TurboVLA 不适合复杂 high-level planning 或开放式语义推理。它能高效执行具体任务，但不是一个完整 agent。

2. 关系推理仍然是隐式的。

虽然论文说 BERT token 保留 object、attribute、spatial relation，但这些关系最终都藏在 token-level cross-attention 里，没有显式结构。

比如：

```text
put the mug on the plate
```

TurboVLA 学到的是：

```text
image patches + text tokens -> action
```

它没有显式表示：

```text
target object = mug
reference object = plate
desired relation = on(mug, plate)
current missing relation = on(mug, plate)
next transition = grasp(mug)
```

这正是后续做 graph-centric VLA 的空间。

3. 泛化故事主要是高效，不是组合推理。

LIBERO 的 success 很高，但这类 benchmark 是否足够验证 compositional relation generalization，还需要谨慎。特别是如果目标是证明“新架构更会关系推理”，光看 LIBERO 平均成功率不够，需要 wrong-object rate、wrong-reference rate、relation completion rate 这类诊断指标。

4. Grounding DINO 初始化是个细节但很关键。

论文的 V-L interaction layers 使用 grounding-pretrained feature-enhancement weights 初始化。这说明它并不是完全从零学视觉语言对应关系，而是借了视觉 grounding 的先验。这个点对复现和比较很重要。

## 对 ActionGraph-VLA / SGG 新架构的启发

这篇论文对我们前面讨论的 SGG-VLA 很有价值，但不是因为它已经用了 SGG，而是因为它暴露了一个清晰的下一步问题。

TurboVLA 的范式是：

```text
V + L -> A
```

它打掉了 LLM 中心接口，但仍然是 token-centric：

```text
visual tokens + text tokens -> action-ready features -> action
```

我们想做的新架构可以从这里继续往前推：

```text
V -> G_t
L -> G_goal
G_t + G_goal -> Delta G
Delta G -> relational transition
transition -> action
```

也就是：

```text
V + L -> A
升级为
Vision-to-Graph + Language-to-Goal-Graph + Graph-Delta-to-Action
```

TurboVLA 证明的是：

```text
execution-level control 不必以 LLM 为中心
```

ActionGraph-VLA 可以进一步主张：

```text
execution-level control 也不必以 dense visual-language token fusion 为中心，
它可以以 actionable scene graph transformation 为中心。
```

### 可以借 TurboVLA 的部分

- DINOv3 visual backbone。
- BERT / lightweight text encoder。
- ACT-style action chunk decoder。
- LIBERO / RoboTwin 训练管线。
- latency / VRAM / params 三维评估方式。

### 必须替换的部分

TurboVLA 的中心模块：

```text
bidirectional V-L interaction
```

在 ActionGraph-VLA 里应替换为：

```text
current actionable scene graph G_t
goal graph G_goal
graph delta reasoner
relational transition policy
```

这不是增量加一个 graph token，而是换掉主路径：

```text
token fusion as hidden policy state
->
graph transformation as explicit policy state
```

### 最好的论文切口

TurboVLA 已经把“快”做得很强，所以不要和它正面争低延迟。更好的切口是：

```text
TurboVLA is efficient but relation reasoning remains implicit.
We make relation reasoning explicit while preserving an efficient execution path.
```

对应实验应该不是只比平均成功率，而是比：

- target object accuracy
- reference object accuracy
- relation completion accuracy
- spatial relation success
- graph progress per action chunk
- failure recovery after wrong intermediate relation

## 可复现要点

重要参数：

```text
visual backbone: DINOv3
text backbone: BERT
hidden dimension: 256
V-L interaction depth: N = 6
action decoder: ACT-style transformer decoder
LIBERO action horizon: H = 12
LIBERO training: 80k steps, 10k warmup, effective batch 256
RoboTwin horizon: H = 50
training GPUs: 4 x RTX 4090
learning rate: 5e-5
```

需要注意：

- LIBERO 使用 modified no-noops RLDS datasets。
- RoboTwin 只用 clean demonstrations。
- latency 是从 multimodal input 到 action chunk 输出的完整 online policy latency。
- inference VRAM 是完整 online policy 的 peak GPU memory。
- 其他方法的效率指标是在 RTX 4090 batch size 1 上用官方实现和 checkpoint 测的。

## 可以追的问题

1. 如果任务需要明确的 target-reference relation，TurboVLA 的 attention 是否真的学到了正确 object grounding？

可看：

```text
cross-attention map 是否聚焦到目标物体和参照物？
```

2. 如果把 instruction 中的空间关系系统性替换，例如 left/right, in/on, near/far，性能如何变化？

这能测它的 relation compositionality。

3. 如果场景里放入 distractor object，wrong-object rate 是否升高？

这能判断 token fusion 是否足够稳。

4. 如果提供 oracle scene graph 或 predicted scene graph，能不能提升 spatial / long tasks？

这正是 ActionGraph-VLA 的切入实验。

5. 它的高性能是否部分来自 LIBERO 任务分布较闭集？

需要用更强的 compositional split 验证。

## 和我们当前 idea 的连接

当前我们想做：

```text
ActionGraph-VLA:
Vision -> Actionable Scene Graph
Language -> Goal Graph
Graph Delta -> Relational Transition
Transition + Local Vision -> Action Chunk
```

TurboVLA 是非常好的 baseline，因为它代表了：

```text
最强的轻量 token-centric VLA
```

我们的目标不是说 TurboVLA 不行，而是说：

```text
在多物体、多关系、目标/参照物消歧和空间组合泛化中，
显式 graph transformation 可能比隐式 V-L token fusion 更稳。
```

因此实验设计应该把 TurboVLA 当成强 baseline，并尽量复用它的低层 decoder 和训练设置，让差异集中在：

```text
policy state representation:
token fusion vs graph transformation
```

## 总结评价

这篇论文很适合作为我们 SGG/ActionGraph-VLA 方向的起点。它已经证明了：

```text
去掉 LLM 中心接口后，VLA 仍然可以在执行级 manipulation 上又快又强。
```

但它也留下了一个明确缺口：

```text
视觉-语言-动作之间的对象关系和目标关系仍然是隐式的。
```

如果我们要做新的架构，最好不要再做：

```text
TurboVLA + SGG token
```

而应该做：

```text
TurboVLA-style efficient execution
+ graph-centric task/state representation
+ relation transition policy
```

最好的论文句子可以是：

> TurboVLA shows that execution-level VLA need not be LLM-centric; we further show that it need not be dense-token-centric either. By recasting manipulation as goal-conditioned scene-graph transformation, ActionGraph-VLA makes object-relation reasoning explicit while retaining fast continuous action execution.

