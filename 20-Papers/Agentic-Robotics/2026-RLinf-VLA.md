---
title: "RLinf-VLA: A Unified and Efficient Framework for Reinforcement Learning of Vision-Language-Action Models"
authors:
  - Hongzhi Zang
  - Mingjie Wei
  - Si Xu
  - Yongji Wu
  - Zhen Guo
  - Yuanqing Wang
  - Hao Lin
  - Peihong Wang
  - Liangzhi Shi
  - Yuqing Xie
  - Zhexuan Xu
  - Zhihao Liu
  - Kang Chen
  - Wenhao Tang
  - Quanlu Zhang
  - Weinan Zhang
  - Chao Yu
  - Yu Wang
year: 2026
venue: arXiv preprint
arxiv: 2510.06710v2
tags:
  - paper
  - VLA
  - reinforcement-learning
  - robot-learning
  - systems
  - simulator
  - PPO
  - GRPO
  - agentic-robotics
status: read
rating: 8/10
date-read: 2026-08-01
url: https://arxiv.org/abs/2510.06710
code: https://github.com/RLinf/RLinf
project: https://huggingface.co/RLinf
pdf: "C:/Users/huawei/Zotero/storage/A3EH6QIP/Zang et al. - 2026 - RLinf-VLA A Unified and Efficient Framework for Reinforcement Learning of Vision-Language-Action Mo.pdf"
related:
  - "[[2026-Harness-VLA-Deep-Read]]"
  - "[[2026-Tau0-VLA]]"
  - "[[2026-TTHE-Test-Time-Harness-Evolution]]"
  - "[[2026-SkillRise]]"
  - "[[2026-Hi-VLA-Deep-Read]]"
---

# RLinf-VLA: VLA 强化学习的统一高效训练框架

## TL;DR

RLinf-VLA 不是一篇“新 VLA 架构”论文，而是一篇很典型的 **VLA-RL system paper**。它要解决的是：现在用强化学习 post-train VLA 的代码栈太碎，不同模拟器、不同模型、不同 RL 算法各写一套；同时 VLA-RL 比 LLM-RL 更麻烦，因为 rollout 里不只有 generation，还有 simulator rendering / physics execution，它们会和 inference / training 抢 GPU。

一句话总结：

> RLinf-VLA 把 VLA-RL 训练拆成 Generation、Simulator、Training 三个组件，用统一接口接入 ManiSkill、LIBERO、RoboTwin、OpenVLA、OpenVLA-OFT、PPO、GRPO，并通过灵活 GPU 分配和 fine-grained pipelining 把吞吐显著拉上去。

最重要的结果：

- 支持多个模拟器：ManiSkill、LIBERO、RoboTwin；
- 支持多个 VLA：OpenVLA、OpenVLA-OFT；
- 支持 PPO 和 GRPO；
- GPU-parallelized simulator 上 hybrid pipeline 比 naive disaggregated 最多快 1.88x；
- 相比 SimpleVLA-RL，在 LIBERO / RoboTwin collocated 设置下最多快 2.27x；
- 单个 OpenVLA-OFT + RLinf-GRPO 模型在 130 个 LIBERO 任务上达到 98.11%；
- RoboTwin 六个任务平均从 24.48% 到 84.63%；
- ManiSkill OOD 上 OpenVLA 从 39.10% 到 81.93%，OpenVLA-OFT 从 18.29% 到 77.05%。

我的判断：这篇最有用的不是“RL 让 VLA 分数变高”这个结论，而是它把 VLA-RL 的工程痛点拆清楚了：action chunk 的 credit 粒度、early success 后的有效动作 mask、partial reset、rollout batch size、LoRA learning rate、simulator/GPU placement，这些东西会直接决定实验能不能跑稳、跑快、可复现。

## 1. 问题背景

VLA 模型把视觉、语言和动作接在一起，通常先靠大规模 robot demonstrations 做 imitation / supervised fine-tuning。问题是部署时会遇到：

- 新物体；
- 新布局；
- 新指令组合；
- 新执行条件；
- 训练数据没有覆盖到的状态；
- 局部失败后的恢复需求。

RL post-training 的吸引力是：它直接优化任务成功率，并允许模型从交互中探索 corrective behaviors，而不是只模仿专家轨迹。

但 VLA-RL 的系统复杂度比 LLM-RL 高很多。LLM-RL 主要是：

```text
generate tokens -> score -> optimize
```

VLA-RL 则是：

```text
observe image + instruction
-> generate action or action chunk
-> simulator executes physics / rendering
-> get next observation
-> repeat until trajectory ends
-> train policy
```

这里 simulator、inference、training 都可能占 GPU。没有专门调度，就会出现 GPU idle time、pipeline bubbles、offload/onload 开销、模拟器接口不统一、算法实现重复等问题。

## 2. 论文贡献

作者把贡献分成四类：

1. **统一系统抽象**：统一接入 ManiSkill、LIBERO、RoboTwin，统一 OpenVLA / OpenVLA-OFT，统一 PPO / GRPO。
2. **高效系统设计**：提供 collocated、disaggregated、hybrid 三种 GPU allocation modes，并提出 hybrid fine-grained pipelining。
3. **高性能结果**：RLinf-VLA 在 LIBERO、ManiSkill、RoboTwin 上带来约 20%-85% 的性能提升。
4. **开源平台**：作为 VLA-RL 可复现实验基座发布。

它的定位很清楚：不是说 PPO/GRPO 是新算法，而是“让 VLA-RL 能被统一、稳定、高效地跑起来”。

## 3. VLA-RL pipeline

RLinf-VLA 把训练过程抽象成三个组件：

| 组件 | 作用 | 资源特点 |
|---|---|---|
| Generation | 用当前 VLA 根据 observation 生成动作或 action chunk | GPU inference |
| Simulator | 执行动作，更新物理状态，渲染下一帧 | CPU 或 GPU，取决于模拟器 |
| Training | 用 rollout 数据更新 policy | GPU training |

VLA 和 LLM 最大区别之一是 action hierarchy：

```text
Chunk -> Atomic Action -> Token
```

- 一个 chunk 包含多个 atomic actions；
- 一个 atomic action 包含多个 token；
- 每个 token 常对应一个 action dimension，例如末端位姿或关节角。

OpenVLA 更接近 single-step action，OpenVLA-OFT 使用 multi-step action chunk。这个区别会直接影响 advantage、log-prob、value、mask 和 simulator scheduling。

## 4. GPU allocation modes

### 4.1 Collocated

所有组件共享同一组 GPU。

早期做法是组件轮流上 GPU，结束就 offload 到 CPU，需要时再 onload。对 embodied rollout 很糟，因为 Generation 和 Simulator 频繁交替，每一步都 offload/onload 会炸开销。

RLinf-VLA 的改法是：Simulator 和 Generation 只在 rollout phase 开始/结束时 offload/onload，避免每步搬运。但它仍会有利用率问题，因为二者交替等待。

适合：

- CPU-parallelized simulator；
- simulator 不怎么占 GPU；
- 训练/rollout 可以阶段化执行。

### 4.2 Disaggregated

Simulator、Generation、Training 被放到不重叠 GPU partition。

优点：

- 简单；
- 各组件资源隔离；
- 不需要频繁 offload/onload。

缺点：

- 组件之间有依赖，某些 GPU 会在别的组件运行时空转；
- 例如 rollout 时 training GPUs 闲置。

它是一个好 baseline，但不是最高效方案。

### 4.3 Hybrid with fine-grained pipelining

Hybrid 的典型配置：

```text
Simulator: subset of GPUs
Generation: another subset of GPUs
Training: all GPUs
```

rollout 时 Simulator 和 Generation 分开跑；training 时 Training 可以用所有 GPU。进一步，fine-grained pipelining 会把一个 GPU 上的 simulator instance 切成多个 sub-simulators：

```text
S1 generates observation o0
-> Generation produces action a0 for S1
while S2 is generating its own o0
-> feed a0 to S1
while Generation handles S2
```

这样减少 Generation 和 Simulator 互相等待造成的 GPU bubbles。

但它不是万能的。对于 OpenVLA-OFT 这种 action chunk 模型，generation:execution 比例约从 1:1 变成 1:15，simulator 执行 chunk 的时间更长；如果 simulator 是 LIBERO 这种 CPU-parallelized，CPU 反而成瓶颈，pipeline 的收益会变小。

## 5. Unified interface

RLinf-VLA 的接口分两类。

### Core functions

它实现 Gym-style API：

- `reset`
- `step`
- `auto_reset`
- `ignore_terminations`
- `chunk_step`

`auto_reset`：某个 sub-environment 结束或截断后自动 reset，避免环境闲置。

`ignore_terminations`：忽略成功 termination，只看 truncation，也就是一直跑到最大 episode length。这可用于 valid action mask 或 fixed-length rollout。

`chunk_step`：专门处理 action chunk，不只是循环调用 `step`。它要处理 chunk 中途 episode 结束时的 reset 策略：

- 立即 reset；
- 等整个 chunk 执行完再 reset。

这点很关键，因为 chunked VLA 的 episode boundary 如果处理错，advantage 和 log-prob 都会污染。

### Utility functions

包括：

- rollout / evaluation 视频可视化；
- GRPO 所需的 fixed reset state ids；
- 分组轨迹共享同一初始状态；
- 不同 simulator 的 observation/action 标准化。

GRPO 要求一组 trajectories 从同一个初始状态出发，才能做 group-relative comparison。因此 `use_fixed_reset_state_ids=True` 是一个很实用的小接口。

## 6. 算法设计选择

### 6.1 Multi-granularity

RLinf-VLA 支持 advantage 和 log-prob 在不同粒度上计算：

| 粒度 | 含义 |
|---|---|
| chunk-level | 整个 action chunk 当一个 macro-action |
| action-level | chunk 内每个 atomic action 单独算 |
| token-level | atomic action 内每个 action-dimension token 单独算 |

对 action chunks $c_t=(a_{t,1},...,a_{t,C})$，有两类 advantage：

- chunk-level：整个 chunk 共享一个 summed reward / advantage；
- action-level：每个 atomic action 有自己的 reward / advantage。

log-prob 则可在 chunk/action/token 三个层级算。

论文支持的组合是：

| Advantage \ Log-prob | Chunk | Action | Token |
|---|---:|---:|---:|
| Chunk-level | yes | yes | yes |
| Action-level | yes | yes | yes |

这个设计对 VLA 很实用，因为不同模型的 action representation 不一样，统一框架必须允许粒度切换。

### 6.2 PPO 设计

PPO 需要 critic。对大 VLA 单独维护 critic 网络成本太高，所以作者采用 actor-critic 参数共享：

- actor 是 VLA；
- critic 在 language model component 上加轻量 value head；
- 参考 RL4VLA。

对 chunked action，有两种 value strategy：

| 方式 | Value 输出 | 解释 |
|---|---|---|
| chunk-level value | $V:S\to R$ | 整个 chunk 一个值 |
| action-level value | $V:S\to R^C$ | 每个 atomic action 一个值 |

实验发现 action-level value consistently better：

- ManiSkill 上成功率更高；
- value loss 更低；
- LIBERO-Goal 上趋势相同。

另一个 PPO 经验是 **Partial Reset** 明显优于 Fixed Episode Length。因为任务目标是 success once，如果某个 sub-env 已经成功，继续等到最大长度会浪费 rollout。Partial Reset 让成功环境立刻重置，提升 sample efficiency。

### 6.3 GRPO 设计

GRPO 不训练 value model，而是从同一初始状态采样 $G$ 条 trajectory，用 group return 标准化 advantage：

$$
\hat A^{(i)} =
\frac{R^{(i)}-\text{mean}(\{R^{(j)}\}_{j=1}^G)}
{\text{std}(\{R^{(j)}\}_{j=1}^G)}
$$

它适合大模型，因为省掉 critic，但迁到 embodied tasks 有几个坑。

第一，Valid Action Mask。任务可能提前成功，但 rollout 是固定长度。优化时可以：

- 使用完整轨迹；
- 只用成功前的有效 timesteps。

作者称后者为 valid action mask。LIBERO-Goal 上 mask 有明显收益，ManiSkill 上收益不明显。

第二，Trajectory Length Normalization。在 valid action mask 下，不同 trajectory 的有效长度不同。作者把每条 trajectory 的 timestep loss 按 $1/T_i^{succ}$ 缩放，避免长轨迹主导梯度。LIBERO-Goal 上 normalization 明显提升。

第三，Success Rate Filter。GRPO 如果一组轨迹全成功或全失败，group advantage 为零或无信息。作者借鉴 DAPO，过滤掉全同 outcome 的 group。在 OpenVLA ManiSkill 上能避免 step 400 左右 collapse，但对 OpenVLA-OFT ManiSkill / LIBERO-Goal 收益不普遍。

## 7. 实验设置

### ManiSkill

- 训练 25 个 pick-and-place tasks；
- 有 object 和 receptacle variations；
- OOD protocol 参考 RL4VLA；
- 分 vision / semantic / execution generalization；
- 每个子设置 256 random episodes。

### LIBERO

- 使用 LIBERO-Spatial、Object、Goal、Long、90；
- 合计 130 tasks；
- 不是每个 group 单独训练，而是训练一个 unified model；
- 每个任务 50 episodes，3 random seeds；
- 使用 OpenVLA-OFT + GRPO。

### RoboTwin

六个任务：

- beat block hammer；
- move can pot；
- place empty cup；
- pick dual bottles；
- lift pot；
- handover block。

训练：

- 每任务 1,000 fixed randomized scene seeds；
- 评估 128 unseen seeds；
- 用于 OOD generalization。

## 8. 主结果

### ManiSkill

| Method | In-Distribution | OOD Avg. |
|---|---:|---:|
| OpenVLA Base | 53.91 | 39.10 |
| OpenVLA RLinf-GRPO | 84.38 | 75.15 |
| OpenVLA RLinf-PPO | 96.09 | 81.93 |
| OpenVLA-OFT Base | 28.13 | 18.29 |
| OpenVLA-OFT RLinf-GRPO | 94.14 | 60.64 |
| OpenVLA-OFT RLinf-PPO | 97.66 | 77.05 |

结论：

- RL post-training 对 ID 和 OOD 都有大幅提升；
- ManiSkill 上 PPO 比 GRPO 更稳、更强；
- OpenVLA-OFT base OOD 很低，但 RL 后提升很大；
- base model 的起点会影响最终 generalization，不能只看最终绝对值。

### ManiSkill OOD breakdown

| Method | Vision | Semantic | Execution |
|---|---:|---:|---:|
| OpenVLA Base | 38.75 | 35.94 | 42.11 |
| OpenVLA RLinf-GRPO | 74.69 | 72.99 | 77.86 |
| OpenVLA RLinf-PPO | 82.03 | 78.35 | 85.42 |
| OpenVLA-OFT Base | 27.73 | 12.95 | 11.72 |
| OpenVLA-OFT RLinf-GRPO | 84.69 | 45.54 | 44.66 |
| OpenVLA-OFT RLinf-PPO | 92.11 | 64.84 | 73.57 |

这里最值得注意的是：OpenVLA-OFT 的 vision OOD 经 PPO 后到 92.11，但 semantic / execution 仍明显低于 vision。这说明 RL 很会补视觉扰动和局部执行，但语义重绑定、复杂执行泛化仍更难。

### LIBERO

单个 OpenVLA-OFT + RLinf-GRPO 模型训练 130 tasks：

| Group | Base | RLinf-GRPO |
|---|---:|---:|
| Object | 50.20 | 99.67 |
| Spatial | 51.61 | 98.93 |
| Goal | 49.40 | 98.32 |
| Long | 11.90 | 93.55 |
| 90 | 42.67 | 98.12 |
| Avg. | 42.09 | 98.11 |

最强的是 Long：从 11.90 到 93.55。这个结果说明大规模 multi-task RL 能显著补 LIBERO 长程任务，但也要注意 LIBERO 是模拟器 benchmark，成功 predicate 清楚，不能直接外推到真实机器人。

### RoboTwin

| Task | Base | RLinf-GRPO |
|---|---:|---:|
| Cup | 75.78 | 94.53 |
| Hammer | 10.15 | 96.09 |
| Bottles | 20.31 | 92.96 |
| Can | 9.37 | 83.59 |
| Pot | 3.13 | 70.31 |
| Hand | 28.13 | 70.31 |
| Avg. | 24.48 | 84.63 |

这个表很亮，尤其 Pot 从 3.13 到 70.31，Hammer 从 10.15 到 96.09。作者强调这是复杂 OOD 和长程双臂 manipulation 的 strong generalization。

我会谨慎一点：这证明 RLinf-VLA 的训练栈能把一个已有 VLA 在 RoboTwin 上有效 post-train，但还不能说明真实双臂机器人上同等成立。

## 9. 系统效率结果

效率 metric 是 throughput：

$$
\text{throughput} =
\frac{\text{total rollout environment frames}}
{\text{wall-clock time of one training epoch}}
$$

大致等于 rollout time + training time 的反比。

主要结论：

- OpenVLA + ManiSkill 上，Hybrid(pipe=2) 在 8 GPUs 相比 disaggregated baseline 最多 1.88x；
- 扩到更多 GPU 时仍有 1.61x-1.69x 优势；
- pipe=2 通常比 pipe=1 更能减少 rollout bubbles；
- OpenVLA-OFT + LIBERO / ManiSkill 上，collocated 或 hybrid(pipe=1) 可能优于 pipe=2，因为 action chunk 改变了 generation/execution 比例；
- 对 LIBERO 和 RoboTwin，RLinf-VLA 相比 SimpleVLA-RL 在 collocated 设定下有 1.34x-2.27x speedup；
- 加速来自 vectorized environments、adaptive communication、去掉冗余 log-prob recomputation 等系统优化。

这里的实践结论很重要：

> 最优 GPU placement 不是固定公式，而是由 simulator 类型、action chunk size、generation/execution ratio、CPU/GPU bottleneck 共同决定。

## 10. 额外消融

### Rollout data size

更大的 rollout batch size 在 PPO / GRPO 上都让 training epochs 维度的提升更明显。对 on-policy RL 来说，每次 policy update 的 rollout 数据量非常关键。

但这不等于无限增大 batch。大 batch 增加通信、显存和 wall-clock；论文主要展示的是在他们实验范围内，larger rollouts per epoch 更稳。

### LoRA

LoRA 本身不一定直接改变最终 performance，但会改变合适的超参。

在 ManiSkill + OpenVLA-OFT：

- PPO 中 w/ LoRA 和 w/o LoRA 曲线整体相近；
- GRPO 中，如果不使用 LoRA 但仍用 `lr=1e-4`，训练会 collapse 到 0；
- 把 no-LoRA 的学习率降到 `1e-5` 后可以稳定提升。

结论：

> LoRA 不是简单开关，它会改变 optimization stability 和 learning rate sweet spot。

### Success rate filtering

GRPO 的 success-rate filter 有时很有效，但不普适：

- OpenVLA ManiSkill：without filter 在 step 400 左右 collapse，with filter 缓解；
- OpenVLA-OFT ManiSkill / LIBERO-Goal：收益不明显。

所以 filter 应该当作可配置训练技巧，而不是默认真理。

## 11. 对 PPO 和 GRPO 的实践建议

### PPO

建议：

- 对 action chunk，优先尝试 action-level value；
- 用 shared actor-critic + lightweight value head，避免单独大 critic；
- 使用 Partial Reset 提高 sample efficiency；
- 尤其在 success-once objective 下，不要让成功环境继续空跑到固定长度；
- 对 chunk boundary 和 value target 要非常小心。

PPO 的优点：

- 在 ManiSkill 上更稳；
- OOD 结果强；
- value 提供细粒度 credit。

PPO 的代价：

- 要训练 critic；
- value head 设计和 value granularity 会影响很大；
- 系统资源更复杂。

### GRPO

建议：

- 每组 trajectories 要共享同一初始状态；
- valid action mask 对 early success task 很有用；
- 配合 trajectory length normalization；
- success-rate filter 可试，但要按任务验证；
- 注意 group 全成功/全失败会缺有效 advantage；
- 对超参、LoRA、rollout size 更敏感。

GRPO 的优点：

- 不需要 critic；
- 系统更轻；
- 适合大模型；
- LIBERO 130-task multitask 表现非常强。

GRPO 的代价：

- 依赖 group diversity；
- 在一些 setting 上稳定性不如 PPO；
- mask / norm / filter 的选择比较 task-dependent。

## 12. 与现有 VLA 笔记的关系

### 与 [[2026-Harness-VLA-Deep-Read]]

Harness VLA 是“部署时重新分工”：冻结 VLA 负责接触密集 primitive，agent harness 负责语义重绑定、解析控制、失败恢复。

RLinf-VLA 是“训练时基建”：如果我们未来不想只用 frozen VLA，而是要对 `VLA_ACT` 或 primitive policy 做 RL post-training，那么需要这样的统一 rollout/training/simulator 框架。

二者可以这样组合：

```text
Harness VLA:
  high-level agent + memory + primitives

RLinf-VLA:
  train / improve primitive policies with RL in simulation
```

换句话说，Harness VLA 告诉我们如何部署，RLinf-VLA 告诉我们如何把底层 VLA primitive 用 RL 打磨。

### 与 [[2026-Tau0-VLA]]

Tau0-VLA 强调 test-time computation 和 world-model-guided subtask search。RLinf-VLA 则是让 low-level / VLA policy 在模拟器里通过 RL 提升。

二者互补：

- Tau0-VLA：高层“下一步做什么”；
- RLinf-VLA：低层“这一步如何通过 RL 变得更可靠”。

如果把二者结合，可以有：

```text
world model proposes / evaluates subtask choices
-> RL-trained VLA executes local skill
-> simulator or real logs provide reward
-> RLinf-style infrastructure updates policy
```

### 与 [[2026-TTHE-Test-Time-Harness-Evolution]]

TTHE 改的是 harness 程序，RLinf-VLA 改的是 VLA policy 参数。它们处在两个适应层：

| 层 | 适应对象 | 信号 |
|---|---|---|
| TTHE | executable harness | execution traces + proxy |
| RLinf-VLA | VLA policy | simulator reward |

对机器人系统来说，最佳形态可能是双层：

- 用 TTHE-like 方法在测试时改调度、验证、恢复；
- 用 RLinf-like 方法离线或在线仿真中改 primitive / VLA 参数。

### 与 [[2026-SkillRise]]

SkillRise 关注跨任务技能文档如何被后续任务 reward 监督。RLinf-VLA 关注 RL rollout 的系统效率。两者结合后，可以训练：

- skill curator；
- primitive selector；
- VLA low-level policy；
- memory writer；
- verifier。

但要注意 credit assignment 会非常复杂：skill document、harness rule、low-level action 都影响最终 reward。

## 13. 对 Harness VLA 的启发

### 启发 1：如果要让 Harness VLA 从 system paper 变成 trainable system，需要 RLinf 这类基建

当前 Harness VLA 的主线是：冻结 VLA + agentic harness。它强在无需重新训练模型，弱在 primitive 的能力边界固定。

RLinf-VLA 说明：如果我们要继续推进，可以把 `VLA_ACT` 从 frozen primitive 变成 RL-refined primitive：

```text
frozen VLA_ACT
-> collect simulation rollouts under Harness VLA scheduler
-> train with PPO / GRPO
-> deploy improved contact primitive
```

这样 harness 不只是在“绕开 VLA 的弱点”，还可以反过来收集任务分布，让 VLA 的弱点被 RL 修补。

### 启发 2：Harness VLA 的 low-level primitive 应该按 action chunk 粒度设计日志

RLinf-VLA 强调 `Chunk -> Atomic Action -> Token`。Harness VLA 如果要训练或诊断 VLA primitive，也应该保存多粒度日志：

```yaml
chunk:
  observation: image/depth/proprio
  instruction: "grasp the mug handle"
  atomic_actions:
    - ee_delta: ...
      gripper: ...
      contact: ...
      reward: ...
      success_predicate: ...
  token_logprobs:
    - ...
```

否则后续很难做 PPO value、GRPO log-prob、failure localization 或 verifier training。

### 启发 3：success once 和 early termination 要单独处理

机器人任务经常是 success-once：

- 抓起来一次就算当前 subtask 成功；
- 门打开一次就算成功；
- 物体放到目标区域就算成功。

如果环境成功后还继续 rollout，训练信号会被后续无意义动作污染。RLinf-VLA 的 partial reset 和 valid action mask 对 Harness VLA 很实用：

- physical / sim training：成功后 reset 或进入下一 subtask；
- log analysis：只给成功前动作 credit；
- retry policy：不要把成功后的冗余动作当失败或学习目标。

### 启发 4：不要把 PPO / GRPO 选择当宗教问题

这篇显示：

- ManiSkill 上 PPO 更强更稳；
- LIBERO 130-task 上 GRPO 表现很强；
- GRPO 的 mask / norm / filter 很 task-dependent。

所以 Harness VLA 后续如果做 RL，不应预设“GRPO 一定适合所有 VLA”。更合理的是：

| 场景 | 倾向 |
|---|---|
| dense-ish reward / action chunk credit 明确 | PPO + action-level value |
| 大模型、critic 成本太高、binary outcome | GRPO |
| early success / fixed-length rollout | GRPO + valid action mask + length norm |
| group 内全成全败很多 | success-rate filter 或 curriculum |

### 启发 5：系统吞吐会决定研究能不能迭代

Harness VLA / Tau0-VLA / SkillRise 这类想法都很吃实验迭代。如果 VLA-RL 一轮训练要跑几天，很多 idea 会死在工程成本上。

RLinf-VLA 的价值在这里很朴素：

- simulator placement；
- vectorized env；
- offload/onload 策略；
- chunk_step；
- log-prob recomputation 去冗余；
- rollout batch size；
- multi-GPU pipeline。

这些“脏活”会决定一个方法到底是论文图里好看，还是能被反复 ablate。

## 14. 局限

这篇的局限也明显：

- 主要是模拟器结果，没有真实机器人部署；
- 任务 reward / success predicate 相对清楚；
- 统一模型在 LIBERO 上很强，但 LIBERO 本身可能存在 benchmark saturation；
- 系统设计复杂，复现需要较大 GPU 资源；
- PPO/GRPO 的实践建议仍是经验性，不是理论保证；
- 与 SimpleVLA-RL 的效率对比受实现细节影响，不能只看 speedup 数字；
- 没有深入讨论 sim-to-real、reward hacking、safety constraint；
- 没有把 high-level planning / memory / harness evolution 纳入训练闭环。

## 15. 我对这篇的判断

我给 8/10。

优点：

- 把 VLA-RL 的系统问题拆得很清楚；
- 多模拟器、多模型、多算法统一接口很实用；
- hybrid allocation + pipelining 对 GPU simulator 很有意义；
- 实验覆盖 ManiSkill、LIBERO、RoboTwin；
- PPO/GRPO 的细节建议对后续实验非常可操作。

扣分点：

- 方法新意主要是系统整合，不是核心 RL 算法；
- 真实机器人和 sim-to-real 证据缺失；
- 结果强但容易被理解成“RL 解决 VLA 泛化”，实际还需要看任务分布、reward 和 benchmark 难度；
- 对 high-level agentic robotics 的连接还需要我们自己补。

我的一句结论：

> RLinf-VLA 是给 VLA-RL 铺路的工程基座：它不回答“机器人应该怎么想”，但回答了“想用 RL 打磨 VLA 时，怎样让 simulator、generation、training 不互相拖后腿”。

