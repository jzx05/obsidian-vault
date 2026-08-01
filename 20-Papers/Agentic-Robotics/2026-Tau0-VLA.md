---
title: "Tau0-VLA: A Hierarchical Robot Foundation Model with World-Model-Guided Test-Time Computation"
authors:
  - Xiaowei Cai
  - Yunuo Cai
  - Bingao Chen
  - Jingxiao Chen
  - Zhi Chen
  - Siyuan Feng
  - Tengyu Hou
  - Jingshun Huang
  - Han Jiang
  - Runkun Ju
  - Dong Li
  - Mingxiang Li
  - Shaowei Li
  - Xinchen Li
  - Yifan Li
  - Yi Liu
  - Zhongyuan Liu
  - Jianlan Luo
  - Junwen Miao
  - Ruiqi Ni
  - Buqing Nie
  - Mingjie Pan
  - Xinlin Ren
  - Jianheng Song
  - Jiaxu Wang
  - Peiqi Wang
  - Sen Wang
  - Xiaoyan Wang
  - Dafeng Wei
  - Dongming Wu
  - Pengwei Xie
  - Pu Yang
  - Hangjian Ye
  - Xiangyu Yue
  - Jinyu Zhang
  - Qinglin Zhang
  - Xueyong Zhao
  - Yue Zhou
year: 2026
venue: preprint
tags:
  - paper
  - VLA
  - hierarchical-policy
  - world-model
  - test-time-compute
  - long-horizon-manipulation
  - cross-embodiment
status: read
rating: 8.5/10
date-read: 2026-08-01
url: https://tau0-vla.github.io/
pdf: "C:/Users/huawei/Zotero/storage/S4QE2ARI/Cai et al. - τ0-VLA a Hierarchical Robot Foundation Model with World-Model-Guided Test-Time Computation.pdf"
related:
  - "[[2026-Hi-VLA-Deep-Read]]"
  - "[[2026-Harness-VLA-Deep-Read]]"
  - "[[2026-TurboVLA]]"
  - "[[World-Model]]"
---

## TL;DR

Tau0-VLA 的核心不是让低层 VLA 生成更多动作候选，而是把长程任务中最关键的决策单位提升到**语言子任务**：高层 VLM 先提出候选子任务，世界模型预测每个候选执行后的终态图像，价值模型按全局任务进度打分，beam search 向前展开若干层，最后反思模型结合保留分支生成真正提交给低层 VLA 的下一条指令。

它回答的问题是：

> 当“下一步做什么”比“这一步怎么做”更容易毁掉长程任务时，能否把额外推理算力按需分配给高层决策，并在物理执行前比较候选后果？

论文给出的答案是肯定的。最有说服力的结果有两层：

- 即使不启用搜索，只加入高层子任务分解与执行记忆，四个长程真机任务的平均成功率也从直接执行 Tau0-VLA 的 **27.5% 提升到 45.0%**。
- 在三个任务上加入 world-model-guided TTC 后，成功次数分别从 **5/10、6/10、5/10** 提升到 **7/10、9/10、7/10**。

一句话抓住它：

> Tau0-VLA 把世界模型从“生成低层动作的辅助表征”变成“高层子任务提交前的后果模拟器”，并把长程机器人推理写成一个可扩展计算预算的搜索问题。

## 1. 问题与动机

长程操作不是把短程动作简单重复很多次。房间清理、做饭、冲奶茶这类任务要求机器人持续回答：

- 当前已经完成了什么？
- 哪一步失败了，记忆是否需要回滚？
- 下一条子任务应该是什么？
- 这个候选子任务执行后，会让物理状态更接近最终目标吗？

低层控制误差有时可以靠重试纠正，但高层语义错误更危险：

```text
机器人可能把错误的子任务执行得非常完美。
```

已有分层 VLA 通常已经有“高层出子任务、低层出动作”的接口，但高层仍常用一次前向传播直接提交决策。这样的问题是：

1. 不会显式比较多个候选。
2. 不会在执行前检查候选的物理后果。
3. 简单状态和困难状态使用相同计算量。
4. 错误通常要等环境已经被改变后才能发现。

Tau0-VLA 的关键判断是：**subtask 是最适合做 test-time computation 的粒度**。

| 搜索粒度 | 优点 | 局限 |
|---|---|---|
| action level | 能处理局部控制歧义 | 视野短、搜索树巨大、难表达长程语义后果 |
| language-only plan | 能看较长时间尺度 | 候选比较没有落到真实物理状态 |
| language subtask | 稀疏、语义清晰、执行后会产生明显状态变化 | 需要可靠的视觉后果预测与价值判断 |

因此论文选择在子任务层做 `propose -> predict -> evaluate -> search -> reflect -> commit`。

## 2. 整体架构

系统分为高低两层：

```text
全局任务 + 当前多视角观察 + 上一子任务 + 执行记忆
                         |
                         v
              Proposal Model (Qwen3.5-9B)
                  /                 \
          高置信：直接提交       低置信：进入 TTC
                                   |
                    候选语言子任务 x N
                                   |
                   World Model 预测终态图像
                                   |
                    Value Model 对后果打分
                                   |
                    Top-B beam，递归到深度 D
                                   |
                       Reflective Model
                                   |
                         提交下一子任务
                                   |
                                   v
          Low-Level VLA (Qwen3.5-2B + MoT action expert)
                                   |
                     H-step continuous action chunk
                                   |
                   真实新观察写回下一轮记忆
```

高层决定 **what to do now**，低层决定 **how to execute it**。

### 2.1 高层输入与记忆

第 `t` 个逻辑推理步的上下文为：

$$
h_t = (\ell, M_{t-1}, z^\star_{t-1}, o_t)
$$

其中：

- $\ell$：全局任务指令；
- $M_{t-1}$：此前携带的执行记忆；
- $z^\star_{t-1}$：上一轮提交的子任务；
- $o_t$：当前多视角观察。

Proposal model 同时输出更新后的记忆和直接候选：

$$
(z_t^{dir}, M_t) = P(h_t)
$$

这里的记忆不是静态日志，而是**必须与当前视觉状态对齐的可纠错状态估计**。如果上一轮抓空，模型不应因为文本记忆写着“已经拿起杯子”就继续下一步。

### 2.2 自适应路由：什么时候值得多算

系统不对每个状态都运行昂贵搜索。Proposal model 一次前向传播后，直接复用生成 token 的置信统计：

- 全部生成 token 的平均概率 $u_t^{all}$；
- `<memory>` 字段内 top-1 与 top-2 logit 的平均 margin $u_t^{mem}$。

当任一统计低于阈值时进入 TTC：

$$
g_t = \mathbf{1}[u_t^{all} \le \delta_{all} \lor u_t^{mem} \le \delta_{mem}]
$$

路由本身不需要额外模型调用。统计形式跨任务相同，但论文又说明**阈值按任务在 held-out validation data 上单独校准**，所以它还不是完全免校准的通用路由器。

### 2.3 World-model-guided beam search

当 $g_t=1$ 时，系统运行：

$$
C_t = \operatorname{Search}(h_t, P, W, V, N, B, D)
$$

三个预算参数是：

- $N$：每个保留分支采样多少个候选，branching factor；
- $B$：每层全局保留多少条路径，beam width；
- $D$：向未来展开多少个子任务，search depth。

对分支 $b$ 的候选子任务 $z$：

$$
\hat{o}(b \oplus z) = W(\tilde{o}(b), z)
$$

$$
v(b \oplus z) = V(\ell, z, \hat{o}(b \oplus z))
$$

$$
S(b \oplus z) = S(b) + v(b \oplus z)
$$

世界模型只使用 head-camera RGB 和候选子任务，预测该子任务结束时的 head-camera 图像。价值模型同时看全局任务、候选子任务和预测终态，输出候选质量。

搜索有三个容易忽略的细节：

1. 每层不是每个父节点各留 top-k，而是把所有 child 按累计分数做**全局 Top-B**。
2. 搜索中的 memory 是 branch-local，不会覆盖真实持久记忆 $M_t$。
3. 最终 reflective model 不必从候选集合中硬选一个，它可以综合保留分支后生成新的子任务，作用更像“带证据的修订器”。

### 2.4 为什么要有 reflective model

如果直接取 beam 中最高分候选，系统完全受限于 proposal 的采样集合和 value model 的排序误差。Reflective model 接收：

```text
真实当前上下文 + 最终保留路径 + 每条路径的终态图像 + 累积分数
```

再生成最终子任务：

$$
z_t^\star = F(\bar{h}_t, C_t)
$$

训练目标不是复制第一候选，而是匹配 ground-truth next subtask，因此它可以修复候选计划中的缺陷。

## 3. 低层 VLA

低层模型以 Qwen3.5-2B 为视觉语言骨干，连接 Mixture-of-Transformers action expert：

- backbone tokens 与 action tokens 在 full-attention layer 中联合注意；
- 两类 token 由参数独立的 Transformer stream 处理；
- action expert 用 conditional flow matching 从噪声恢复连续动作 chunk；
- 主模型 action horizon 为 `H = 30`，推理采用 10 次 uniform Euler update。

### 3.1 统一 40 维状态/动作空间

论文用共享的 40 维槽位覆盖多种机器人：

| 维度 | 含义 |
|---:|---|
| 1-3 | 左末端位置 |
| 4-9 | 左末端 Rot6D 姿态 |
| 10-12 | 右末端位置 |
| 13-18 | 右末端 Rot6D 姿态 |
| 19-20 | 左右夹爪 |
| 21-22 | 腰部两个自由度 |
| 23-24 | 平面底盘速度 |
| 25-32 | 左臂最多 8 个关节 |
| 33-40 | 右臂最多 8 个关节 |

不同 embodiment 用 state/action mask 激活自己的通道，无效通道置零且不计入 loss。控制元数据明确机器人类型、`eef/eef_wbc/joint` 模式，以及是否启用 whole-body control。

这个设计的价值是：同一个动作头可以支持固定底座、双臂、移动操作和全身控制，而不需要每个机器人单独做输出 head。

## 4. 训练数据与训练方式

### 4.1 低层数据规模

低层共训练于 **40,115 小时**异构真实机器人数据：

- 内部数据约 23.4K 小时：AGIBOT G1 21.9K、G2 585h、ARX AC One 578h、Franka 347h；
- 公共数据约 16.7K 小时，其中 9.25K 小时是开源 UMI 数据；
- 来源包括人工遥操作、自主策略 rollout、UMI-style recording；
- 额外混入 instruction following、visual grounding、空间/深度推理和 robot-centric perception 数据。

低层训练分三阶段：

1. **Knowledge-isolated co-training**：action expert 可以注意 VLM backbone，但 action loss gradient 在接口处截断，避免过早破坏预训练语义能力。
2. **End-to-end co-training**：解除隔离，全模型联合训练，同时保留多模态辅助监督。
3. **Task-specific adaptation**：对每个部署任务用少量专用 demonstrations 适配机器人、本体、相机、物体布局和成功条件。

因此“foundation model”并不等于所有实验都零样本。主结果里的部署任务经过 task-specific adaptation，这一点必须和大规模预训练能力分开理解。

### 4.2 高层四个模型如何训练

| 模块 | 初始化 | 监督来源 |
|---|---|---|
| Proposal | robot-pretrained Qwen3.5-9B | 任务、阶段、可执行子任务、视频与纠错记忆 |
| World model | Step1X-Edit | 子任务 segment 的起始/终止 head-camera 帧 |
| Value model | 同一 Qwen3.5-9B checkpoint | 离线 imagined rollout 中候选相对 ground truth 的五档质量 |
| Reflective model | 同一 Qwen3.5-9B checkpoint | 候选路径、预测图像、分数到 ground-truth next subtask |

Value model 被写成五选一 VQA，五档从 clearly wrong 到 clearly correct，映射到 `[0.05, 0.95]` 的标量。

### 4.3 执行记忆的自纠错训练

高层数据从已有 `task / stage / executable-subtask` 标注自动构造，不需要为记忆纠错再做人工作标。作者只扰动输入 memory，target 仍读取与真实 demonstration 对齐的正确状态。

| 样本族 | 输入记忆 -> 目标记忆 | 训练目的 | 比例 |
|---|---|---|---:|
| within-subtask | $M_n \to M_n$ | 正常保持当前进度 | 58% |
| transition | $M_n \to M_{n+1}$ | 完成后切换下一子任务 | 15% |
| catch-up | $M_{n-1} \to M_n$ | 修复落后于画面的记忆 | 10% |
| rollback | $M_{n+1...n+3} \to M_n$ | 修复过度乐观、提前推进的记忆 | 12% |
| error-think | $M_n \to$ 按错误类型修复 | 识别失败并给恢复动作 | 5% |

例如抓空属于 recoverable failure：记忆保持当前阶段并重试；如果动作破坏了此前状态，则回滚到前一子任务。这个设计非常重要，因为高层规划正确与否取决于它相信的进度，而不只是当前图像。

## 5. 实验设置

### 5.1 机器人平台

- AGIBOT G1：四个主要长程任务；
- ARX AC One：Book Organization、Collect Laundry；
- 双臂 Franka Research 3：Tidy Makeup Table。

### 5.2 任务

| 任务 | 子任务数 | 典型时长 | 核心难点 |
|---|---:|---:|---|
| Clean Room | 25 | 8 min | 多房间导航、取放、挂包、交接、垃圾处理 |
| Prepare Ingredients | 14 | 4 min | 冰箱交互、取食材、打蛋、搅拌、工具归位 |
| Tomato and Egg Stir Fry | 22 | 10 min | 切配、烹饪、调味、装盘、清理，盐的状态难观察 |
| Make Milk Tea | 13 | 3 min | 加料、倒奶茶、封盖、插吸管 |
| Collect Laundry | 5 | 1 min | 移动操作与跨房间执行 |
| Tidy Makeup Table | 2/2/4 | 共约 30 s | 相同画面下按语言选择对象、手臂、顺序和目的地 |
| Book Organization | 3 swaps | 1 min | 根据书高排序，OOD 初始排列 |

物理实验每个 method-task setting 都只有 **10 次独立 trial**。成功要求所有必要 milestone 和终止条件均满足；progress 使用带前置依赖的 milestone DAG，并给首次完成、重试后完成、未完成分别记 1、0.5、0 分。

## 6. 主要结果

### 6.1 长程任务：层级本身是否有用

以下四个长程任务都是真机结果，每格 10 次。最终层级行是 **Plan Once，不使用 beam search**。

| 方法 | Clean Room | Prepare Ingredients | Stir Fry | Make Milk Tea | 平均 SR | 平均 Progress |
|---|---:|---:|---:|---:|---:|---:|
| GR00T N1.7 | 0/10 | 1/10 | 0/10 | 0/10 | 2.5% | 45.29% |
| LingBot-VLA | 0/10 | 0/10 | 0/10 | 0/10 | 0.0% | 44.43% |
| pi0.5 | 4/10 | 2/10 | 0/10 | 3/10 | 22.5% | 73.05% |
| Tau0-VLA，直接执行 | 4/10 | 2/10 | 0/10 | 5/10 | 27.5% | 80.10% |
| **Tau0-VLA，层级 Plan Once** | **5/10** | **4/10** | **4/10** | **5/10** | **45.0%** | **87.85%** |

最关键的不是平均数，而是 Stir Fry：直接执行为 `0/10`，层级记忆与子任务控制后为 `4/10`。加盐几乎不改变视觉外观，直接策略容易重复加盐或跳过；显式 memory 能记录这一不可直接观察的事件。

Make Milk Tea 中两种 Tau0-VLA 都是 `5/10`，说明剩余瓶颈在封盖、插吸管等接触密集低层操作，而不是高层任务排序。分层不会自动修复所有 low-level failure。

### 6.2 跨本体直接执行

短任务不使用高层分解、记忆或搜索，只测试低层模型：

| 方法 | Collect Laundry | Cotton Pad | Eyelash Curler | Makeup Puff |
|---|---:|---:|---:|---:|
| GR00T N1.7 | 4/10 | 10/10 | 8/10 | 7/10 |
| LingBot-VLA | 2/10 | 9/10 | 3/10 | 3/10 |
| pi0.5 | 9/10 | 9/10 | 8/10 | 7/10 |
| **Tau0-VLA** | **10/10** | **10/10** | **9/10** | **10/10** |

这张表支持低层 VLA 的跨本体执行能力，但任务较短、每项同样只有 10 次，不应外推为开放世界通用性。

### 6.3 开放环子任务预测：TTC 是否真比采样更多候选强

| 设置 | Plan Once | Best-of-N | TTC |
|---|---:|---:|---:|
| Make Milk Tea | 64.7% | 70.0% | **83.0%** |
| Book Organization，ID | 66.0% | 87.3% | **88.0%** |
| Book Organization，OOD | 50.0% | 57.5% | **74.0%** |
| Clean Room | 72.0% | 74.0% | **87.0%** |

OOD Book Organization 最说明问题：

```text
Plan Once 50.0 -> Best-of-N 57.5 -> TTC 74.0
```

如果收益只来自“多采几个候选”，Best-of-N 应该接近 TTC；但多步后果展开和反思又带来 16.5pp，说明 search depth 与 consequence-aware reflection 确实提供了额外信息。

### 6.4 闭环真机：开放环正确率是否转化为任务成功

| 任务 | Plan Once SR / Progress | TTC SR / Progress |
|---|---:|---:|
| Make Milk Tea | 5/10 / 91.92% | **7/10 / 95.38%** |
| Book Organization | 6/10 / 66.67% | **9/10 / 93.33%** |
| Clean Room | 5/10 / 94.80% | **7/10 / 97.60%** |

三项都提升，说明开放环 next-subtask accuracy 的增益确实能传导到闭环执行，而不是只改善一个离线分类指标。

### 6.5 Compute scaling

论文改变 `N/B/D`，以 PFLOPs/sample 衡量推理成本。奶茶和书籍排序都呈现相同趋势：

```text
低预算区增加计算 -> 准确率快速增长
中等预算 -> 仍有收益
高预算 -> 边际收益下降并趋于饱和
```

这个结果支持“高层推理可 scaling”，但没有给出一个跨任务统一的最优预算，也没有把 PFLOPs 转化为完整系统的延迟、能耗或成本。

## 7. 论文最重要的贡献

### 7.1 把 TTC 放在正确的时间尺度

这篇论文最有价值的不是 beam search 本身，而是明确了额外推理应该用在哪：

```text
不是每个动作 tick 都搜索，
而是在稀疏但后果重大的 subtask boundary 上搜索。
```

这使高层可以慢、低层可以快。部署时两者异步运行：高层后台约每 1 秒刷新一次子任务 cache，低层约 30 Hz 读取缓存并继续控制。高层偶尔变慢不会阻塞每个控制 tick。

### 7.2 在“提交前”使用世界模型

与用 world model 给已经选定的子任务生成视觉 subgoal 不同，Tau0-VLA 在 commit 前比较多个候选的 predicted outcome。世界模型改变的是**选哪条子任务**，而不只是**怎样执行已选子任务**。

### 7.3 把记忆错误当成训练分布的一部分

长程系统最危险的错误之一不是忘记，而是“记错了还很自信”。catch-up、rollback 和 error-think 数据让 memory update 具备显式恢复目标。这比单纯增加 context window 更贴近真实部署故障。

### 7.4 一个模型覆盖多本体

40 维统一空间、mask 和 control metadata 是扎实的系统工程贡献。它让论文的高层机制可以复用同一低层接口，而不需要为移动人形、双臂台面机器人和固定臂分别设计动作头。

## 8. 批判性解读

### 8.1 强结论

1. **层级子任务接口和 execution memory 对长程任务有效。** 同一个低层 Tau0-VLA 从 27.5% 到 45.0% 是最干净的控制比较。
2. **TTC 不只是 Best-of-N。** 四个开放环设置全部领先，尤其 OOD 排书的差距支持多步后果搜索。
3. **开放环提升能转化为闭环真机收益。** 三个任务成功率都增加 2-3 次/10 trials。
4. **记忆对视觉不可辨识的进度尤其重要。** 加盐案例很好地说明，当前 RGB 不能替代事件记忆。
5. **额外计算存在收益递减。** 这给自适应预算留下了清晰工程空间。

### 8.2 不能过度解读的地方

#### 样本量很小

每个真机 setting 只有 10 次 trial。`5/10 -> 7/10` 很有工程意义，但统计不确定性仍大；论文没有报告置信区间或多随机种子重复。

#### 开放环正确率依赖 LLM judge

next-subtask prediction 使用 GPT-5.4 做语义裁判。每个 unique tuple 两次 temperature-zero 判断，冲突时追加第三次，必要时第四次多数投票。协议比单次 LLM judgment 严谨，但仍不是人工 gold evaluation，也可能继承裁判模型的语义偏差。

#### “自适应”仍有任务级校准

路由特征跨任务共用，但阈值按任务在验证集上分别校准。面对真正未知任务时，怎样自动选择阈值尚未解决。

#### 世界模型只预测单个头部 RGB 终态

它没有显式预测深度、触觉、本体状态、隐蔽对象状态或多视角一致性。对“是否加过盐”这种视觉不可辨识事件，真正可靠的仍是 execution memory，而不是世界模型。

#### Search score 是简单累计

路径分数直接相加，没有说明深度归一化、风险敏感目标、uncertainty propagation 或 calibration。较长路径、早期过高 value、世界模型复合误差都可能扭曲排序。

#### 高层组件多，因果拆分仍不够完整

论文比较了 Plan Once、Best-of-N、TTC，也展示 memory 的任务作用，但没有完整报告：

```text
proposal only
+ corrected memory
+ world model
+ value model
+ depth search
+ reflection
+ confidence routing
```

因此还不能精确分配每个组件对最终 SR 的独立贡献。

#### 部署任务并非完全零样本

低层训练的 Stage 3 会对每个部署任务做少量 task-specific demonstrations 适配。高层数据也来自已有细粒度 task/stage/subtask annotation。论文证明的是“大规模预训练 + 任务适配 + 推理时搜索”的系统能力，不是仅凭自然语言就解决任意新长程任务。

#### 缺少与更接近的 agentic / planning 系统的完整真机对照

主长程表对比的是直接执行的 GR00T、LingBot-VLA、pi0.5，而不是同等工具、记忆、世界模型预算下的 Harness VLA、Goal2Skill、VINE 或其他强 agentic planner。它证明 Tau0 的层级版本优于 flat baselines，但尚未证明整个 TTC 设计在统一协议下是最佳高层规划器。

#### 计算成本报告不够部署化

论文给 PFLOPs/sample 和趋势图，但没有完整报告：

- 高层路由比例；
- 每次 TTC 的 wall-clock latency；
- 端到端任务额外耗时；
- world model / value / reflection 的分项成本；
- 云端调用失败、延迟抖动和能耗。

异步 cache 能隐藏部分延迟，但不能消除决策陈旧和资源成本。

## 9. 与现有笔记的关系

### 对比 [[2026-Hi-VLA-Deep-Read|Hi-VLA]]

| 维度 | Hi-VLA | Tau0-VLA |
|---|---|---|
| 主要问题 | 哪些分层 VLA 设计选择真正重要 | 如何给困难高层决策分配更多推理算力 |
| 高层决策 | VLM 直接生成子指令 | proposal + WM + value + beam + reflection |
| 世界模型 | 不是核心必需组件 | 候选提交前后果预测器 |
| 记忆 | 比较窗口、摘要、跨 episode 经验 | 可纠错的在线 execution state |
| 核心证据 | 系统化设计空间消融 | TTC compute-accuracy 与真机闭环收益 |

Hi-VLA 给的是“分层系统应该怎样搭”的设计地图；Tau0-VLA 在其中进一步选择了一个方向：**让高层决策本身成为可搜索、可验证、可按预算扩展的推理过程**。

### 对比 [[2026-Harness-VLA-Deep-Read|Harness VLA]]

| 维度 | Harness VLA | Tau0-VLA |
|---|---|---|
| 低层能力组织 | 冻结 VLA + 解析控制器，统一作为工具 | 单个通用低层 VLA |
| 高层输出 | 结构化原语调用 | 自然语言子任务 |
| 记忆 | task trace + global failure model | 当前 episode 的可纠错执行记忆 |
| 规划依据 | 真实观察、工具反馈、重试 | 真实观察 + imagined visual consequences |
| 学习策略 | 尽量冻结，靠系统编排 | 大规模预训练并做任务适配 |

两者可以结合：Harness 的可审计工具接口负责明确约束与恢复，Tau0 的 TTC 在高风险工具调用前比较未来后果。

### 对比 [[2026-TurboVLA|TurboVLA]]

两篇论文处理的是互补瓶颈：

```text
TurboVLA：具体子任务给定后，如何把低层执行做得更快、更小。
Tau0-VLA：长程任务中，如何更可靠地决定下一条具体子任务。
```

最自然的系统组合不是二选一，而是“慢速 TTC 高层 + 高速轻量低层”。Tau0 当前低层约 30 Hz，已采用异步双系统；TurboVLA 进一步说明低层不一定需要始终以大 VLM 为中心。

## 10. 对后续研究的启发

1. **将 value 从单点五档分类扩展到风险分布。** 同时预测成功概率、不可逆失败概率和剩余步数，避免简单累计分数。
2. **让 world model 输出不确定性。** 路由和 beam pruning 应考虑“看起来好但预测不可靠”的候选。
3. **多模态后果预测。** 除 RGB 外加入深度、接触、夹爪状态、对象关系图和事件变量。
4. **学习跨任务统一路由。** 用预计 decision regret / value of computation 代替任务手调阈值。
5. **搜索预算按风险分配。** 不可逆操作、接触密集步骤、状态歧义处用更深搜索，普通运输走 fast route。
6. **显式验证 imagined vs. real outcome。** 每次执行后比较世界模型预期与真实观察，把 prediction error 写入 memory 并校准后续搜索。
7. **报告系统成本。** 除 PFLOPs 外，增加每次决策延迟、route ratio、能耗、任务完成时间和每个成功 trial 的计算成本。
8. **更强因果消融。** 固定同一低层、数据和推理预算，逐项拆分 memory correction、WM、value、depth、reflection 和 router。
9. **与 agentic baseline 对齐。** 在相同观察、工具、成功检测和试验预算下，与 Harness VLA、Goal2Skill、VINE 等比较。

## 11. 最终评价

| 维度 | 评价 |
|---|---|
| 核心思想 | **9/10**：把额外计算放到子任务边界，问题粒度选得很准 |
| 方法完整性 | **8.5/10**：proposal、记忆、WM、value、search、reflection、低层执行形成完整闭环 |
| 低层工程 | **9/10**：40K 小时、统一 40-D、三类本体、异步 30 Hz 控制很扎实 |
| 实验说服力 | **7.5/10**：有真机闭环和 OOD，但每格仅 10 trials，缺少置信区间 |
| 因果分析 | **7/10**：Plan Once/Best-of-N/TTC 对照有效，但组件级拆分不完整 |
| 可复现性 | **5.5/10**：大规模私有数据、任务适配、多个专用模型使外部完整复现困难 |
| 综合 | **8.5/10**：是 world-model + hierarchical VLA + test-time compute 方向的重要系统论文 |

> [!summary] 只记住四件事
> 1. Tau0-VLA 搜索的是**语言子任务序列**，不是低层动作序列。
> 2. 世界模型在执行前预测候选终态，value model 排序，beam search 展开，reflective model 最终提交。
> 3. 不带搜索的层级记忆已经把长程平均 SR 从 27.5% 提到 45.0%；TTC 又在三个真机任务上继续提升。
> 4. 结论仍受 10-trial 小样本、LLM judge、任务级路由阈值和 task-specific adaptation 限制。

## 资料

- 项目主页：<https://tau0-vla.github.io/>
- 本地 PDF：`C:\Users\huawei\Zotero\storage\S4QE2ARI\Cai et al. - τ0-VLA a Hierarchical Robot Foundation Model with World-Model-Guided Test-Time Computation.pdf`

