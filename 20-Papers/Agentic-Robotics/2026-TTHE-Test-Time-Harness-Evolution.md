---
title: "TTHE: Test-Time Harness Evolution"
authors:
  - Jun Nie
  - Yonggang Zhang
  - Jun Song
  - Qianshu Cai
  - Dahai Yu
  - Yike Guo
  - Xinmei Tian
  - Bo Han
year: 2026
venue: arXiv preprint
arxiv: 2607.08124v1
tags:
  - paper
  - agentic-ai
  - test-time-adaptation
  - harness
  - self-evolving-agents
  - verification
  - agentic-robotics
status: read
rating: 8.5/10
date-read: 2026-08-01
url: https://arxiv.org/abs/2607.08124
code: https://github.com/junnie00/TTHE
pdf: "C:/Users/huawei/Zotero/storage/BYWNGGUC/Nie et al. - 2026 - TTHE Test-Time Harness Evolution.pdf"
related:
  - "[[2026-Harness-VLA-Deep-Read]]"
  - "[[2026-Tau0-VLA]]"
  - "[[2026-Agentic-VLA-Deep-Read]]"
  - "[[World-Model]]"
---

# TTHE: 测试时让 harness 自己进化

## TL;DR

TTHE 的核心不是让 LLM 在测试时多采样几次，也不是让模型权重在线微调，而是把 agent 外面的 **executable harness** 当成测试时适应的状态：在每个未标注 test batch 上，运行当前 harness 得到执行轨迹，再让多个 proposer 读这些轨迹并改写 harness 代码，最后由 judge 根据无标签执行信号选择一个候选，把它持久化到后续 batch。

一句话抓住它：

> 同一个 frozen LLM 的能力不是只由权重决定，也由它外面的程序决定；TTHE 让这个程序在测试流中根据自己的执行痕迹持续变好。

它最有价值的地方在于，把“调试、验证、恢复、grounding”从一次回答里的临时技巧，提升为可以跨输入保留的程序级行为。对 Harness VLA 来说，这篇的直接启发是：不要只把 harness 设计成固定工具库和固定 memory，可以把 harness 本身变成一个可演化对象，让机器人执行日志反过来修改 primitive 调度、验证规则、重试策略和记忆写入逻辑。

## 1. 论文想解决什么问题

现代 LLM agent 已经不是单次预测器，而是一个可执行系统：

- 构造上下文；
- 调用工具；
- 执行代码或动作；
- 读取中间状态；
- 检查错误；
- 恢复失败。

这些行为不是模型权重直接决定的，而是由模型外面的 harness 决定的。传统做法通常是在开发集上调 prompt、调 workflow、调工具调用逻辑，最终冻结一个静态 scaffold，再拿去测试。

TTHE 质疑的是这个“测试时静态”的习惯：

> 既然测试过程会产生大量 execution traces，为什么 harness 不能在测试时用这些未标注轨迹继续改进？

这里的关键约束非常重要：

- 不更新模型权重；
- 不使用 gold labels；
- 不用额外强 teacher；
- 不训练单独 adaptation model；
- proposer、solver、judge 都是同一个 frozen LLM 的不同角色和 harness；
- 所有变化都发生在 surrounding executable program。

因此这篇的位置介于三类工作之间：

- 不是 test-time training，因为不改参数；
- 不是 Reflexion，因为不只是给单个回答追加文字经验；
- 不是开发时 workflow search，因为优化发生在 evaluation stream 内部。

## 2. 核心概念：harness 是测试时状态

论文把一个 agent 写成：

$$
(o_H(x), \tau_H(x)) = H(x)
$$

其中：

- $H$ 是 harness，也就是控制 frozen model、tools 和 execution environment 的程序；
- $o_H(x)$ 是输出；
- $\tau_H(x)$ 是执行轨迹；
- gold $y$ 只用于最后测量，不进入适应循环。

真实测量目标是：

$$
J_D(H) = \mathbb{E}_{(x,y)\sim D}[\mathbf{1}[o_H(x) \cong y]]
$$

但 $y$ 在测试时不可见，所以 TTHE 不能直接优化这个目标。它只能用执行产生的 proxy signals 和 traces 来判断“这个 harness 看起来是不是更可靠”。

TTHE 的适应状态不是 prompt string，而是 arbitrary executable program。作者实现里 harness 是 Python class，proposer 可以改：

- context construction；
- tool-use order；
- execution feedback loop；
- multi-sample generation；
- deterministic contract checks；
- fallback / recovery branch；
- output validation；
- task-specific grounding。

这个设定比 prompt optimization 更强，因为代码级 harness 可以表达条件分支、缓存、重跑、结构化验证和工具交互逻辑。

## 3. 方法：population-based generate-and-judge

测试流被分成 batch。对于 batch $X_t$，TTHE 从当前已提交 harness $H_t$ 出发，演化 $G$ 条分支，跑 $R$ 轮，最后选出一个 $H_{t+1}$：

```text
current committed harness H_t
        |
        v
observe on unlabeled batch X_t
        |
        v
G candidate branches
        |
        v
proposers read code + traces + proxy signals, then edit harness code
        |
        v
execute new candidates and collect traces
        |
        v
repeat for R rounds
        |
        v
judge chooses one candidate
        |
        v
committed harness H_{t+1}, carried to next batch
```

算法里的三个操作：

| 操作 | 输入 | 输出 | 作用 |
|---|---|---|---|
| `OBSERVE` | candidate harness + unlabeled batch | traces, cached outputs, proxy signals | 让候选真实跑一遍，收集行为证据 |
| `PROPOSE` | active branches + traces | edited child harnesses | 让 agentic coding proposer 根据失败模式改代码 |
| `JUDGE` | final branches + traces + inputs | committed harness | 选择一个候选持久化 |

这里有两个设计点值得记：

1. **Lineage**：每条分支继承自己的 parent，因此一个好规则可以在后续 rounds 里继续叠加。
2. **Diversity**：不同 proposer 被引导成不同角色，例如 conservative-repair、exploratory、adversarial，避免所有候选太像。

但 judge 只在最后一轮候选里选，而不是从所有历史候选里选。这会导致 selection regret：历史上出现过的好候选可能在最后没留下。

## 4. 优化信号：不是 reward，而是 execution evidence

TTHE 不假设有可信 scalar reward。每次运行 harness 会留下：

- prompts and completions；
- tool calls and outputs；
- stdout / stderr；
- intermediate artifacts；
- runtime states；
- errors；
- extra probes。

此外作者设计了少量任务相关 proxy：

| Proxy | 含义 | 局限 |
|---|---|---|
| execution health $s \in \{0,1\}$ | 输出能否执行、格式是否良好、shape 是否一致 | SQL 执行成功不等于回答正确 |
| round-trip consistency $b \in [0,1]$ | 把输出反推成描述，再和输入匹配 | 只能发现明显语义漂移 |
| public-test pass rate $p \in [0,1]$ | 公开测试或 repo test 通过率 | public pass 可以过拟合，hidden 仍失败 |

作者明确不把这些 proxy collapse 成一个 reward，而是把 raw trace 和 proxy 一起给 proposer / judge。这个选择很对，因为 agent harness 的可靠性往往要读“怎么错的”，不是只看一个分数。

对机器人尤其如此：一次抓取失败不是一个 scalar 能解释完的，必须知道是定位错、预接触姿态错、夹爪闭合无物、碰撞、目标被遮挡、还是释放后滑落。

## 5. 实验设置

作者在五类 execution-grounded 任务上评估：

| 领域 | Benchmark / slice | 样本数 | 评价 |
|---|---:|---:|---|
| Text-to-SQL | BIRD hard slice | 50 | SQL 执行结果与 gold result set 等价 |
| Competitive programming | LiveCodeBench hard slice | 60 | hidden tests pass |
| Software engineering | SWE-bench Verified hard slice | 40 | hidden tests pass |
| Data-science coding | DS-1000 hard slice | 50 | hidden tests pass / correctness |
| Agentic tool-use | claw-eval headroom slice | 30 | graded score in [0,1] |

主实验使用 deepseek-v4-flash 作为 solver/proposer/judge 的底层模型。SWE-bench baseline 是 mini-swe-agent，TTHE 对这个 scaffold 做演化。

注意两个协议细节：

- 这些 hard slices 很多是为了暴露 baseline 的 harness failure 而选，不应该把绝对分数当通用 benchmark 排名。
- 主结果是 **transductive scoring**：batch $X_t$ 先用于无标签适应，然后用被选中的 harness 在同一 batch 的 cached outputs 上测分。它证明的是 within-batch adaptation，不等于 prequential forward generalization。

## 6. 主要结果

主结果很强，但要按协议谨慎读。

| 任务 | Baseline | TTHE | 提升 |
|---|---:|---:|---:|
| BIRD hard slice | 12.0 | 50.0 | +38.0 |
| LiveCodeBench | 30.0 | 38.3 | +8.3 |
| SWE-bench Verified | 20.0 | 35.0 | +15.0 |
| DS-1000 | 38.0 | 44.0 | +6.0 |
| claw-eval score | 48.9 | 69.8 | +20.9 |

BIRD 的 +38 很亮眼，但因为 slice 是从 baseline 反复失败的问题中筛出来的，所以这不是说 TTHE 在普通 BIRD 上必然有同等幅度，而是说：在 harness failure 密集的 hard slice 上，测试时程序改写很有效。

claw-eval 的结果也有意思：30 个 agentic office workflow 中，TTHE 改善了 28 个，平均 score 从 48.9% 到 69.8%。这说明它不只适合代码生成，也适合多工具 workflow。作者观察到它自动学到：

- tool payload 必须单行，避免写入动作没被记录；
- 最终回答前要引用每个 retrieved record identifier；
- 跨服务串联要用 exact ID，而不是 display name；
- 不能声称自己做了没有实际执行的动作。

这些都是典型 harness policy，不是模型知识本身。

## 7. 演化出来的 harness policy

论文最值得借鉴的是它展示了“程序级行为”的形状。

### Text-to-SQL

演化后的 harness 会：

- 用数据库 probe ground filter values，修正大小写；
- 推断问题要求的 output shape；
- 检查是否返回多余列；
- 遇到 SQL error、empty rows、wrong columns 时把 targeted error message 回灌给模型；
- 最后做 shape fix。

这类改动不是一句“请认真检查”能稳定替代的，它需要代码层面的循环和验证器。

### SWE-bench

演化后的 harness 学到：

- reproduce first；
- 先定位 root cause，再编辑；
- 最小补丁；
- rerun reproduction；
- run existing tests；
- 如果 first rollout 返回 empty patch，就进入 anti-paralysis branch，用更具体的起手命令打破卡住状态。

这对机器人也很像：如果 VLA 或 planner 没有产生可执行动作，不能只继续 prompt 它，应该有专门的 recovery branch，把系统带回可行动状态。

### LiveCodeBench

演化出的策略是 conditional verification-and-repair：

- 默认走低成本路径；
- public tests 失败后再 escalation；
- 根据 observed vs expected 做 targeted repair。

这跟机器人里的自适应算力分配相通：平常别把 TTC 开满，只有遇到高不确定性、验证失败或物理状态异常时再升级。

## 8. 消融与诊断

### 搜索预算不是越大越好

BIRD hard slice，baseline 是 12.0：

|  | G=1 | G=3 |
|---|---:|---:|
| R=1 | 46.0 | 44.0 |
| R=3 | 40.0 | 50.0 |

解释：

- 单分支多轮可能把早期好候选改坏，而且最后 judge 没得选；
- 多分支多轮更容易保留多样候选；
- 但 G=3/R=3 同时改变了 proposer 数量和角色 prompt，不能把收益单独归因于分支数。

### Batch size 有甜点

在 G=3/R=3 下：

- B=5：44.0
- B=10：50.0
- B=25：38.0
- B=50：44.0

B 太小，每次 commit 证据不足；B 太大，适应次数太少。B=10 在这个 slice 上平衡较好。

### 跨模型有效

| Backbone | Base | TTHE | 提升 |
|---|---:|---:|---:|
| DeepSeek V4 Flash | 12.0 | 50.0 | +38.0 |
| MiMo V2.5 | 32.0 | 52.0 | +20.0 |
| Kimi K2.5 | 28.0 | 48.0 | +20.0 |

作者提醒：slice 是按 DeepSeek 失败构造，所以绝对分数不是模型排名。关键是每个模型相对自己都有提升。

### Carry harness across batches 有用

BIRD, B=10/G=3/R=3：

| 设置 | B1 | B2 | B3 | B4 | B5 | Overall |
|---|---:|---:|---:|---:|---:|---:|
| Accumulate | 30 | 70 | 40 | 60 | 50 | 50.0 |
| Reset each batch | 30 | 40 | 40 | 50 | 60 | 44.0 |

累计提交的 harness 比每批重置高 6 分，主要来自中间 batch。说明 warm-start structure 有价值，但由于协议是 transductive，还不能完全排除 batch-local specialization。

## 9. 失败来源：coverage gap 与 selection regret

作者做了 post-hoc oracle 诊断：

- judge 实际提交：50.0%
- oracle 只在 judge 看到的 final candidates 中选：64.0%
- oracle 在所有曾生成 candidates 中选：70.0%

这分成两个问题：

1. **Selection regret**：有些 candidate 已经解出来了，但 judge 没选中。final-pool oracle 比 judge 高 14 分。
2. **Coverage gap**：有些任务没有任何候选解出来。all-candidates oracle 也只有 70 分，说明搜索生成能力本身不足。

最扎心的一点是：judge 会自信地选择执行看起来健康但 benchmark-incompatible 的输出，比如 SQL 返回了额外列或粒度更粗。也就是说 proxy 可以骗过 judge。

对 Harness VLA，这对应两个风险：

- 机器人执行日志可能显示“动作执行完成”，但任务语义没完成；
- VLM judge 可能觉得画面“差不多了”，但真实 predicate 或人类标准不通过。

所以机器人 harness evolution 不能只靠视觉语言 judge，必须把几何、接触、状态估计、任务 predicate 和失败日志一起接入。

## 10. 局限

这篇证据强，但边界也明确：

- 主协议是 transductive，不是 prequential；还缺“batch t 学到的 harness 在 batch t+1 未适应前是否变强”的严肃测试。
- 缺 multi-seed 和 compute-matched baselines，例如 best-of-N execution selection。
- judge 是瓶颈，并且 proxy 不完美。
- hard slice 的选择会放大 harness 改进空间，绝对分数不能外推。
- proposer 能改 arbitrary code，真实部署需要强 sandbox 和安全边界。
- 现在的领域以代码、SQL、工具调用为主；物理机器人会多出不可逆状态变化、安全约束和昂贵试错。

## 11. 对 Harness VLA 的启发

### 启发 1：Harness VLA 不应该只是固定 scaffold，而应该能演化

现有 [[2026-Harness-VLA-Deep-Read]] 的核心是把 frozen VLA 封装成 `VLA_ACT`，再用 LLM/VLM agent 调度 analytic primitives、memory 和 retry。TTHE 进一步指出：这套调度程序本身可以是测试时适应对象。

可迁移设计：

```text
robot execution batch
-> collect traces: images, depth, poses, primitive calls, success predicates, failures
-> proposer edits harness policy
-> judge chooses safe candidate policy
-> committed harness governs later tasks
```

这里的“edit harness policy”不一定一开始就让 agent 改 Python 源码。为了机器人安全，可以先限制成 DSL / config：

- 哪些阶段调用 `VLA_ACT`；
- 哪些阶段调用 analytic motion；
- 何时 retry；
- retry 前是否 re-localize；
- 何时 ask world model / VLM judge；
- failure memory 写什么；
- primitive stop condition 怎么调。

等 DSL 稳了，再考虑更开放的代码级 evolution。

### 启发 2：机器人也需要 execution-derived proxy，但 proxy 必须多模态

TTHE 的 proxy 是 execution health、round-trip consistency、public-test pass。机器人里可以设计对应版本：

| TTHE proxy | Harness VLA proxy |
|---|---|
| execution health | primitive 是否完成、IK 是否可解、碰撞检查是否通过、动作是否超时 |
| round-trip consistency | VLM 从当前图像反推任务状态，是否和目标/子任务一致 |
| public-test pass | simulator predicate / task checker / affordance verifier / contact sensor |
| output shape check | 子任务是否只改变应改变的状态，是否误动其他物体 |
| hidden-test gap | 视觉看似成功但真实物理 predicate 失败 |

最关键的是：机器人 proxy 不能只用 RGB VLM。应该组合：

- VLM task progress；
- object pose tracker；
- gripper force/contact；
- state predicate；
- collision / reachability；
- before-after image difference；
- low-level controller return code。

### 启发 3：把 memory 从“经验库”升级成“可提交的 harness diff”

Harness VLA 当前的 memory 更像任务级程序骨架和全局经验。TTHE 提醒我们，memory 可以进一步结构化为可执行规则：

```yaml
rule:
  trigger: "gripper closed but target pose did not move with end-effector"
  diagnosis: "empty grasp"
  patch:
    - re_localize_target
    - move_to_pregrasp_from_new_pose
    - call_vla_act_with_shorter_prompt
  verification:
    - target_pose_attached_to_gripper
    - gripper_width_below_threshold
```

这样 memory 不只是“下次注意”，而是能直接改变 harness 的控制流。

### 启发 4：引入 population，而不是单一路径 retry

Harness VLA 里常见失败恢复是：

```text
fail -> reflect -> retry
```

TTHE 建议更强的形式：

```text
fail batch
-> candidate A: conservative fix, only re-localize
-> candidate B: exploratory fix, change primitive boundary
-> candidate C: adversarial fix, add stricter verifier
-> evaluate proxy traces
-> commit one policy
```

机器人里不一定要对同一个真实场景反复执行所有候选，因为代价和安全风险太高。可以混合：

- simulation replay；
- world-model rollout；
- logged trajectory counterfactual check；
- low-risk physical probe；
- only then commit。

这和 [[2026-Tau0-VLA]] 的 world-model-guided TTC 可以合体：world model 不只评估下一条 subtask，也评估候选 harness policy 的后果。

### 启发 5：必须显式防 selection regret

TTHE 最大可恢复损失来自 judge 没选中好 candidate。机器人上这会更严重，因为一次错误 commit 可能破坏环境。

建议 Harness VLA 加：

- candidate archive，不只从最后一轮选；
- conservative rollback；
- safety-first tie breaker；
- human-inspectable diff；
- predicate-based veto；
- canary task / shadow mode；
- prequential evaluation：在新任务上先 shadow-score，再正式接管。

### 启发 6：把 Harness VLA 的研究问题改写得更锋利

TTHE 给 Harness VLA 一个很漂亮的问题表述：

> Can a robot harness adapt at test time by editing its own executable control policy from unlabeled physical execution traces, while keeping the VLA frozen and using no task gold labels?

中文就是：

> 冻结 VLA 后，能否仅根据机器人自己的未标注执行轨迹，让外部 harness 在测试时持续改进 primitive 调度、验证和恢复策略？

这比“给 VLA 加 memory”更像一个独立论文问题。

## 12. 我对这篇的判断

我会给 8.5/10。

优点：

- 问题定义很准，把 test-time adaptation 的对象从 weights / prompts 推到 executable harness；
- 结果跨 SQL、代码、SWE、工具调用，说明不是单 benchmark 技巧；
- 诚实分析了 transductive 协议、selection regret 和 coverage gap；
- 对 agentic robotics 很有迁移价值，因为机器人本来就是强 harness 系统。

主要扣分：

- 还没有 prequential 主结果；
- proxy 和 judge 的可靠性仍是核心未解；
- hard slices 会放大收益；
- 没有充分 compute-matched best-of-N / dev-time optimizer 对比；
- arbitrary code evolution 距离真实部署还有安全工程鸿沟。

我的一句结论：

> TTHE 不是“LLM agent 自我进化已经解决了”，而是把自我进化最实际的落点找准了：先别动大模型，先让它外面的执行程序学会从失败日志里改自己。

