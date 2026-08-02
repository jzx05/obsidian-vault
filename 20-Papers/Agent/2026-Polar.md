---
title: "Polar: Agentic RL on Any Harness at Scale"
authors:
  - Binfeng Xu
  - Hao Zhang
  - Shaokun Zhang
  - Songyang Han
  - Mingjie Liu
  - Jian Hu
  - Shizhe Diao
  - Zhenghui Jin
  - Yunheng Zou
  - Michael Demoret
  - Jan Kautz
  - Yi Dong
year: 2026
venue: arXiv preprint
arxiv: 2605.24220v1
tags:
  - paper
  - agent
  - agentic-rl
  - rl-infrastructure
  - rollout-as-a-service
  - agent-harness
  - trajectory-reconstruction
  - software-engineering-agent
status: read
rating: 8.5/10
date-read: 2026-08-02
url: https://arxiv.org/abs/2605.24220
code: https://github.com/NVIDIA-NeMo/ProRL-Agent-Server
pdf: "C:/Users/huawei/Zotero/storage/IVLSQ5FT/Xu et al. - 2026 - Polar Agentic RL on Any Harness at Scale.pdf"
related:
  - "[[2026-OpenClaw-RL]]"
  - "[[2026-SkillClaw]]"
---

# Polar：不打开 Agent Harness，也能把它接入大规模 RL

## TL;DR

Polar 解决的是一个 Agentic RL 的基础设施难题：现有 Codex、Claude Code、Qwen Code、Pi 等 Agent harness 各有自己的 prompt、上下文压缩、工具协议、subagent、文件操作和终止逻辑。传统做法要求把它们重写成 RL 框架规定的 `env.reset()/env.step()`，工作量大，而且重写后训练的可能不再是产品实际运行的那条 execution path。

Polar 的核心观察非常直接：

> 无论 Agent 内部怎样实现，只要它由 LLM 驱动，就一定要调用模型 API。模型 API 是所有 harness 共同拥有、且位于 harness 外部的最小训练边界。

所以 Polar 不进入 harness 内部，而是在 harness 与 inference server 之间放置一个 provider-compatible proxy：

```text
unchanged agent harness
  -> OpenAI / Anthropic / Google-style model request
  -> Polar API proxy
  -> local inference server, e.g. SGLang

Polar proxy simultaneously records:
  prompt token IDs
  sampled response token IDs
  behavior-policy log probabilities
  messages / tools / finish reason
  session provenance
```

Harness 仍按原来的方式规划、调用工具、压缩上下文、启动 subagent；Polar 只“监听”模型调用，并在运行结束后把多个 API request 重建成 token-faithful RL trajectory，再把轨迹和 reward 交给任意 trainer。

Polar 的三项关键设计是：

1. **Black-box harness integration**：用模型 API 代理替代 framework-specific environment 重写；
2. **Rollout as a service**：rollout server 与 gateway nodes 独立于 trainer，异步处理 session；
3. **Token-faithful prefix merging**：只把 rollout 时模型真正采样的 token 设为可训练，harness 插入的上下文全部 mask，同时将属于同一 append-only 对话链的 requests 合并。

一句话概括：

> Polar 把模型 API 流量变成 Agentic RL 的“网络抓包层”，既保留原生 harness 行为，又恢复训练所需的 token、logprob、loss mask、reward 和 provenance。

论文在 SWE-Gym 上用 Qwen3.5-4B + GRPO 训练四种原生 coding harness。在完整 SWE-Bench Verified 上：Codex 从 3.8% 提升至 26.4%，Claude Code 从 29.8% 至 34.6%，Qwen Code 从 34.6% 至 35.2%，Pi 从 34.2% 至 40.4%。最大提升出现在 Qwen 原本最不熟悉的 Codex execution protocol。

我的判断：这篇论文的系统贡献很扎实，尤其是把“任意 Agent 接 RL”的接口选择从 Agent SDK 内部移到了 provider API boundary，并认真解决了 retokenization、context compaction 和 subagent 分叉后的轨迹正确性。但标题里的 “Any Harness” 目前主要是架构能力：实验只覆盖软件工程、四种 coding harness、Qwen3.5 和 outcome reward；对 GUI、Web、OS、multi-agent、闭源远程模型与 process reward 的通用性仍待验证。

## 1. Polar 到底要解决什么问题？

### 1.1 Agentic RL 的训练对象已经不是一个简单函数

传统 RL 习惯把环境写成统一接口：

```python
obs = env.reset()
obs, reward, done, info = env.step(action)
```

但真实 Agent harness 是一个复杂软件系统，可能包含：

- 自己的 system prompt 与 context policy；
- OpenAI、Anthropic、Google 等不同 API schema；
- bash、read、edit、write、browser、MCP 等工具；
- 多轮执行和数万 token 上下文；
- context compaction、injection、replacement；
- subagent 或并行分支；
- repo/container/OS 环境；
- cron、hook、patch submission 与自定义停止条件；
- CLI、SDK、甚至闭源 binary。

如果为了 RL 把 harness 逻辑重新实现一遍，会产生两类问题：

1. **集成成本**：每个新 harness 都要写一套 framework-specific environment；
2. **execution-path drift**：训练时的重实现与真正产品 harness 不完全相同，reward 优化的不是部署时的行为。

Polar 提问：

> Can we train agents with RL without opening the box?

答案是把 observation boundary 放在模型 API，而不是 Agent 内部的 event loop。

### 1.2 为什么模型 API 是合适的公共边界？

不同 harness 内部完全不同，但都必须完成：

$$
\text{context}\rightarrow\text{model request}
\rightarrow\text{sampled tokens}\rightarrow\text{agent action}.
$$

只要能够在 request/response 边界捕获：

- 原始 prompt token IDs；
- 模型实际生成的 response token IDs；
- rollout policy 的 log probabilities；
- message/tool schema；
- session 与 task 归属；

就可以在 harness 外部构造 RL 训练数据，而无需理解 Agent 为什么发起这次调用。

## 2. Polar 的总体架构

Polar 由两个核心部分组成：

```text
Trainer
  | submit tasks / receive callbacks
  v
Rollout Server
  | task expansion, session registry, load balancing,
  | status persistence, health tracking
  v
Gateway Nodes
  | INIT -> READY -> RUNNING -> POSTRUN
  | isolated runtime + native harness
  | provider-compatible model proxy
  | trajectory builder + evaluator
  v
Inference Server
```

### 2.1 Rollout server

Rollout server 接受一个 `TaskRequest`，把它展开为 `num_samples` 个独立 session。每个 session 是调度单位，包含：

- `session_id` 与 `task_id`；
- timeout budget；
- runtime specification；
- agent/harness specification；
- trajectory builder；
- evaluator；
- trainer callback URL。

Server 负责分发 session、持久化紧凑的终态结果、提供 polling 状态，并接收 gateway 完成后的 callback。它不负责具体跑 Agent。

### 2.2 Gateway node

Gateway 拥有一次 session 的完整生命周期：

1. 启动 runtime；
2. 准备 repo、依赖和 harness 配置；
3. 运行原生 harness；
4. 在本机 proxy 捕获所有模型调用；
5. 从 completions 重建 trajectories；
6. 在 fresh runtime 中执行 evaluator；
7. 回调 trainer；
8. 清理资源。

Proxy 与 session registry 共置，能确保捕获的 completion 明确属于哪次 session，不需要另建分布式 tracing service。

### 2.3 Trainer 与 Polar 解耦

Polar 不是新的 optimizer，也不替代 Slime、PRIME-RL 等训练系统。它提供的是 rollout substrate：trainer 可以异步提交任务，等待达到 batch size 的已评价 trajectory groups 后再更新策略。

因此三部分职责清晰：

| 组件 | 负责什么 | 不负责什么 |
|---|---|---|
| Harness | planning、context、tools、subagent、stop | 不关心 RL 数据格式 |
| Polar | native execution、capture、reconstruction、evaluation | 不规定 RL algorithm |
| Trainer | GRPO/PPO/SFT、policy version、optimization | 不重写 harness |

## 3. Proxy 如何抓取模型调用？

Harness 只需通过环境变量或配置文件，把 model base URL 指向 gateway proxy。每个请求经历四步。

### 3.1 Detect provider API

根据 request path 与 headers 判断：

- Anthropic Messages；
- OpenAI Chat Completions；
- OpenAI Responses；
- Google `generateContent`。

### 3.2 Normalize request

Provider transformer 将不同协议的 roles、content parts、tool definitions、tool choice、stop controls 和 generation parameters 统一成 local inference server 使用的 OpenAI Chat Completions shape，并强制请求训练需要的字段，例如 `logprobs=true`。

### 3.3 Capture token-level data

Proxy 转发请求，并保存 completion record：

- request/response messages；
- prompt token IDs；
- sampled response token IDs；
- response log probabilities；
- finish reason；
- tool schema 与其他元数据。

### 3.4 Return original provider shape

Response 再转换回 harness 期待的 provider schema。对于 streaming，Polar 当前向上游获取 non-streaming response，再合成 provider-shaped server-sent stream。

这样简化了完整 token 捕获，但也意味着它并不严格保留真实流式生成的逐 token 时间、提前消费、客户端中途取消等语义。对于依赖流式 side effect 的 harness，这可能是兼容性边界。

## 4. Harness adapter 为什么仍然需要？

“Black box” 不等于零配置。Polar 不需要重写 Agent event loop，但仍要有一个很小的 adapter，用于：

- 安装/准备 harness；
- 写 provider settings；
- 注册 MCP servers 或 skills；
- 设置 model base URL；
- 返回启动 Agent 的 shell commands。

通用 shell command harness 可以包装任意可执行程序；内置 shortcut 包括：

- `claude_code`；
- `codex`；
- `gemini_cli`；
- `qwen_code`；
- `opencode`；
- `pi`。

因此“无需修改 harness”是准确的，“无需任何集成工作”则不准确。仍需保证 provider compatibility、runtime 安装、认证替换和结果提取。

## 5. Runtime 与异步 staging

### 5.1 Runtime interface

Runtime 提供统一的：

```text
start / stop / exec / upload / download / cancel
```

首版支持 Docker 与适合 HPC 的 rootless Apptainer。Gateway 只依赖这层接口，因此同一个 task 可以切换 isolation backend。

### 5.2 为什么 long-horizon rollout 需要阶段隔离？

一次 SWE Agent rollout 混合了不同类型的成本：

- container startup；
- dependency preparation；
- Agent/model GPU execution；
- evaluator environment setup；
- tests、patch application；
- trajectory reconstruction；
- teardown。

如果串行执行，CPU-bound 的初始化和测试会让昂贵的 rollout GPU 空闲。Polar 在每个 gateway 内设置：

```text
INIT workers
  -> READY bounded buffer
  -> RUNNING workers
  -> POSTRUN workers
```

- `INIT`：启动 runtime、执行 prepare actions；
- `READY`：保留已初始化 runtime，等待 run slot；
- `RUNNING`：执行 harness；
- `POSTRUN`：重建轨迹、评价、hook、callback、清理。

Evaluator 如果需要 clean runtime，会在 Agent 仍运行时预热。Session 共用一个 deadline；即使 harness timeout，只要之前已捕获模型调用，仍进入 post-run，恢复 partial traces 并标注 timeout terminal status。

## 6. 最关键的方法：为什么不能直接把所有 API response 拼起来？

RL 更新需要保证：用于 loss 的 token 真的是 rollout policy 当时采样出来的 token，而且对应正确的 behavior logprob。

真实 harness 会在两次模型调用之间做很多事：

- 把上一次 assistant response 重新序列化进下一次 prompt；
- 插入 tool output；
- 改写 system prompt；
- compact 历史；
- 产生 subagent；
- 并行分支；
- 使用不同 provider 的 canonical rendering。

如果把下一次 prompt 重新 tokenize，再把里面的历史 assistant 文本当作原生成 token，可能发生 **retokenization drift**：同一文本在新的模板、边界或 tokenizer context 下对应不同 token 序列。这会让 token IDs 与 rollout logprob 错位，破坏 on-policy/importance-ratio 假设。

## 7. 两种 trajectory reconstruction 策略

### 7.1 `per_request`：保守但碎片化

每个 model completion 单独形成一条 trace。

优点：

- 对单次调用无损；
- 不需要推断 requests 是否属于同一 conversation chain；
- context compaction 和 subagent 不会被错误拼接。

缺点：

- 一次 coding session 可能产生数百个短 traces；
- trainer-facing updates 数量爆炸；
- session-level outcome reward 难以合理分配给每个 request；
- 将终局成功广播给所有短 trace 容易 reward hacking。

### 7.2 `prefix_merging`：只合并严格 append-only 的链

对 completion $C_i$，记录：

- prompt token sequence $p_i$；
- raw sampled response tokens $a_i$；
- response logprobs $\ell_i$；
- structured messages $m_i$。

Polar 先把 completions 分成若干有序 chains：

$$
\mathcal{G}=\{G_1,\ldots,G_J\}.
$$

一个新 completion 只有同时满足两项条件才能接到已有 chain：

1. normalized message-level grouping key 表明它可能是同一对话的延续；
2. 新 prompt 对旧 prompt 满足严格 token prefix：

$$
p_{m+1}[1:|p_m|]=p_m.
$$

因此以下情况不会被强行拼进一条 global trace：

- subagent；
- parallel branches；
- context compaction；
- prompt rewrite；
- 独立 tool-mediated conversation。

它们自然形成新 chain。

## 8. Token-faithful merging 的细节

即使 $p_{m+1}$ 延续 $p_m$，它的新增尾部仍包含两类内容：

1. 上一个 assistant turn 的 canonical server rendering；
2. harness 插入的 tool result 或其他 interstitial context。

关键是不能从 canonical rendering 复制上一轮 assistant body，因为真正由 behavior policy 采样的是原始 $a_m$。

令 $e$ 为 end-of-turn token，定义新 prompt 相对旧 prompt 的 canonical tail：

$$
t_m=p_{m+1}[|p_m|+1:].
$$

Polar 在 $t_m$ 中找到第一个 $e$，由此分离 interstitial $u_m$。整条 chain 被重建为：

$$
z^{(j)}=p_1\;||\;a_1\;||\;u_1\;||\;a_2
\;||\;u_2\;||\cdots||\;a_K.
$$

Loss mask 为：

$$
M(x)=
\begin{cases}
1, & x\in a_m\text{，即模型真实采样 token},\\
0, & x\in u_m\text{，即 harness/canonical 插入 token}.
\end{cases}
$$

- $a_m$ 使用 rollout 时捕获的真实 logprob；
- $u_m$ 填 synthetic logprob 只为保持数组对齐；
- 是否训练由 `loss_mask` 决定。

Polar 给出的正确性 invariant 是：

> Every trainable token matches the behavior policy during rollout, and any non-generated tokens are masked out.

这才是 prefix merging 的实质。它不是普通的字符串去重，而是保护 RL likelihood ratio 正确性的 token-level provenance construction。

## 9. Reward 如何传回轨迹？

Evaluator 在 trajectory construction 后运行，可以访问：

- trajectory；
- session artifacts；
- 可选的 fresh runtime context。

内置 evaluator 包括：

- session completion reward；
- test-on-output；
- SWE-Bench/SWE-Gym evaluator。

Session-level outcome reward 可以广播到 traces；process reward 则需要 per-trace assignment。

论文报告：对 `per_request` traces 把终局 outcome reward 广播给每个 request 时，出现明显 reward hacking。原因是一个成功 session 中的无用甚至错误中间调用也获得正奖励。作者没有解决这一 credit assignment 问题，只把 session normalization 与 PRM-style process reward 列为 roadmap。

因此 Polar 解决的是“正确捕获和组织轨迹”，不是长时程 credit assignment 本身。

## 10. 在线 RL 实验

### 10.1 设置

- Base checkpoint：Qwen3.5-4B；
- Training data：`NovaSky-AI/SkyRL-v0-293-data` train split，293 tasks；
- Trainer：Slime asynchronous GRPO；
- Epochs：1；
- Rollout batch size：4；
- Samples per prompt：16；
- Trace builder：`prefix_merging`；
- Learning rate：$10^{-6}$；
- Weight decay：0.1；
- TIS：enabled；
- Evaluation：完整 SWE-Bench Verified，使用对应原生 harness，pass@1。

四组实验都从同一个 base checkpoint 开始，但分别在对应 harness 生成的轨迹上训练，并用同一个 harness 评测。这测的是 **harness-native adaptation**，不是一个统一 checkpoint 同时适配四种 harness。

### 10.2 SWE-Bench Verified 结果

| Harness | Base | Polar RL | Absolute gain |
|---|---:|---:|---:|
| Codex | 3.8% | **26.4%** | **+22.6** |
| Claude Code | 29.8% | **34.6%** | +4.8 |
| Qwen Code | 34.6% | **35.2%** | +0.6 |
| Pi | 34.2% | **40.4%** | +6.2 |

### 10.3 如何解释差异？

Codex 增益最大，不代表 Codex harness 最强，而是 Qwen3.5-4B 初始时最不适应 Codex 的 tool schema、context policy 和 patch submission style。Base 只有 3.8%，存在巨大的 harness mismatch；在真正 Codex execution path 上训练后升到 26.4%。

Qwen Code 是更原生的执行路径，base 已有 34.6%，只提升 0.6。这里存在明显 ceiling/alignment effect：Polar 的价值在 unfamiliar harness 上更突出。

训练集 rollout reward 的首末 10 steps：

| Harness | First 10 | Last 10 |
|---|---:|---:|
| Codex | 9.5% | 54.5% |
| Claude Code | 28.8% | 67.0% |
| Qwen Code | 61.6% | 66.0% |
| Pi | 61.6% | 76.2% |

这些是 SWE-Gym training reward，不应与 SWE-Bench Verified 的 held-out pass@1 混淆。训练曲线涨幅远大于 benchmark 增益，说明仍有训练分布适配或泛化差距。

## 11. Prefix merging 的效率消融

在相同模型、硬件、topology 和三步训练窗口下：

| 指标 | `per_request` | `prefix_merging` |
|---|---:|---:|
| Trainer-facing updates | 1,185 | 218 |
| Wall-clock | 189.5 min | 35.2 min |
| Rollout GPU utilization | 20.4% | 87.7% |
| Speedup | 1x | 5.39x |

为什么合并后更快？不是模型少生成了 token，而是 trainer 不再把一次长 session 的数百个 requests 当成数百个独立 update 单位。更少、更长、更连贯的 trace 减少了 trainer 调度与等待碎片，也让 rollout 与 training 更好地流水化。

不过这只是一个 partial utilization profile：三步训练窗口、特定 workload 与 topology。论文没有给多集群规模、gateway 数量、failure rate、P50/P95 latency 或不同 session length 下的系统曲线，所以 5.39x 不能视为普遍加速常数。

## 12. Offline SFT 数据生成

Polar 也能固定 checkpoint，不做在线更新，仅作为分布式 Agent trajectory generator：

```text
fixed teacher checkpoint
  -> fan out native harness sessions
  -> isolated runtime execution
  -> journal all trajectories
  -> run binary verifier
  -> retain successful traces for SFT
```

实验设置：

- Teacher：Qwen3.5-122B-A10B；
- Serving：单个 `8 x H100` SGLang job，TP=8；
- Harness：`pi-coding-agent v0.67.68`；
- 1,638 SWE-Gym instances，来自 7 个 repositories；
- Apptainer SIF，fresh target commit；
- `max_concurrent=5-8`；
- timeout 3,600 seconds；
- `empty_generation` retry once；
- verifier 同时要求 `FAIL_TO_PASS` 与 `PASS_TO_PASS` tests 通过。

结果：

| Repo | Attempts | Accepted | Rate |
|---|---:|---:|---:|
| getmoto/moto | 343 | 184 | 53.6% |
| python/mypy | 257 | 101 | 39.3% |
| conan-io/conan | 71 | 27 | 38.0% |
| pydantic/pydantic | 81 | 24 | 29.6% |
| iterative/dvc | 219 | 45 | 20.5% |
| pandas-dev/pandas | 477 | 98 | 19.7% |
| dask/dask | 141 | 25 | 17.7% |
| **Total** | **1,638** | **504** | **30.8%** |

交互部分约花费 64 GPU-hours。Accepted trajectories 平均 104 messages、51 assistant turns，long tail 超过 200 turns。

这个实验说明 Polar 的轨迹格式不仅可用于 RL，也可产出 SFT、preference 或 verifier training 数据。但 30.8% acceptance 也意味着约七成昂贵 rollout 未进入正向 SFT corpus；它们仍可作为 verifier 或 preference negatives，论文只是没有进一步利用。

## 13. 论文真正证明了什么？

### 证据较强支持

- 模型 API proxy 是一种低侵入、实际可行的 native harness RL 边界；
- OpenAI/Anthropic/Google-style 请求可归一化到本地 inference backend，再恢复 provider shape；
- Codex、Claude Code、Qwen Code、Pi 在不重写内部 event loop 的情况下可产生可训练轨迹；
- raw sampled token IDs 与 logprobs 可以避免 retokenization drift；
- strict prefix chain + loss mask 能处理正常追加上下文，同时隔离 compaction 和 subagent branches；
- stage-isolated gateway 能把 runtime preparation、execution 和 evaluation 流水化；
- harness-native GRPO 可以让模型适应陌生的 tool/action protocol；
- 同一 rollout service 可复用于在线 RL 与离线 SFT 数据生成。

### 尚未充分证明

- **Any harness**：实验只有 coding harness，不含 browser、GUI、OS control、robotics 或真正复杂 multi-agent；
- **任意训练框架**：主要端到端集成是 Slime，trainer agnosticism 更多来自 API 设计；
- **任意 RL algorithm**：只展示简单 GRPO 和 offline SFT；
- **大规模扩展曲线**：没有 gateway/node/GPU 数量随吞吐的 scaling law；
- **闭源 binary 完全兼容**：只要能改 base URL 理论上可行，但认证、TLS pinning、streaming 和 proprietary protocol 可能阻碍；
- **过程奖励正确分配**：per-request outcome broadcast 已观察到 reward hacking；
- **多 harness 单模型泛化**：四个 harness 是分别训练，没有联合训练或 cross-harness transfer matrix；
- **稳定性和统计显著性**：未报告多 seed、error bars、run-to-run variance；
- **安全性**：proxy 能看到完整 prompt、源码、tool arguments 和 secrets，论文未系统讨论数据保护。

## 14. 关键局限与风险

### 14.1 API traffic 不等于全部 Agent state

Proxy 能看到模型调用，却看不到未进入 prompt 的 harness internal state，例如：

- scheduler queue；
- hidden cache；
- deterministic tool logic；
- local retry counter；
- UI side effects；
- 并发 race；
- harness 自己的 safety gate。

因此它能重建 model-facing trajectory，不一定能完整重建环境的 Markov state 或 Agent 控制流。

### 14.2 Provider normalization 可能改变语义

不同 provider 对：

- tool-call ordering；
- reasoning fields；
- stop tokens；
- parallel tool calls；
- JSON schema；
- caching；
- streaming cancellation；

存在细微差异。转换成 OpenAI Chat shape 再转回，必须通过严格 compatibility tests 才能声称 harness 完全 unchanged in behavior。

### 14.3 Prefix merging 依赖严格 prefix，覆盖率可能有限

只要 harness 每轮轻微改写 system prompt、重新排序 tool schema、加入时间戳或进行 compaction，就会断链，退化为多条 traces。论文展示了正确性和一次效率消融，但没有报告不同 harness 的 merge rate、平均 chains/session 或断链原因。

### 14.4 Outcome reward 仍然稀疏

Polar 正确记录了 token，不代表知道哪个 token 或哪次 tool call导致最终成功。长时程 Agentic RL 仍需要：

- process reward model；
- step-wise advantage；
- return decomposition；
- causal credit assignment；
- session-level normalization。

这与 [[2026-OpenClaw-RL]] 的互补关系很清楚：Polar 解决“怎样从任意 harness 获取可靠轨迹”，OpenClaw-RL 关注“怎样从 next state 提取 evaluative/directive signal”。

### 14.5 Proxy 是高敏感度控制面

它能读取：

- 用户 prompt；
- 私有代码；
- tool outputs；
- system prompt；
- MCP definitions；
- 可能包含的 credentials 与内部 URL。

生产环境必须具备 tenant isolation、TLS、secret redaction、访问审计、最短保留期、加密存储和训练数据 consent。否则为训练而加的 proxy 会成为新的数据泄漏面。

### 14.6 Async RL 的 policy staleness

Rollout 与 trainer 解耦后，某条 session 可能由较旧 policy 生成。论文的 framework comparison 把 explicit policy-version/staleness handling 作为 async RL 条件，训练设置启用 TIS，但没有深入报告 staleness distribution、最大 lag 或它对最终性能的影响。

## 15. 与 OpenClaw-RL、SkillClaw 的关系

三篇论文分别解决 Agent 持续改进链条的不同层：

| 论文 | 主要问题 | 更新对象 | 关键接口 |
|---|---|---|---|
| Polar | 如何从任意原生 harness 得到可训练轨迹 | 不规定 | 模型 API proxy |
| [[2026-OpenClaw-RL]] | 如何把 next state 变成 reward 与 token hint | 模型权重 | action-next-state pair |
| [[2026-SkillClaw]] | 如何把多用户经验压缩成共享 procedure | 外部 skills | session evidence group |

可以组合为：

```text
Native harnesses
  -> Polar proxy captures token-faithful trajectories
  -> next states / evaluator outputs
      -> OpenClaw-RL: policy optimization
      -> SkillClaw: explicit skill evolution
  -> regression gate
  -> updated policy + updated skills return to native harness
```

Polar 位于数据面，OpenClaw-RL 位于参数学习面，SkillClaw 位于显式程序性知识面。

## 16. 工程上最值得借鉴的设计

### 16.1 把兼容边界放在最稳定的外部接口

不要为每个 Agent framework 适配内部 callback graph。如果所有产品最终都走 model provider API，就在这个边界做 capture、routing、versioning 和 policy substitution。

### 16.2 Token provenance 必须是一等数据

训练 trace 不应只有文本，而应包含：

```text
token_id
source = sampled | harness_inserted | canonical_rendered
behavior_logprob
loss_mask
session_id / request_id / chain_id
policy_version
```

否则复杂 Agent context 经过序列化与压缩后，很难保证 loss 真正作用在 rollout token 上。

### 16.3 Session、request、chain、trace 必须区分

- session：一次完整任务执行；
- request：一次模型 API 调用；
- chain：满足严格 append-only 的对话分支；
- trace：trainer-facing 的训练样本。

把它们混为一谈会导致 subagent 拼错、reward 广播错误或样本碎片化。

### 16.4 初始化、运行、评价必须独立调度

Agent workload 的 CPU、GPU、I/O 与测试成本高度异构。Stage isolation、bounded READY buffer 和 evaluator prewarm 比简单增加并发数更能提高利用率。

## 17. 值得补充的实验

1. Browser、GUI、terminal、OS、research 和 multi-agent harness 的端到端训练；
2. 不同 harness 的 prefix merge rate、chains/session、断链原因；
3. 1/4/16/64 gateway nodes 的吞吐与 P50/P95/P99 latency；
4. Docker vs Apptainer 的启动成本和稳定性；
5. 多 seed SWE-Bench Verified 结果与显著性；
6. 同一 policy 联合训练四个 harness，测试 transfer 与 interference；
7. A harness 训练、B harness 测试的 cross-harness matrix；
8. per-request、prefix merge、process reward 和session normalization 的完整消融；
9. context compaction 高频 harness 下的有效训练 token 覆盖率；
10. synthetic streaming 与真实 streaming 的兼容性测试；
11. policy staleness 分布及 TIS 消融；
12. proxy normalization 对 tool-call semantics 的 differential testing；
13. secret/PII redaction 与 tenant isolation 的安全评测；
14. 失败 trajectories 用于 verifier、preference learning 的数据效率；
15. 与 Agent Lightning、rLLM、SkyRL-Agent 的同硬件端到端成本比较。

## 18. 一句话评价

> Polar 的关键贡献不是发明新的 RL 算法，而是找到一个足够低、足够稳定的系统切入点：在模型 API 边界旁路观察原生 Agent，通过 token-faithful reconstruction 把黑盒 harness 的真实执行路径变成训练轨迹；它显著降低了 harness-native RL 的集成成本，但 credit assignment、安全性与非 coding 场景的通用性仍需后续工作。

## Source

- Xu, B., Zhang, H., Zhang, S., Han, S., Liu, M., Hu, J., Diao, S., Jin, Z., Zou, Y., Demoret, M., Kautz, J., & Dong, Y. (2026). *Polar: Agentic RL on Any Harness at Scale*. arXiv:2605.24220v1.
- Local PDF: `C:/Users/huawei/Zotero/storage/IVLSQ5FT/Xu et al. - 2026 - Polar Agentic RL on Any Harness at Scale.pdf`
- Code: https://github.com/NVIDIA-NeMo/ProRL-Agent-Server

