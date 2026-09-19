---
title: "Fast-ThinkAct: Efficient Vision-Language-Action Reasoning via Verbalizable Latent Planning"
authors:
  - Chi-Pin Huang
  - Yunze Man
  - Zhiding Yu
  - Min-Hung Chen
  - Jan Kautz
  - Yu-Chiang Frank Wang
  - Fu-En Yang
year: 2026
venue: arXiv
arxiv: "2601.09708"
tags:
  - paper
  - VLA
  - embodied-reasoning
  - latent-reasoning
  - diffusion-policy
status: read
---

# Fast-ThinkAct

> [!abstract]
> Fast-ThinkAct 将 reasoning VLA 中约 250 个 explicit textual CoT tokens 压缩为少量 continuous latent tokens，并用 language preference、teacher visual-plan representation 和 ground-truth waypoints 三类信号约束 latent space。推理时不生成 textual CoT，也不使用 Verbalizer，而是将 spatial tokens 的中间层 KV cache 作为 visual planning context，注入 Diffusion Policy 生成 low-level robot actions。

## 1. 论文信息

- **Title**: Fast-ThinkAct: Efficient Vision-Language-Action Reasoning via Verbalizable Latent Planning
- **Authors**: Chi-Pin Huang, Yunze Man, Zhiding Yu, Min-Hung Chen, Jan Kautz, Yu-Chiang Frank Wang, Fu-En Yang
- **Version**: arXiv:2601.09708v2, 2026
- **Local PDF**: `C:\Users\huawei\Zotero\storage\6B6G7P4W\Huang et al. - 2026 - Fast-ThinkAct Efficient Vision-Language-Action Reasoning via Verbalizable Latent Planning.pdf`

## 2. Problem

Reasoning VLA 通过 explicit CoT 提高 long-horizon planning、failure recovery 和 out-of-distribution generalization，但逐 token 生成长 reasoning trace 会导致数秒级 latency，无法满足机器人约 1--15 Hz 的控制需求。

直接缩短 textual CoT 也有两个问题：

1. 关键 reasoning information 可能随 token 删除而丢失；
2. 纯语言表示不能充分保留 manipulation 所需的 spatial-temporal information。

Fast-ThinkAct 的目标不是简单地让模型“少说几句”，而是将 linguistic reasoning 和 visual planning 压缩到 compact continuous representations 中，并让这些表示能够指导 action execution。

## 3. Overall Architecture

给定 timestep $t$ 的 observation $o_t$ 和 instruction $l$：

\[
(o_t,l)
\xrightarrow{\mathcal F_\theta}
\{z_1,\ldots,z_M\},\{s_1,\ldots,s_K\}
\xrightarrow{\text{KV extraction}}
c_t
\xrightarrow{\pi_\phi}
a_t.
\]

- $\mathcal F_{\theta^T}$：Textual Teacher VLM，生成 explicit CoT 和 visual-plan answer；
- $\mathcal F_\theta$：Latent Student VLM，生成 compact latent CoT $z$ 和 spatial tokens；
- $\mathcal V_\psi$：Verbalizer，仅在训练时把 latent CoT 解码为文本；
- $\pi_\phi$：Diffusion Transformer-based Action Model，将 visual planning 转成 action chunk；
- (M=6)：latent reasoning token 数量；
- (K=5)：trajectory waypoint / spatial token 数量。

核心层级为：

```text
latent reasoning
    -> visual spatial planning
    -> continuous robot action
```

## 4. Models

| Component | Model / initialization | Role |
|---|---|---|
| Teacher VLM | Qwen2.5-VL-3B, initialized from CoT-SFT checkpoint | GRPO textual reasoning |
| Student VLM | Qwen2.5-VL-3B, initialized from the same checkpoint | latent reasoning and spatial planning |
| Larger-scale variant | Qwen2.5-VL-7B | scalability experiment |
| Verbalizer | Qwen3-0.6B + per-layer cross-attention | decode latent CoT during training |
| Action Model on SimplerEnv | DiT-Policy | continuous action generation |
| Action Model on LIBERO / RoboTwin2.0 | RDT | continuous/bimanual action generation |

Teacher 和 Student 使用相同 backbone 和相同 CoT-SFT initialization，这也使 hidden-state L2 alignment 更合理。

## 5. Efficient Embodied Reasoning

### 5.1 Teacher GRPO and preference construction

Teacher 对同一个输入采样 (N=5) 条 reasoning rollouts：

\[
\{\tau_1,\ldots,\tau_N\}.
\]

每条 rollout 获得 task、trajectory 或 QA reward，并在 group 内归一化：

\[
A(\tau)=
\frac{R_\tau-\operatorname{mean}(\{R_i\})}
{\operatorname{std}(\{R_i\})}.
\]

随后选出：

\[
\tau^+=\arg\max_{\tau\in G}A(\tau),
\qquad
\tau^-=\arg\min_{\tau\in G}A(\tau).
\]

$\tau^+$ 和 $\tau^-$ 分别构成 preferred / rejected reasoning trace。Reward 并不直接回归到 Student latent，而是用来构造 preference pair。

### 5.2 Student latent CoT

Student 不生成 textual reasoning，而是 autoregressively 生成 (M) 个 continuous latent vectors：

\[
z=\{z_m\}_{m=1}^{M},
\qquad z_m\in\mathbb R^d,
\qquad M=6.
\]

主要效率收益来自把约 250 个 textual tokens 缩短为 6 个 latent tokens。这里仍然是 autoregressive latent generation，只是 sequence length 大幅缩短。

## 6. Three Training Losses

Student 的总目标为：

\[
\mathcal L_{\text{student}}
=
\mathcal L_{\text{verb}}
+
\mathcal L_{\text{distill}}
+
\mathcal L_{\text{ans}}.
\]

论文未报告额外的 loss weights，正文按等权相加描述。

### 6.1 Verbalization loss: \(\mathcal L_{\text{verb}}\)

#### Motivation

Continuous latent space 没有 token-level ground truth。作者用 Verbalizer $\mathcal V_\psi$ 将 latent $z$ 解码为 natural-language reasoning，并要求 latent 更支持高质量的 $\tau^+$，而不是低质量的 $\tau^-$。

#### Objective

\[
\mathcal L_{\text{verb}}
=-\mathbb E\left[
\log\sigma\left(
\beta\left(
\log\frac{p_\psi(\tau^+\mid z)}{p_{\mathrm{ref}}(\tau^+)}
-
\log\frac{p_\psi(\tau^-\mid z)}{p_{\mathrm{ref}}(\tau^-)}
\right)
\right)
\right],
\qquad \beta=0.1.
\]

$p_{\mathrm{ref}}$ 是不使用 latent conditioning 的 reference model，用来抵消语言模型本身对某类句子的 prior preference。

该 loss 不要求 Student 逐 token 复制 Teacher，而是要求 (z) 中包含足够的信息，使 Verbalizer 对 preferred reasoning 的相对 likelihood 高于 rejected reasoning。

#### Optimization target

- 保留 task-relevant semantic reasoning；
- 压制低 reward、冗余或错误的 reasoning pattern；
- 让 latent 保持 verbalizable，而不是不可解释的 arbitrary vector。

#### Warm-up strategy

- 前 3000 iterations：使用 $\tau^+$ 作为 ground truth，以 standard language-modeling loss 训练 Verbalizer 与 Student；
- 后 1500 iterations：冻结 Verbalizer，使用 $\mathcal L_{\text{verb}}$ 更新 Student；
- Student 在两个阶段始终更新。

后半程中，冻结的 Verbalizer 相当于 differentiable semantic evaluator，梯度通过它回传到 Student latent。

### 6.2 Visual-plan distillation loss: \(\mathcal L_{\text{distill}}\)

只有 language preference 不能保证 latent 包含机器人控制所需的 spatial planning。作者因此对齐 Teacher 和 Student 在 `<answer>` token 位置的 hidden state：

\[
\mathcal L_{\text{distill}}
=
\left\|h_t^T-h_t\right\|_2^2.
\]

- $h_t^T$：Teacher 在 preferred trace $\tau^+$ 之后、`<answer>` token 位置的 hidden state；
- $h_t$：Student 在 latent CoT 之后、`<answer>` token 位置的 hidden state。

`<answer>` 是 reasoning 与 final answer / visual plan 之间的 special boundary token。其 contextual hidden state 已经聚合 image、instruction 和 preceding reasoning，因此可作为“准备输出 visual plan 时的内部状态”。

使用该位置的好处是 Teacher 的长 textual CoT 和 Student 的 6 个 latent tokens 无法逐步对齐，但二者都共享 `<answer>` 这一语义边界。

该 loss 的监督链条是：

```text
trajectory/task reward
    -> preferred Teacher rollout tau+
    -> Teacher <answer> hidden state
    -> Student hidden-state alignment
```

### 6.3 Waypoint regression loss: \(\mathcal L_{\text{ans}}\)

Student 在 latent sequence 后追加 (K=5) 个 learnable spatial tokens：

\[
s_1,\ldots,s_K.
\]

每个 spatial token 的 final hidden state 通过 MLP 并行预测 waypoint：

\[
p_i=\operatorname{MLP}(h'(s_i)),
\]

\[
\mathcal L_{\text{ans}}
=
\sum_{i=1}^{K}\|p_i-\hat p_i\|_2^2.
\]

实现中每个 waypoint 为六维：

\[
p_i=
[x_{\text{single}},y_{\text{single}},
x_{\text{left}},y_{\text{left}},
x_{\text{right}},y_{\text{right}}].
\]

- single-arm sample：监督前两维，mask 后四维；
- bimanual sample：监督左右夹爪对应的后四维，mask 前两维。

Teacher 用文本表示 5 个 waypoint 时需要约 60--70 tokens；Student 将 5 个 spatial tokens 预先放进序列，在一次 Transformer forward 中同时回归全部 waypoint。

### 6.4 Roles of the three losses

| Loss | Supervision source | What it constrains |
|---|---|---|
| $\mathcal L_{\text{verb}}$ | preferred/rejected Teacher traces | semantic reasoning quality |
| $\mathcal L_{\text{distill}}$ | Teacher `<answer>` hidden state | internal visual-plan representation |
| $\mathcal L_{\text{ans}}$ | ground-truth 2D gripper trajectory | explicit spatial grounding |

可以简化为：

```text
Lverb    : think correctly
Ldistill : inherit Teacher visual planning
Lans     : ground planning in actual coordinates
```

## 7. Spatial Tokens and KV Cache

### 7.1 Layer-wise computation

对于第 $l$ 层中的 spatial token $s_i$，严格地说：

\[
k_{s_i}^{(l)}
=
\operatorname{Norm}(x_{s_i}^{(l)})W_K^{(l)},
\qquad
v_{s_i}^{(l)}
=
\operatorname{Norm}(x_{s_i}^{(l)})W_V^{(l)}.
\]

该层使用 (Q^{(l)},K^{(l)},V^{(l)}) 完成 self-attention 和 MLP，再得到下一层 hidden state：

\[
X^{(l+1)}
=
\operatorname{TransformerBlock}^{(l)}(X^{(l)}).
\]

下一层不会直接把上一层 (K,V) 当输入，而是从更新后的 (X^{(l+1)}) 用新一层 projection matrices 重新计算 (Q,K,V)。真正沿网络深度传递的是 hidden state。

### 7.2 Why spatial tokens contain planning information

Spatial tokens 可以通过 self-attention 读取此前的 image tokens、instruction tokens 和 latent reasoning tokens。由于 $\mathcal L_{\text{ans}}$ 要求它们预测真实轨迹，backpropagation 会逐渐训练其 Query 去检索与 waypoint 相关的视觉和语言特征。

“spatial token 关注杯子或夹爪”只是直观说法。严格地说，它对 multimodal token values 做 content-dependent weighted aggregation，并形成足以供后续网络恢复位置和空间关系的 distributed representation。

在标准 causal mask 下，如果 sequence order 为：

```text
image tokens + instruction tokens + z1...zM + s1...sK
```

则 $s_K$ 能看到所有 preceding tokens，包括 $s_1,\ldots,s_{K-1}$；$s_i$ 不能看到后续 $s_{i+1},\ldots,s_K$。论文没有明确报告 spatial block 使用特殊 bidirectional mask，因此该点应以开源实现为准。

### 7.3 Visual latent planning \(c_t\)

论文不是直接把二维 waypoint 喂给 Action Model，而是从 earlier VLM layers 提取 spatial tokens 对应的 KV cache：

\[
c_t
=
\left\{
K_{\text{spatial}}^{(l)},
V_{\text{spatial}}^{(l)}
\right\}_{l\in\mathcal L_{\text{early}}}.
\]

因此：

- waypoint $p_i$：final hidden state 经 MLP 得到，便于 coordinate supervision 和 visualization；
- visual plan $c_t$：intermediate spatial-token KV，提供给 Action Model，保留更丰富的 object、task phase、spatial relation 和 trajectory information。

论文的 LIBERO ablation：

| Action conditioning | Success rate |
|---|---:|
| early-layer KV | 89.7 |
| late-layer KV | 88.3 |
| final output hidden states | 87.1 |

作者据此认为 earlier-layer representations 更适合保留 action prediction 所需的 visual-spatial information。

## 8. Reasoning-Enhanced Policy Learning

### 8.1 Purpose

3.2 得到的是 high-level visual plan，而机器人最终需要 continuous low-level controls，例如 end-effector translation、rotation、joint motion 和 gripper state。3.3 使用 Diffusion Action Model 完成：

\[
\text{visual plan}
\rightarrow
\text{robot action chunk}.
\]

### 8.2 Conditioning Action Model with planning and state

VLM planning KV 先通过 linear projector 映射到 Action Model dimension：

- DiT-Policy：1024；
- RDT：2048。

随后与 frozen state encoder 产生的 state KV 拼接：

\[
K_{\text{cond}}
=
[K_{\text{state}};K_{\text{plan}}],
\qquad
V_{\text{cond}}
=
[V_{\text{state}};V_{\text{plan}}].
\]

Action Model 的 cross-attention 使用 action tokens 作为 Query：

\[
\operatorname{Attention}
(Q_{\text{action}},K_{\text{cond}},V_{\text{cond}}).
\]

其中：

- visual planning context $c_t$ 提供“应该做什么、目标在哪里”；
- state observation 提供“机器人当前处于什么状态”；
- Action Model 学习“具体应该如何运动”。

### 8.3 Imitation-learning objective

训练数据为 action-annotated robot demonstrations：

\[
(o_t,l,\hat a_t),
\]

其中 $\hat a_t$ 通常是一段 ground-truth action chunk。论文写为：

\[
\mathcal L_{\mathrm{IL}}(\phi)
=
\ell\left(
\pi_\phi(o_t,l,c_t),
\hat a_t
\right),
\]

其中 $\ell$ 沿用 DiT-Policy / RDT 的 standard diffusion denoising objective。

若采用常见 epsilon-prediction parameterization，训练过程为：

1. 从 demonstration 取得 clean action chunk $a_0=\hat a_t$；
2. 采样 $\epsilon\sim\mathcal N(0,I)$ 和 diffusion step $\gamma$；
3. 构造 noisy action：

\[
a_\gamma
=
\sqrt{\bar\alpha_\gamma}a_0
+
\sqrt{1-\bar\alpha_\gamma}\epsilon;
\]

4. 模型在条件 $(o_t,l,c_t)$ 下预测噪声：

\[
\hat\epsilon_\phi
=
\pi_\phi(a_\gamma,\gamma,o_t,l,c_t);
\]

5. 最小化：

\[
\mathcal L_{\mathrm{IL}}
=
\|\epsilon-\hat\epsilon_\phi\|_2^2.
\]

预测噪声等价于学习当前 noisy action 应朝哪个方向修正。根据：

\[
\hat a_{0,\phi}
=
\frac{
a_\gamma-
\sqrt{1-\bar\alpha_\gamma}\hat\epsilon_\phi
}{
\sqrt{\bar\alpha_\gamma}
},
\]

可以得到当前 clean action estimate。训练时 clean action 和 sampled noise 都已知；inference 时 clean action 未知，模型从 Gaussian noise 开始进行多步 reverse diffusion，最后得到 action chunk。

论文正文仅抽象指定 denoising objective，具体实现也可能采用 $x_0$-prediction 或 $v$-prediction，应以 DiT-Policy/RDT implementation 为准。

### 8.4 Frozen and trainable modules

Policy post-training 阶段：

| Module | Status |
|---|---|
| Student VLM $\mathcal F_\theta$ | frozen |
| State Encoder | frozen |
| Latent Projector | trainable |
| Action Model $\pi_\phi$ | trainable |
| Teacher / Verbalizer | not used |

正文的“only update $\pi_\phi$”是简化表达；附录明确指出 latent projector 也会更新。

## 9. Training Data

### 9.1 2D visual trajectories

#### Single-arm

- Source: Open X-Embodiment (OXE)
- Labels: MolmoAct 2D visual trajectories
- Scale: approximately 1.3M trajectories

#### Bimanual

- Source: AIST bimanual manipulation dataset
- Scale: approximately 92K trajectories
- Processing:
  1. 用 Molmo-72B 在第一帧检测 left/right gripper positions；
  2. 用 CoTracker3 在后续视频帧中跟踪 grippers；
  3. 将 tracking results 解析为 bimanual 2D trajectories。

这些轨迹是 image-plane gripper trajectories，不是 robot joint trajectories。它们主要用于 $\mathcal L_{\text{ans}}$ 和 spatial grounding，最终 low-level controls 仍由 Action Model 结合 robot state 生成。

### 9.2 Reasoning and QA data

- PixMo
- RoboFAC
- RoboVQA
- ShareRobot
- EgoPlan
- Video-R1-CoT

训练阶段：

1. **SFT**：约 4M samples，建立基础 visual understanding、task comprehension 和 manipulation knowledge；
2. **CoT-SFT**：约 200K sampled SFT data + 165K Video-R1-CoT；
3. **Teacher-Student training**：从每个 dataset/data type 采样约 5K，总计约 50K samples。

### 9.3 Action data

- SimplerEnv / DiT-Policy：OXE action data；
- LIBERO / RoboTwin2.0 / RDT：OXE + static ALOHA data；
- target environment adaptation：使用对应 benchmark demonstrations fine-tune。

## 10. Baselines

### 10.1 Embodied reasoning

- GPT-4V
- Gemini-2.5-Flash
- InternVL2.5 / InternVL3
- NVILA
- Qwen2.5-VL
- Magma
- RoboBrain2.0
- ThinkAct

### 10.2 Robot manipulation

- Diffusion Policy (DP)
- ACT
- $\pi_0$
- RDT
- OpenVLA
- CoT-VLA
- ThinkAct
- MolmoAct

### 10.3 Efficient reasoning baselines

- Textual Teacher $\mathcal F_{\theta^T}$
- Teacher inference without thinking
- Teacher inference with only 6 textual tokens
- Teacher with RL Length-Penalty
- Fast-ThinkAct latent reasoning

`Teacher w/ RL Length-Penalty` 指在 GRPO reward 中加入 reasoning-length penalty，例如：

\[
R_{\text{total}}
=
R_{\text{task}}-\lambda\,|\tau|.
\]

它仍然 autoregressively 生成 textual CoT，只是鼓励更短的 reasoning；Fast-ThinkAct 则改变 reasoning representation，用 continuous latent tokens 代替文本。

## 11. Main Results

### 11.1 Reasoning benchmarks, 3B scale

| Method | Overall average |
|---|---:|
| Qwen2.5-VL-3B | 35.6 |
| RoboBrain2.0-3B | 46.1 |
| ThinkAct-3B | 49.4 |
| Fast-ThinkAct-3B | **52.8** |

### 11.2 Robot manipulation and latency

| Method | LIBERO | SimplerEnv-Google | Latency |
|---|---:|---:|---:|
| OpenVLA-7B | 76.5 | 40.2 | N/A |
| CoT-VLA-7B | 83.9 | N/A | N/A |
| ThinkAct-7B | 84.4 | 68.3 | 7513 ms |
| MolmoAct-7B | 86.8 | 64.9 | 6723 ms |
| ThinkAct-3B | 83.1 | 64.7 | 5674 ms |
| Fast-ThinkAct-3B | **89.7** | **68.7** | **805 ms** |

论文报告相对 ThinkAct-7B 最高 89.3% latency reduction。更公平的 same-scale comparison 是 Fast-ThinkAct-3B 对 ThinkAct-3B，仍约快 7 倍。

### 11.3 Training-objective ablation

| Method | Reasoning average | Manipulation average |
|---|---:|---:|
| Fast-ThinkAct | **52.8** | **68.2** |
| w/o $\mathcal L_{\text{verb}}$ | 48.5 | 66.9 |
| w/o $\mathcal L_{\text{verb}},\mathcal L_{\text{distill}}$ | 47.7 | 64.9 |
| Textual Teacher | 49.8 | 67.2 |
| SFT + CoT-SFT | 45.0 | 65.4 |
| SFT only | 46.5 | 64.7 |

结果支持 preference-guided verbalization 和 visual-plan alignment 的增益，但论文未单独报告去掉 $\mathcal L_{\text{ans}}$ 的完整消融。

### 11.4 Efficient textual reasoning comparison

| Method | Average |
|---|---:|
| Textual Teacher | 49.8 |
| Inference without thinking | 46.5 |
| Inference with 6 textual tokens | 46.3 |
| RL Length-Penalty | 47.8 |
| Fast-ThinkAct-3B | **53.3** |

这说明直接裁剪 textual reasoning 或对长度加 penalty 会损害能力；latent compression 并不等价于简单缩短文本。

## 12. Verbalized Latent Reasoning

在 RoboVQA 上，Teacher textual reasoning 和 Student verbalized latent reasoning 都能捕获 task-relevant information，但 Teacher 更冗长，包含更多与任务不直接相关的内容；Student 经过 Verbalizer 解码后的 reasoning 更简洁、聚焦。

作者据此认为 preference-guided distillation 不仅降低 computational cost，还会过滤 redundant information。不过这里的“interpretability”是弱意义上的：latent 可以被 Verbalizer 解码，不表示每个 latent token 都具有稳定且可验证的独立语义。

## 13. Critical Analysis

### Strengths

1. 将 semantic reasoning、visual representation alignment 和 explicit trajectory grounding 分开监督，设计逻辑清晰；
2. Inference 完全移除 Teacher、Verbalizer 和 textual CoT，实际 latency 收益显著；
3. spatial-token KV 为 VLM planning 与 Action Model 之间提供信息量较高的接口；
4. 兼容 DiT-Policy 和 RDT，说明方法并非绑定单一 Action Model；
5. same-scale ThinkAct-3B comparison 同时提升 performance 和 efficiency。

### Limitations and open questions

1. **Loss scaling**：三个 loss 直接相加，但缺少 scale、weight sensitivity 和 gradient conflict 分析；
2. **Representation alignment assumption**：hidden-state L2 alignment 依赖 Teacher/Student 相同 initialization 和 representation space，对 heterogeneous Teacher 的适用性未知；
3. **Weak interpretability**：Verbalizer reconstruction 不等于 latent faithful explanation；
4. **KV extraction details**：正文没有给出具体 earlier-layer indices、layer mapping 和 precise KV concatenation implementation；
5. **Attention mask ambiguity**：未明确 spatial tokens 之间使用 causal 还是 special bidirectional mask；
6. **Limited ablation**：缺少单独移除 $\mathcal L_{\text{ans}}$、不同 loss weights 以及 alternative representation interface 的充分实验；
7. **Latency claim**：89.3% 是 3B method 相对 7B baseline 的最优口径，same-scale 结论应优先引用约 7x acceleration；
8. **2D planning bottleneck**：image-plane trajectories 对 depth、contact、force 和 occlusion 的表达有限，实际执行仍高度依赖 pretrained Action Model；
9. **Two-stage optimization**：VLM 在 policy post-training 中 frozen，稳定但无法由 downstream action error 端到端修正 planning representation。

## 14. Key Takeaway

Fast-ThinkAct 的核心不是“把 CoT 截短”，而是建立一条受三类监督约束的 compact planning channel：

\[
\boxed{
\text{preference-guided latent reasoning}
\rightarrow
\text{action-aligned spatial planning}
\rightarrow
\text{diffusion-based action execution}
}
\]

其中：

- $\mathcal L_{\text{verb}}$ 保证 latent 倾向高质量 semantic reasoning；
- $\mathcal L_{\text{distill}}$ 迁移 Teacher 的 internal visual-plan representation；
- $\mathcal L_{\text{ans}}$ 用 ground-truth waypoints 建立 spatial grounding；
- spatial-token KV cache 作为 $c_t$，由 Diffusion Action Model 翻译成 low-level robot actions。
