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

$$
(o_t,l)
\xrightarrow{\mathcal F_\theta}
\{z_1,\ldots,z_M\},\{s_1,\ldots,s_K\}
\xrightarrow{\text{KV extraction}}
c_t
\xrightarrow{\pi_\phi}
a_t.
$$

- $\mathcal F_{\theta^T}$：Textual Teacher VLM，生成 explicit CoT 和 visual-plan answer；
- $\mathcal F_\theta$：Latent Student VLM，生成 compact latent CoT $z$ 和 spatial tokens；
- $\mathcal V_\psi$：Verbalizer，仅在训练时把 latent CoT 解码为文本；
- $\pi_\phi$：Diffusion Transformer-based Action Model，将 visual planning 转成 action chunk；
- $M=6$：latent reasoning token 数量；
- $K=5$：trajectory waypoint / spatial token 数量。

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

### 5.0 Unified notation

| Symbol | Meaning |
|---|---|
| $x_t=(o_t,l)$ | 当前 visual observation 与 language instruction |
| $\mathcal F_{\theta^T}$ | Textual Teacher VLM |
| $\mathcal F_\theta$ | Latent Student VLM |
| $\mathcal V_\psi$ | Verbalizer |
| $\pi_\phi$ | Diffusion Action Model |
| $\tau_i$ | Teacher 生成的第 $i$ 条 textual reasoning trace |
| $G(x_t)$ | 同一输入的 rollout group，$|G|=N=5$ |
| $z=(z_1,\ldots,z_M)$ | Student latent CoT，$M=6$ |
| $s_1,\ldots,s_K$ | learnable spatial tokens，$K=5$ |
| $\hat p_i$ / $p_i$ | ground-truth / predicted waypoint |
| $c_t$ | spatial-token KV 构成的 visual planning context |
| $\hat a_t$ | ground-truth action chunk |

Superscript $T$ 表示 Teacher；time index 始终写在 subscript $t$。

### 5.1 Teacher GRPO objective

Teacher 对同一输入 $x_t$ 采样 $N=5$ 条 rollouts。首先定义 policy ratio：

$$
r_{\theta^T}(\tau)
=
\frac{p_{\theta^T}(\tau\mid x_t)}
{p_{\theta^T_{\mathrm{old}}}(\tau\mid x_t)}.
$$

Teacher 最大化论文 Eq. (1) 的 clipped GRPO objective：

$$
\boxed{
\mathcal J_{\mathrm{GRPO}}(\theta^T)
=
\mathbb E_{\tau\sim\mathcal F_{\theta^T}}
\left[
\min\left(
r_{\theta^T}(\tau)A(\tau),
\operatorname{clip}\!\left(r_{\theta^T}(\tau),1-\varepsilon,1+\varepsilon\right)A(\tau)
\right)
\right]
}
\tag{1}
$$

每条 trace 的 reward 在同一 rollout group 内标准化：

$$
\boxed{
A(\tau)
=
\frac{R_\tau-\operatorname{mean}_{\tau_i\in G(x_t)}R_i}
{\operatorname{std}_{\tau_i\in G(x_t)}R_i}
}
\tag{2}
$$

$A(\tau)>0$ 表示该 trace 优于同组平均水平。Preference pair 为：

$$
\boxed{
\tau^+=\arg\max_{\tau\in G(x_t)}A(\tau),
\qquad
\tau^-=\arg\min_{\tau\in G(x_t)}A(\tau)
}
\tag{3}
$$

Reward 不直接作为 Student 的 scalar regression target，而是通过 $\tau^+\succ\tau^-$ 这一 preference relation 提供监督。

### 5.2 Student latent CoT

Student autoregressively 生成固定长度的 continuous latent sequence：

$$
z=(z_1,\ldots,z_M),
\qquad
z_m\in\mathbb R^d,
\qquad
M=6.
$$

约 250 个 textual reasoning tokens 被压缩为 6 个 latent tokens。这里仍有 dependency $p(z_m\mid z_{<m},x_t)$，只是 decoding steps 大幅减少。

## 6. Three Training Losses

三项 supervision 分别来自 preference pair、Teacher hidden state 和 ground-truth waypoints：

```text
tau+, tau-  --Verbalizer----------> Lverb
tau+        --Teacher hidden state-> Ldistill
waypoints   --spatial-token MLP----> Lans
```

### 6.1 Verbalization loss: $\mathcal L_{\mathrm{verb}}$

Continuous latent 没有直接的 token-level label。Verbalizer $\mathcal V_\psi$ 读取 $z$，并为 textual trace 计算 conditional likelihood。先定义 reference-adjusted score：

$$
s_\psi(\tau;z)
=
\log p_\psi(\tau\mid z)
-
\log p_{\mathrm{ref}}(\tau).
$$

其中整条 trace 的 log-likelihood 是 token log-likelihood 之和：

$$
\log p_\psi(\tau\mid z)
=
\sum_{j=1}^{|\tau|}
\log p_\psi(\tau_j\mid\tau_{<j},z).
$$

定义 preference margin：

$$
\Delta_\psi(z)
=
s_\psi(\tau^+;z)-s_\psi(\tau^-;z).
$$

论文 Eq. (4) 可重写为：

$$
\boxed{
\mathcal L_{\mathrm{verb}}
=
-\mathbb E\left[
\log\sigma\!\left(\beta\Delta_\psi(z)\right)
\right],
\qquad \beta=0.1
}
\tag{4}
$$

展开后等价于：

$$
\mathcal L_{\mathrm{verb}}
=
-\mathbb E\left[
\log\sigma\left(
\beta\left[
\log\frac{p_\psi(\tau^+\mid z)}{p_{\mathrm{ref}}(\tau^+)}
-
\log\frac{p_\psi(\tau^-\mid z)}{p_{\mathrm{ref}}(\tau^-)}
\right]
\right)
\right].
$$

最小化该 loss 会增大 preferred trace 相对 rejected trace 的 score。它约束的是“latent 支持哪一种 reasoning”，而非要求 Student 逐 token 复现 Teacher。

#### Verbalizer warm-up

前 3000 iterations 使用 $\tau^+$ 做 teacher forcing：

$$
\mathcal L_{\mathrm{warm}}
=
-\mathbb E\left[
\sum_{j=1}^{|\tau^+|}
\log p_\psi(\tau_j^+\mid\tau_{<j}^+,z)
\right].
$$

该阶段同时更新 $\psi$ 和 $\theta$，建立 $z\leftrightarrow\text{text}$ 的映射。后 1500 iterations 冻结 $\psi$ 并改用 $\mathcal L_{\mathrm{verb}}$；梯度穿过 frozen Verbalizer，只更新 Student $\theta$。

### 6.2 Visual-plan distillation loss: $\mathcal L_{\mathrm{distill}}$

定义 Teacher 与 Student 在 `<answer>` boundary 处的 contextual hidden states：

$$
h_{\mathrm{ans}}^T
=
H_{\theta^T}(x_t,\tau^+,\texttt{<answer>}),
$$

$$
h_{\mathrm{ans}}^S
=
H_\theta(x_t,z,\texttt{<answer>}).
$$

论文 Eq. (5) 使用 L2 alignment：

$$
\boxed{
\mathcal L_{\mathrm{distill}}
=
\left\|h_{\mathrm{ans}}^T-h_{\mathrm{ans}}^S\right\|_2^2
}
\tag{5}
$$

Teacher 与 Student 的 reasoning sequence 长度和表示形式不同，因此无法逐 token 对齐；`<answer>` 是两条路径共享的 semantic boundary。该 loss 只使用 preferred trace $\tau^+$ 的 Teacher state，将 action-aligned visual planning representation 迁移给 Student。

### 6.3 Waypoint regression loss: $\mathcal L_{\mathrm{ans}}$

Student 在 latent CoT 后追加 $K=5$ 个 learnable spatial tokens。第 $i$ 个 token 的 final hidden state 经过 waypoint head：

$$
p_i
=
f_{\mathrm{wp}}\!\left(h'(s_i)\right)
\in\mathbb R^6.
$$

六个维度统一编码 single-arm 和 bimanual trajectories：

$$
p_i
=
\left[
x_{\mathrm{single}},y_{\mathrm{single}},
x_{\mathrm{left}},y_{\mathrm{left}},
x_{\mathrm{right}},y_{\mathrm{right}}
\right].
$$

论文 Eq. (6) 给出的 waypoint loss 为：

$$
\mathcal L_{\mathrm{ans}}
=
\sum_{i=1}^{K}\left\|p_i-\hat p_i\right\|_2^2.
$$

实现中对不适用的 robot dimensions 使用 mask $m\in\{0,1\}^6$，因此更准确的实现形式是：

$$
\boxed{
\mathcal L_{\mathrm{ans}}
=
\sum_{i=1}^{K}
\left\|m\odot(p_i-\hat p_i)\right\|_2^2
}
$$

$$
m_{\mathrm{single}}=[1,1,0,0,0,0],
\qquad
m_{\mathrm{bimanual}}=[0,0,1,1,1,1].
$$

所有 spatial tokens 在同一次 Transformer forward 中计算，因此 waypoint prediction 是并行的；它不同于把坐标序列化为 60--70 个 textual tokens 后逐 token decoding。

### 6.4 Combined Student objective

论文 Eq. (6) 的完整目标为：

$$
\boxed{
\mathcal L_{\mathrm{student}}(\theta)
=
\mathcal L_{\mathrm{verb}}
+
\mathcal L_{\mathrm{distill}}
+
\mathcal L_{\mathrm{ans}}
}
\tag{6}
$$

论文没有报告额外的 weighting coefficients。三项监督的 target 与 gradient path 为：

| Loss | Target | Gradient destination | Function |
|---|---|---|---|
| $\mathcal L_{\mathrm{verb}}$ | $\tau^+\succ\tau^-$ | Student；warm-up 时也更新 Verbalizer | semantic preference |
| $\mathcal L_{\mathrm{distill}}$ | $h_{\mathrm{ans}}^T$ | Student | internal visual-plan transfer |
| $\mathcal L_{\mathrm{ans}}$ | $\hat p_{1:K}$ | Student + waypoint head | explicit spatial grounding |

## 7. Spatial Tokens and KV Cache

### 7.1 One Transformer layer

设第 $l$ 层输入为 $X^{(l)}\in\mathbb R^{n\times d}$。以 Pre-Norm block 为例：

$$
\widetilde X^{(l)}=\operatorname{Norm}(X^{(l)}),
$$

$$
Q^{(l)}=\widetilde X^{(l)}W_Q^{(l)},
\quad
K^{(l)}=\widetilde X^{(l)}W_K^{(l)},
\quad
V^{(l)}=\widetilde X^{(l)}W_V^{(l)}.
$$

Self-attention 与 residual update 为：

$$
A^{(l)}
=
\operatorname{softmax}\!\left(
\frac{Q^{(l)}K^{(l)\top}}{\sqrt{d_k}}+M_{\mathrm{attn}}
\right),
$$

$$
Y^{(l)}=X^{(l)}+A^{(l)}V^{(l)}W_O^{(l)},
$$

$$
X^{(l+1)}
=
Y^{(l)}
+
\operatorname{MLP}^{(l)}\!\left(\operatorname{Norm}(Y^{(l)})\right).
$$

对于 spatial token $s_i$：

$$
k_{s_i}^{(l)}
=
\operatorname{Norm}(x_{s_i}^{(l)})W_K^{(l)},
\qquad
v_{s_i}^{(l)}
=
\operatorname{Norm}(x_{s_i}^{(l)})W_V^{(l)}.
$$

层间真正传递的是 $X^{(l+1)}$。下一层使用自己的 $W_Q^{(l+1)},W_K^{(l+1)},W_V^{(l+1)}$ 重新生成 QKV，而不是直接接收上一层的 KV。

### 7.2 Why spatial tokens contain planning information

Spatial token 的 attention output 是 visible token values 的加权和：

$$
o_{s_i}^{(l)}
=
\sum_{j\in\mathcal V(i)}\alpha_{ij}^{(l)}v_j^{(l)},
\qquad
\alpha_{ij}^{(l)}
=
\operatorname{softmax}_j\!\left(
\frac{q_{s_i}^{(l)}k_j^{(l)\top}}{\sqrt{d_k}}
\right).
$$

$\mathcal V(i)$ 是 attention mask 允许 $s_i$ 访问的 token set。$\mathcal L_{\mathrm{ans}}$ 的 gradient 会调整 $W_Q,W_K,W_V$，使 spatial tokens 更有效地聚合 waypoint prediction 所需的 image、instruction 和 latent-reasoning features。

若沿用标准 causal mask，sequence order 为：

```text
image tokens + instruction tokens + z1...zM + s1...sK
```

则：

$$
\mathcal V(s_i)
=
\{\text{all preceding multimodal tokens},s_1,\ldots,s_i\}.
$$

因此 $s_K$ 可以访问全部 preceding tokens，而 $s_i$ 不能访问 $s_{i+1:K}$。论文没有明确说明是否为 spatial-token block 修改 attention mask，该点应以源码为准。

### 7.3 Extracting visual planning context $c_t$

令 $\mathcal E$ 表示选中的 earlier VLM layers，$S$ 表示 spatial-token positions。从完整 KV cache 中取对应切片：

$$
K_{\mathrm{sp}}^{(l)}=K^{(l)}[:,S,:],
\qquad
V_{\mathrm{sp}}^{(l)}=V^{(l)}[:,S,:],
\qquad l\in\mathcal E.
$$

Visual planning context 定义为：

$$
\boxed{
c_t
=
\left\{K_{\mathrm{sp}}^{(l)},V_{\mathrm{sp}}^{(l)}\right\}_{l\in\mathcal E}
}
$$

$c_t$ 不是预测出的 2D waypoints。二者来自同一组 spatial tokens，但用途不同：

| Representation | Source | Use |
|---|---|---|
| $p_i$ | final spatial hidden state $h'(s_i)$ | coordinate supervision / visualization |
| $c_t$ | earlier-layer spatial KV | Action Model conditioning |

LIBERO ablation 中，early-layer KV、late-layer KV、final hidden states 的 success rates 分别为 89.7、88.3、87.1。

## 8. Reasoning-Enhanced Policy Learning

### 8.1 Projecting and combining conditions

3.2 得到 high-level visual plan，3.3 将其转成 continuous low-level action chunk。VLM 与 Action Model 的 hidden dimensions 不同，因此先对 planning KV 做 projection：

$$
\widetilde K_{\mathrm{plan}}^{(j)}
=P_K^{(j)}\!\left(K_{\mathrm{sp}}^{(l_j)}\right),
\qquad
\widetilde V_{\mathrm{plan}}^{(j)}
=P_V^{(j)}\!\left(V_{\mathrm{sp}}^{(l_j)}\right).
$$

Projection target dimension 对 DiT-Policy 为 1024，对 RDT 为 2048。将 planning KV 与 frozen state encoder 的 KV 沿 context-token dimension 拼接：

$$
K_{\mathrm{cond}}^{(j)}
=
\left[K_{\mathrm{state}}^{(j)};\widetilde K_{\mathrm{plan}}^{(j)}\right],
\qquad
V_{\mathrm{cond}}^{(j)}
=
\left[V_{\mathrm{state}}^{(j)};\widetilde V_{\mathrm{plan}}^{(j)}\right].
$$

第 $j$ 个 Action Transformer block 的 cross-attention 为：

$$
\operatorname{CrossAttn}^{(j)}
=
\operatorname{softmax}\!\left(
\frac{Q_{\mathrm{action}}^{(j)}K_{\mathrm{cond}}^{(j)\top}}{\sqrt{d_k}}
\right)V_{\mathrm{cond}}^{(j)}.
$$

Planning context 回答“要做什么、目标在哪里”，state context 回答“机器人当前在哪里”，action queries 学习“具体应该怎样运动”。

### 8.2 Paper-level imitation-learning objective

Action demonstration dataset 记为：

$$
\mathcal D_{\mathrm{act}}
=
\{(o_t,l,\hat a_t)\}.
$$

论文 Eq. (7) 将所用 Diffusion Policy 的内部 denoising process 抽象为：

$$
\boxed{
\mathcal L_{\mathrm{IL}}(\phi)
=
\ell_{\mathrm{denoise}}\!\left(
\pi_\phi(o_t,l,c_t),\hat a_t
\right)
}
\tag{7}
$$

这不是普通的 direct action regression；$\ell_{\mathrm{denoise}}$ 沿用 DiT-Policy 或 RDT 的 diffusion objective。

### 8.3 Expanded epsilon-prediction form

下面是帮助理解 Eq. (7) 的常见 epsilon-prediction 展开式，不是论文额外提出的新 loss。令 clean demonstration action chunk 为：

$$
a_0\equiv\hat a_t.
$$

采样 diffusion step 与 Gaussian noise：

$$
\gamma\sim\operatorname{Uniform}\{1,\ldots,T\},
\qquad
\epsilon\sim\mathcal N(0,I).
$$

Forward noising process：

$$
a_\gamma
=
\sqrt{\bar\alpha_\gamma}\,a_0
+
\sqrt{1-\bar\alpha_\gamma}\,\epsilon.
$$

Conditional noise prediction：

$$
\hat\epsilon_\phi
=
\epsilon_\phi\!\left(a_\gamma,\gamma\mid o_t,l,c_t\right).
$$

对应 denoising loss：

$$
\boxed{
\mathcal L_{\mathrm{IL}}(\phi)
=
\mathbb E_{(o_t,l,a_0),\gamma,\epsilon}
\left[\left\|\epsilon-\hat\epsilon_\phi\right\|_2^2\right]
}
$$

根据当前 noise estimate 可以得到 clean-action estimate：

$$
\widehat a_{0,\phi}
=
\frac{
a_\gamma-\sqrt{1-\bar\alpha_\gamma}\,\hat\epsilon_\phi
}{
\sqrt{\bar\alpha_\gamma}
}.
$$

训练时 $a_0$ 和 sampled $\epsilon$ 均已知；inference 时二者未知，只有当前 sample $a_\gamma$、noise schedule 和 prediction $\hat\epsilon_\phi$ 已知。模型从 $a_T\sim\mathcal N(0,I)$ 开始，通过多步 reverse diffusion 得到 action chunk。

论文只指定 generic denoising objective；实际 DiT-Policy/RDT 也可能采用 $x_0$-prediction 或 $v$-prediction。因此以上是常见具体化，不是对原文实现的额外断言。

### 8.4 Frozen and trainable modules

| Module | Policy post-training status | Gradient source |
|---|---|---|
| Student VLM $\mathcal F_\theta$ | frozen | none |
| State Encoder | frozen | none |
| Latent Projector $P_K,P_V$ | trainable | $\mathcal L_{\mathrm{IL}}$ |
| Action Model $\pi_\phi$ | trainable | $\mathcal L_{\mathrm{IL}}$ |
| Teacher / Verbalizer | not used | none |

正文的“only update $\pi_\phi$”是简化表达；附录明确说明 latent projector 也随 Action Model 更新。

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

$$
R_{\text{total}}
=
R_{\text{task}}-\lambda\,|\tau|.
$$

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

$$
\boxed{
\text{preference-guided latent reasoning}
\rightarrow
\text{action-aligned spatial planning}
\rightarrow
\text{diffusion-based action execution}
}
$$

其中：

- $\mathcal L_{\text{verb}}$ 保证 latent 倾向高质量 semantic reasoning；
- $\mathcal L_{\text{distill}}$ 迁移 Teacher 的 internal visual-plan representation；
- $\mathcal L_{\text{ans}}$ 用 ground-truth waypoints 建立 spatial grounding；
- spatial-token KV cache 作为 $c_t$，由 Diffusion Action Model 翻译成 low-level robot actions。
