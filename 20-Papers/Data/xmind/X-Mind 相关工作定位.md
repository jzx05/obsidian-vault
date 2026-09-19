---
title: X-Mind 相关工作定位
tags:
  - autonomous-driving
  - related-work
  - world-model
  - chain-of-thought
  - VLA
---

# X-Mind 相关工作定位

## 1. 判断“世界模型是否真的嵌在策略内部”

由弱到强可以分成四级：

| Level | 机制 | 典型形式 |
|---|---|---|
| L1 | 世界模型只提供训练监督 | 推理时 future decoder 被丢弃 |
| L2 | 推理时先预测一个未来，再据此规划 | $o\rightarrow\hat z_{future}\rightarrow a$ |
| L3 | 未来和动作联合生成 | $(\hat z_{future},a)$ 共同去噪/自回归 |
| L4 | 对候选动作生成反事实未来并评分 | $a^{(i)}\rightarrow \hat z^{(i)}\rightarrow score$ |

X-Mind 属于 L2：它在推理时显式 rollout 未来，但没有公开展示 action-conditioned counterfactual rollout。

## 2. 最接近 X-Mind 的工作

| 工作 | 表示 | 推理机制 | 相对 X-Mind 的关键差别 |
|---|---|---|---|
| [FSDrive](https://arxiv.org/abs/2505.17685) | future image + lane/3D box | 未来视觉 CoT 后做 inverse dynamics | 视觉更稠密，token/生成成本更高 |
| [FutureX](https://arxiv.org/abs/2512.11226) | latent future scene | Auto-think switch 决定是否 rollout | 可按场景跳过慢思考 |
| [LCDrive](https://arxiv.org/abs/2512.10226) | action/world tokens | 动作提议与其未来结果交替生成 | future outcome 与 action proposal 对齐，因果性更强 |
| [X-Foresight](https://arxiv.org/abs/2605.24892) | future video chunks | 世界预测与实时动作联合学习 | 保留高保真视频，计算更重 |
| [WA-JEPA](https://arxiv.org/abs/2608.20974) | future scene latent | future tokens 与 trajectory 共同 flow matching | 动作监督直接塑造未来 latent |
| [MM-Future](https://arxiv.org/abs/2609.20377) | 多组 scene-action hypotheses | 多模态未来与动作共同演化并评分 | 显式处理多种可能未来 |
| [MindDrive](https://arxiv.org/abs/2512.04441) | ego-conditioned future scene | what-if simulation + VLM evaluator | 更接近 MPC，但模块和推理链更长 |

## 3. 只在训练期使用世界模型的工作

| 工作 | 训练信号 | 推理阶段 |
|---|---|---|
| [DriveVLA-W0](https://arxiv.org/abs/2510.12796) | 未来图像预测提供 dense supervision | 轻量 action expert，不一定生成未来 |
| [OneVL](https://arxiv.org/abs/2604.18486) | latent 同时重建 text CoT 与 future-frame tokens | 两个辅助 decoder 均丢弃 |
| [LaST-VLA](https://arxiv.org/abs/2603.01928) | 3D 几何与 world-model dynamics 蒸馏到 latent | latent 直接指导轨迹 |
| [SimWAM](https://arxiv.org/abs/2608.07468) | video/action joint flow matching | attention mask 让 action 绕过 future generation |
| [Metis](https://arxiv.org/abs/2606.15869) | video expert + action expert 联合训练 | action expert 不显式生成未来 |

这些方法更快，但不能声称模型在测试时“先想象未来再行动”。更准确的说法是世界模型提供 representation learning 或 privileged training signal。

## 4. 内部 CoT，但没有显式世界 rollout

| 工作 | Reasoning | Action interface |
|---|---|---|
| [AutoVLA](https://arxiv.org/abs/2506.13757) | 同一自回归模型选择 fast/slow textual CoT | discrete physical action tokens |
| [ReCogDrive](https://arxiv.org/abs/2506.08052) | VLM 学习分层驾驶认知 | VLM prior 注入 diffusion planner |
| [ORION](https://arxiv.org/abs/2503.19755) | LLM 结合历史进行场景推理 | generative planner |
| [CoT4AD](https://arxiv.org/abs/2511.22532) | 训练显式 CoT，推理隐式 reasoning | trajectory planning |
| [ColaVLA](https://arxiv.org/abs/2512.22939) | textual cognition 压缩成 meta-action latent | hierarchical parallel planner |
| [[Fast-ThinkAct]] | 6 个 latent reasoning + 5 个 spatial tokens | spatial-token KV 注入 diffusion policy |

## 5. 三条主要技术路线

### 路线 A：Predict then act

$$
o_t\rightarrow \hat z_{future}\rightarrow a_t.
$$

代表：X-Mind、FSDrive、FutureX。

优点：未来表征可视化、可检查。缺点：如果没有 action conditioning，未来可能只是行为数据中最常见的演化。

### 路线 B：Joint world-action modeling

$$
p(z_{future},a\mid o_t).
$$

代表：DriveWAM、WA-JEPA、MM-Future、BrainWAM。

优点：动作与未来互相塑造。缺点：联合生成可能出现 modality competition，视频 token 数量远大于 action token。

### 路线 C：Distill future knowledge into policy latent

$$
\mathcal L
=\mathcal L_{action}
+\lambda\mathcal L_{future},
\qquad
\text{inference: }o_t\rightarrow h\rightarrow a_t.
$$

代表：OneVL、DriveVLA-W0、LaST-VLA、SimWAM、Metis。

优点：速度接近普通 action-only policy。缺点：推理时不能检查模型具体想象了什么，也不能自然增加 rollout 或候选动作数。

## 6. 对 X-Mind 的研究启发

可以沿以下方向升级：

1. **Action-conditioned RBD**：让每组 sketch tokens 同时接收候选轨迹 token；
2. **Multi-hypothesis rollout**：并行生成 $M$ 个 future-action pairs，而非单一未来；
3. **Planning-aware scorer**：从碰撞、法规、舒适、进度等维度评价未来；
4. **Self-supervised sketch latent**：降低对完整结构化 GT 的依赖；
5. **Closed-loop RL/post-training**：让 rollout representation 接受真实交互结果而不仅是日志 imitation；
6. **Uncertainty calibration**：未来不确定时主动减速，而非输出单个过度确定的 sketch；
7. **Public closed-loop evaluation**：在 Bench2Drive/HUGSIM 等交互环境报告碰撞、成功率和延迟。

## 7. 阅读顺序

1. [[X-Mind 深度解读]]：理解内部 layer-wise future generation；
2. [FSDrive](https://arxiv.org/abs/2505.17685)：最直接的 Visual CoT 对照；
3. [LCDrive](https://arxiv.org/abs/2512.10226)：理解 action-aligned latent world tokens；
4. [MM-Future](https://arxiv.org/abs/2609.20377)：理解多未来、多动作联合建模；
5. [OneVL](https://arxiv.org/abs/2604.18486)：理解“在线 rollout”与“训练期蒸馏”的效率取舍。

