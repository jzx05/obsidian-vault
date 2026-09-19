---
title: X-Mind 知识包
aliases:
  - X-Mind MOC
tags:
  - MOC
  - paper
  - autonomous-driving
  - world-model
  - visual-cot
---

# X-Mind 知识包

> [!summary]
> X-Mind 把未来 12 帧结构化 BEV sketch 压缩为 96 个 latent tokens，并把 5 次 flow-matching 去噪分摊到 LLM 的 5 个层组中。模型在一次 backbone forward 内完成未来 rollout，再由 inverse-dynamics planner 根据未来 latent 预测车辆轨迹。

## 阅读入口

- [[X-Mind 深度解读]]：论文动机、表示、架构、损失、实验与批判性分析。
- [[X-Mind 训练与推理流程]]：逐 block 解释 GT 如何进入训练、推理时如何从噪声生成未来，以及如何与 LLM 和 planner 交互。
- [[X-Mind 相关工作定位]]：LCDrive、FutureX、FSDrive、OneVL、World-Action Model 等工作的机制比较。

## 一句话定位

```text
多视角图像 + 文本/导航 + 自车状态
                ↓
      LLM 内部的未来 sketch rollout
                ↓
     未来 latent 条件化 inverse dynamics
                ↓
      acceleration + yaw rate → trajectory
```

## 核心判断

X-Mind 的贡献不是“多加一个未来预测 loss”，而是让 future-sketch tokens 在 LLM 中间层逐步从噪声变清晰，并让 trajectory tokens 在同一组 self-attention 中读取这些未来状态。

但当前版本仍然是：

$$
o_t \rightarrow \hat B_{t+1:t+12} \rightarrow \hat a,
$$

而不是严格的反事实规划：

$$
a^{(i)} \rightarrow \hat B_{t+1:t+12}^{(i)}
\rightarrow \operatorname{Score}(a^{(i)},\hat B^{(i)}).
$$

作者在 Future Work 中也明确承认，控制动作与 sketch 的 joint sampling 尚未实现。

## 来源

- arXiv: [2606.28758](https://arxiv.org/abs/2606.28758)
- Project: [x-mind.github.io](https://x-mind.github.io)
- Local PDF: [打开本地 PDF](<file:///C:/Users/huawei/Zotero/storage/R38A5DNZ/Zhao%20et%20al.%20-%202026%20-%20X-Mind%20Efficient%20Visual%20Chain-of-Thought%20via%20Predictive%20World%20Model%20for%20End-to-End%20Driving.pdf>)

