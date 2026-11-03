---
title: "06 · 感知对齐与 Grounding"
tags: [vla, loss, infonce, grounding, alignment]
created: 2026-11-03
source: "VLA-Handbook/theory/foundation/vla_loss_functions_handbook.md"
status: complete
---

# 06 · 感知对齐与 Grounding

> [!abstract] 一句话
> 当模型“听得懂但做不对”时，问题往往不在动作损失，而在感知与语义没有对齐。InfoNCE/CLIP 把图文、视触觉拉到同一嵌入空间；grounding 监督把语言指代落到具体对象/位置。

## 1. InfoNCE / CLIP 对齐

$$
\mathcal{L}_{\text{InfoNCE}}
=-\log
\frac{\exp(\operatorname{sim}(q,k^+)/\tau)}
{\sum_{j}\exp(\operatorname{sim}(q,k_j)/\tau)}
$$

- `q`：query 嵌入；
- `k^+`：正样本；
- `k_j`：负样本（in-batch negatives）；
- `τ`：温度，控制分布的锐度。

**张量**：

- `img_emb`: `[B, D]`
- `txt_emb`: `[B, D]`
- `logits`: `[B, B]`（batch 内两两相似度）
- `labels`: `[B]`（对角线为正样本）

**最小实现**：

```python
img = F.normalize(img_emb, dim=-1)
txt = F.normalize(txt_emb, dim=-1)
logits = img @ txt.T / tau
labels = torch.arange(B, device=img.device)
loss = F.cross_entropy(logits, labels)
```

实际 CLIP 会同时算 image→text 和 text→image 两个方向，取平均。

## 2. 用途

### 2.1 图文对齐

- 提升指令理解、对象检索、属性绑定；
- 让视觉 token 和语言 token 落在可比较的空间；
- 常用于 VLM 预训练/继续预训练。

### 2.2 视触觉对齐

- 让触觉 token 能“语义化”：接触、滑动、压力、材质；
- 把触觉编码器拉到视觉/语言空间，便于多模态融合；
- 对 VTLA 的接触阶段特别重要。

### 2.3 Grounding

语言理解 OK 但执行错对象/位置，常见原因：

- 模型知道“抓杯子”，但不知道哪个是杯子；
- 指令里的空间关系（左/右/上/下）没有落到图像坐标；
- 对象属性（颜色、材质）绑定错误；
- 遮挡/多物体场景下指代歧义。

**可加的监督**：

- 对象框 / 分割 mask；
- 关键点 / affordance；
- 空间关系标注（左、右、上、下、里、外）；
- 抓取点 / 接触点；
- 轨迹上的 waypoint。

## 3. 工程要点

| 要点 | 说明 |
|---|---|
| 温度 `τ` | 太小 → 分布过锐、训练不稳；太大 → 对比信号弱。常见 0.07 或可学习 |
| batch size | in-batch negatives 越多，对比越强；小 batch 需要 memory bank / 队列 |
| 难负样本 | 随机负样本太简单；用 hard negatives 提升判别力 |
| 归一化 | 嵌入先 L2 normalize，再算相似度 |
| 对称损失 | image→text + text→image，通常更稳 |
| 冻结策略 | 可冻结视觉/语言主干，只训投影头；也可全量/LoRA |
| 数据质量 | 噪声图文对会教坏对齐；清洗比调参重要 |

## 4. 和动作损失的关系

- 对齐损失在 **encoder/fusion 级**，动作损失在 **policy head 级**；
- 对齐好不一定动作好，但对齐差通常动作好不了；
- 如果动作损失已经很低但成功率上不去，优先查 grounding：可视化注意力、对象框、抓取点，确认模型看的是不是正确对象。

## 5. 相关笔记

- 动作损失入口：[[01-行为克隆与回归损失]]
- 总览：[[00-索引与速查]]
- 多模态 VLA：[[20261022-RT-2 — VLA 概念的诞生]]
