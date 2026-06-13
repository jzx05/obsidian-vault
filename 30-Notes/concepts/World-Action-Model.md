---
title: "World Action Model (WAM)"
tags: [concept, WAM, world-model, policy-learning, embodied-AI]
created: 2026-06-13
related:
  - "[[30-Notes/concepts/World-Model]]"
  - "[[30-Notes/concepts/Diffusion-Policy]]"
  - "[[30-Notes/concepts/Diffusion-Transformer]]"
  - "[[30-Notes/concepts/Flow-Matching]]"
  - "[[30-Notes/concepts/Mixture-of-Transformers]]"
---

# World Action Model (WAM)

## 定义

WAM = World Model + Action Model 的联合体。训练时同时学习**视频预测**（world modeling）和**动作预测**（action prediction），两个目标共享部分或全部骨干网络。

与纯 VLA（Vision-Language-Action）的区别：VLA 只学 observation → action 的映射，WAM 额外学 observation → future states 的预测。

---

## 核心假设

> 学会预测世界的未来状态（视频生成）能帮助更好地预测应该采取的动作。

为什么这可能成立：
- 视频预测需要理解物理规律（重力、碰撞、因果）
- 动作预测也需要理解同样的物理规律
- 共享 backbone → 视频任务学到的物理知识自动迁移到动作任务

---

## WAM 的三个层次（按 inference involvement 分类）

| Level | 推理时用 video？ | 代表方法 | 推理延迟 |
|-------|:---:|----------|---------|
| **Level 0: Training-only** | ✗ | [[20-Papers/WAM/2026-Fast-WAM]] | ~10ms |
| **Level 1: Feature extraction** | △ (一次 forward) | [[20-Papers/WAM/2026-DiT4DiT]] | ~50-100ms |
| **Level 2: Full imagination** | ✓ (完整生成) | 传统 WAM (imagine-then-execute) | ~500ms+ |

### Level 0: Video as Training Signal
```
训练: video loss + action loss → 共享 backbone
推理: 去掉 video 分支 → 直接出动作
```
- Video 只是辅助训练目标（auxiliary objective）
- 推理时 video 完全不存在
- 类比：BERT 的 [MASK] 预测——训练时用，推理时丢

### Level 1: Video as Feature Provider
```
训练: video loss + action loss → 双模型
推理: video model forward → hidden states → action model → 动作
```
- Video backbone 只做一次 forward（不做完整去噪 loop）
- 提取中间层特征给 action model
- 比 Level 2 快得多，但比 Level 0 慢

### Level 2: Full Imagination (传统 WAM)
```
训练: 学视频生成 + 学动作（可能分开训练）
推理: 生成未来视频 → 基于视频做决策
```
- 完整的 imagine-then-execute 范式
- 推理慢（需要多步去噪生成视频）
- 但生成的未来帧可用于 planning/MPC

---

## 关键设计维度

| 维度 | 选项 |
|------|------|
| **Video→Action 信息流** | 共享 self-attention (MoT) / feature extraction / cross-attention / 无直接连接 |
| **训练耦合度** | 联合训练 / 两阶段（先 video 再 action）/ 独立+对齐 |
| **推理时 video involvement** | Level 0 / 1 / 2（见上） |
| **生成框架** | Flow Matching / DDPM / Autoregressive |
| **Backbone** | 预训练 video DiT / 从头训练 / VLM-based |
| **Action 表示** | Delta action / Absolute / Chunk / Single-step |

---

## 方法对比表

| 方法 | Video→Action | 推理 Level | Backbone | 结果 (LIBERO) |
|------|:---:|:---:|------|:---:|
| **Fast-WAM** | MoT shared attn | 0 | Wan2.5-1B | **97.6%** |
| **DiT4DiT** | Hidden state extraction | 1 | Cosmos-2.2B | 90.4% |
| **GR-2** | Full video → policy | 2 | 自有 video model | — |
| **Genie-2** | Latent imagination | 2 | 自有 WM | — |
| 纯 VLA (OpenVLA) | 无 video | — | SigLIP + LLM | ~70% |

---

## 核心争论

**Fast-WAM 的论点**："推理时不需要 video，video co-training 的知识已经编码在 backbone 权重里了。"

**DiT4DiT 的论点**："video 的中间层表示在推理时仍有价值，能提供比纯 first-frame 更丰富的条件。"

**传统 WAM 的论点**："必须看到未来才能做好当下的决策（planning requires imagination）。"

这三者可能各有适用场景：
- 短 horizon 桌面操作 → Level 0 足够（Fast-WAM）
- 复杂长 horizon 任务 → Level 1 或 2 可能更好
- 需要显式 planning（如导航）→ Level 2 必要

---

## 与综述的关系

对应综述 §6（WM for Policy Learning）的子分类：
- §6.1：WM 作为 backbone（表征增强） → Level 0-1
- §6.2：WM 作为 simulator（生成 rollout） → Level 2
- §6.3：WM 作为 reward/evaluator → 另一个方向

Fast-WAM + DiT4DiT 一起提供了"WM 在推理时到底要参与多深"这个问题的实证答案。

---

## 相关概念
- [[30-Notes/concepts/World-Model]] — WM 的上位概念
- [[30-Notes/concepts/Diffusion-Policy]] — WAM 中 action head 的去噪策略
- [[30-Notes/concepts/Diffusion-Transformer]] — WAM 的 backbone 架构
- [[30-Notes/concepts/Flow-Matching]] — WAM 常用的生成框架
- [[30-Notes/concepts/Mixture-of-Transformers]] — Fast-WAM 的多模态耦合方式
- [[20-Papers/WAM/2026-Fast-WAM]] — Level 0 代表
- [[20-Papers/WAM/2026-DiT4DiT]] — Level 1 代表
