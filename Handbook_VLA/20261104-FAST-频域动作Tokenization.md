---
title: FAST — 频域动作 Tokenization 深度解读
tags:
  - handbook
  - vla
  - tokenization
  - fast
  - dct
  - bpe
created: 2026-11-04
source: https://arxiv.org/abs/2501.09747
status: reading
related:
  - "[[20261028-Compression Gap]]"
  - "[[20261023-OpenVLA]]"
  - "[[20261022-RT-2 — VLA 概念的诞生]]"
  - "[[20261103-RDT-1B]]"
  - "[[20261103-具身智能14主题面试准备]]"
---

# FAST — 频域动作 Tokenization 深度解读

> [!abstract] 核心思想
> FAST 把动作 tokenization 当成一个**压缩编解码问题**，而不是“每个关节分 256 个 bin”。它先对动作 chunk 沿时间轴做 **DCT**，把平滑轨迹压成少量低频系数；再量化、展平，用 **BPE** 把常见的系数模式合并成短 token 序列。解码时反过来：BPE 展开 → 反量化 → 逆 DCT。它不训练神经码本，是“固定频域变换 + 学出来的 BPE 词表”。

> [!tip] Eureka Moment
> 动作的冗余不在“关节维度”，而在**时间维度**：相邻时间步几乎一样。DCT 把时间冗余压掉，BPE 把符号冗余压掉；两者合起来，把 50Hz × 14 维的 700 个 bin token 压到约 30 个 token/手臂/chunk，同时保住高频细节。

## ⚡ 快速判断

| 维度 | 判断 |
|---|---|
| 核心结论 | DCT + 量化 + BPE 的动作 tokenizer；π0-FAST 自回归 VLA 在灵巧任务上匹配 diffusion π0，训练算力最高省 5× |
| 适合精读 | 做动作 tokenization、自回归 VLA、高频灵巧操作的人 |
| 可以跳过 | 只做连续 diffusion/flow、不关心离散 token 的人 |
| 落地可行性 | 高：tokenizer 开源，可当黑盒；但推理比 diffusion 慢 |
| 主要风险 | 推理延迟、固定 chunk、量化/高频细节、BPE 词表分布、flatten 顺序敏感 |

## 📍 时间线

```text
2023 RT-2：动作 = 文本 token，逐维 256 bin
  → 2024 OpenVLA：7B 开源，专用 action head + bins
  → 2025.01 FAST：DCT + BPE，频域动作 tokenization
  → 2025 π0-FAST：FAST + π0，自回归 VLA
  → 2025 FAST+：1M 轨迹训练的通用 tokenizer
```

---

## 1. 问题：为什么 naive binning 不行

### 1.1 序列太长

假设 1 秒动作、50Hz、14 维（双臂）：

```text
50 时间步 × 14 维 = 700 个 bin token
```

如果每个时间步、每个关节都单独分桶，序列长度是 `H × D`。控制频率越高、动作越长，序列越长；自回归 Transformer 的 attention 成本随序列长度增长，训练和推理都变慢。

### 1.2 相邻 token 高度重复

相邻时间步的动作几乎一样，模型很容易学到“复制上一个 token”的 shortcut，而不是真正理解动作；高频变化反而学不到。

### 1.3 高频任务失败

逐维分桶把每个关节的值量化到 256 个 bin；对快速折叠、接触、精细操作，bin 宽不够细，高频细节被量化误差吃掉。

### 1.4 澄清：256^7 不是 token 数

7-DoF、每维 256 bin：

- **一条动作的 token 数**：7 个（每关节选一个 bin）；
- **关节组合空间**：`256^7 ≈ 7.2×10^16`；
- **H 步 chunk 的 token 数**：`7H`；
- **H 步 chunk 的组合空间**：`256^(7H)`。

`256^7` 是“所有关节 bin 组合”的数量，不是模型要生成的 token 数。只有把整个 7-DoF 动作当成**一个原子 token** 时，词表才是 `256^7`；那需要 `256^7` 类 softmax，不可行。

---

## 2. FAST 的全景：一个压缩编解码器

```text
动作 chunk [H, D]
   │
   ▼
① 归一化
   │
   ▼
② 每个关节沿时间做 DCT
   → [H, D] 频率系数
   │
   ▼
③ 量化 / 截断
   → 少量低频系数，整数化
   │
   ▼
④ 展平（flatten）
   → 一维整数序列
   │
   ▼
⑤ BPE 合并高频相邻对
   → 短 token 序列
   │
   ▼
策略（自回归）学习 token 序列
   │
   ▼（推理）
⑥ BPE 展开 → 反量化 → 逆 DCT
   → 动作 chunk
```

关键点：

- **DCT 不是 tokenizer**，它是 tokenizer 内部的“表示变换”；
- **BPE 才是词表学习**；
- **量化**是有损压缩，决定保真度；
- **逆 DCT** 是 detokenizer 的最后一步；
- 整个 tokenizer 可以离线训练/构建，策略只学 token 序列。

---

## 3. DCT：把时间冗余压掉

### 3.1 DCT-II 公式

对每个关节的时间序列 `x_0..x_{N-1}`：

```text
X_k = α_k · Σ_{n=0}^{N-1} x_n · cos[ π/N · (n + 1/2) · k ]
```

- `k=0`：DC 分量，常数基函数，代表平均值/整体偏移；
- `k>0`：AC 分量，不同频率的余弦波，代表趋势和细节；
- `α_0 = 1/√N`，`α_k = √(2/N)`（orthonormal 约定）。

逆变换：

```text
x_n = Σ_{k=0}^{N-1} α_k · X_k · cos[ π/N · (n + 1/2) · k ]
```

### 3.2 为什么 DCT 适合动作

动作轨迹通常平滑、时间自相关强：

- 大部分能量集中在低频；
- 高频系数很小，甚至接近 0；
- 只保留前 `K` 个系数，就能以很小误差重建整段动作。

这就是 **energy compaction**（能量集中）。FAST 利用它，把 `H` 个时间步压成 `K` 个系数。

### 3.3 DC 分量与平均位置

例子：`x = [10.0, 11.0, 10.5, 9.5]`，`N=4`。

```text
mean = (10 + 11 + 10.5 + 9.5) / 4 = 10.25
X_0 = (1/√4) × 41 = 20.5
mean = X_0 / √4 = 10.25
```

所以 `20.5` 是 orthonormal DCT 的 DC 系数，除以 `√N=2` 才是平均值 10.25。不同库的归一化不同：

| 归一化 | DC 系数 | 平均值 |
|---|---|---|
| 不归一化 | 41 | 41/4 = 10.25 |
| orthonormal | 20.5 | 20.5/2 = 10.25 |
| 均值归一化 | 10.25 | 10.25 |

看 DCT 数值前先确认归一化约定。

### 3.4 DCT vs FFT：边界条件

FFT 假设信号首尾相接：

```text
[0, 1, 2, 3] → ... 0,1,2,3,0,1,2,3 ...
                         ↑ 3→0 跳变
```

如果首尾不相等，周期延拓会在接缝处制造假跳变；压缩截断高频后，接缝附近出现振铃/过冲（Gibbs 现象），在动作上表现为 chunk 边缘抖动。

DCT 先做偶对称镜像：

```text
[0, 1, 2, 3] → 0,1,2,3,3,2,1,0 | 0,1,2,3,...
                         ↑ 接缝 0→0，连续
```

镜像让扩展信号在边界连续，高频系数衰减更快，压缩后的边缘 artifacts 更小。这不是“DCT 强迫首尾相等”，而是“镜像后接缝连续”；原始首尾仍可不同。

对机器人控制，chunk 边缘平滑很重要：chunk 拼接/rolling 时，FFT 的边缘振铃会变成抖动，DCT 更稳。

---

## 4. 量化：决定保真度与压缩率

DCT 系数是连续值，需要量化成整数才能 BPE：

- 保留前 `K` 个低频系数；
- 每个系数按固定步长取整；
- 高频系数可以截断或粗量化。

取舍：

| 保留系数 | 保真度 | token 数 | 适用 |
|---|---|---|---|
| 多 | 高 | 多 | 精细/高频任务 |
| 少 | 低 | 少 | 平滑/低频任务 |

量化是有损的；DCT 本身可逆，量化才是误差来源。

---

## 5. 展平 + BPE：把整数序列压成 token

### 5.1 展平

一个 chunk 的量化系数形状是 `[K, D]`：

```text
K=4 个系数 × D=7 个关节 = 28 个整数
```

展平成一维序列，例如：

```text
[j1_c0, j1_c1, j1_c2, j1_c3, j2_c0, ..., j7_c3]
```

**flatten 顺序很重要**：按关节优先（joint-major）还是按频率优先（frequency-major），会让相邻 token 代表“同一关节的不同频率”或“同一频率的不同关节”，BPE 学到的模式完全不同。论文发现这会显著影响策略训练。

### 5.2 BPE

BPE（Byte Pair Encoding）的步骤：

1. 每个整数是一个初始符号；
2. 统计相邻符号对的频率；
3. 把最高频的相邻对合并成新 token；
4. 重复，直到词表达到目标大小。

小例子：

```text
原始: [42, 15, 3, 1, 0, 0, 0]

[0,0] 常见 → Z: [42, 15, 3, 1, Z, 0]
[42,15] 常见 → A: [A, 3, 1, Z, 0]
```

最终 token 序列更短。BPE 词表从大量动作数据里学出来；一个 token 可能代表“几个 DCT 系数的常见组合”，**不是**一个关节的一个 bin。

### 5.3 解码

```text
token 序列 → BPE 展开 → 量化系数 → 反量化 → 逆 DCT → 动作 chunk
```

BPE 在量化符号上是可逆的；DCT/量化才是 lossy 的部分。

### 5.4 正确的算术

- 一个 chunk：`K × D = 4 × 7 = 28` 个整数；
- 10 个 chunk：`10 × 28 = 280` 个整数；
- `10 × 7 × 4 = 280` 只有在“10 个 chunk”时成立；如果 `10` 是时间步，DCT 已经把时间步压成 4 个系数，不能再乘 10。

---

## 6. FAST+：通用 tokenizer

FAST+ 是在 **1M 真实机器人动作轨迹**上训练的通用 tokenizer：

- 目标：跨形态、跨控制频率，当黑盒使用；
- 不同机器人/频率的动作都可以映射到同一套 token 词表；
- 对下游策略，不需要为每个数据集重新训练 tokenizer。

它仍然是 DCT + 量化 + BPE，不是神经码本；区别在 BPE 词表从大规模、多样化的动作语料里学出来，泛化更好。

---

## 7. π0-FAST：自回归 VLA

π0-FAST = π0 的 VLM backbone + FAST token 自回归头。

论文报告：

- 在灵巧任务（包括叠衣服）上匹配 diffusion π0；
- 训练算力最高省 **5×**（5× fewer GPU hours）；
- 扩展到 **10k 小时**机器人数据；
- 推理比 diffusion π0 慢：v1 约 **750ms/chunk** vs diffusion 约 **100ms**（版本/硬件相关，需以论文为准）。

为什么训练快、推理慢：

- 训练：next-token cross-entropy，简单、并行；
- 推理：自回归逐 token 生成，序列越长越慢；diffusion 可以并行去噪。

---

## 8. 实验与数据

论文评测 7 个任务：

- Libero 仿真；
- table bussing（20Hz）；
- T-shirt folding（50Hz）；
- grocery bagging；
- toaster；
- laundry folding；
- zero-shot DROID tabletop manipulation（未见环境）。

关键数字：

- 典型压缩到 **约 30 token/手臂/chunk**，与原始控制频率基本无关；
- T-shirt folding：`700 → 53` tokens；
- table bussing：`140 → 28` tokens；
- 重建误差与 binning 可比；
- FAST 优于 FSQ 学习式 tokenizer 基线；
- FAST+ 用 1M 轨迹训练，可跨形态/频率。

> 注意：具体数字以论文版本为准；不同任务、chunk 长度、词表大小会变化。

---

## 9. 为什么 FAST 有效（深层解释）

### 9.1 它压的是时间冗余，不是关节组合

- 逐维 binning：`H × D` 个 token，相邻时间步高度相关；
- DCT：把 `H` 个时间步压成 `K` 个频率系数；
- BPE：把系数序列里的重复模式压成更少 token；
- 最终 token 数从 `H×D` 降到 `~30/手臂/chunk`。

### 9.2 它保留了高频细节

- binning 的 256 个 bin 是对**值**的粗量化；
- DCT 保留的是**频率**，可以用更多系数表示快速变化；
- 对折叠、接触、精细操作，高频细节不被 bin 宽吃掉。

### 9.3 它减少了 copying shortcut

- 相邻时间步 token 不再重复；
- 模型不能靠“复制上一个 token”蒙混过关；
- 必须学会频率系数到动作的映射。

### 9.4 它把 tokenizer 变成可复用的“动作词表”

- BPE 词表从大规模动作数据学出来；
- 常见动作模式成为一个 token；
- 不同任务/机器人可以共享词表（FAST+）。

---

## 10. 局限与失败模式

| 局限 | 说明 |
|---|---|
| 推理延迟 | 自回归逐 token 生成，比 diffusion 慢 |
| 固定 chunk | DCT 需要固定长度；变长/流式要分块/padding |
| 高频截断 | 系数保留太少会抹掉接触/快速修正 |
| 全局基 | 局部尖峰会摊到多个系数；短时 DCT/分窗可缓解 |
| flatten 顺序 | 敏感；选错会显著影响策略 |
| 量化误差 | DCT 可逆，但量化有损 |
| 词表分布 | FAST+ 的 1M 轨迹不覆盖你的机器人时，效果下降 |
| 重建 ≠ 控制 | 低 MSE 不代表任务成功；要看闭环成功率 |
| 无语义 | BPE token 是统计模式，不是“抓取/放置”语义 |
| 不解决组合空间 | 256^7 仍是动作组合空间；FAST 只压时间/符号冗余 |

---

## 11. 和其他 tokenizer 对比

| 方法 | 内部 | token 含义 | 序列长度 | 训练 | 高频 |
|---|---|---|---|---|---|
| 逐维 binning（RT-2/OpenVLA） | 每维 256 bin | 一个关节的一个 bin | `H×D` | 简单 | 差 |
| VQ/FSQ（VQ-VLA/OAT） | 学习码本 | 一个码字 | 少量 code | 需训练 | 中 |
| **FAST** | **DCT + 量化 + BPE** | **系数模式** | **~30/手臂/chunk** | **BPE 离线** | **好** |
| Diffusion/Flow | 连续去噪 | 连续动作 | 无 token | 端到端 | 好 |

---

## 12. 对 `fast.md` 的纠错

| 原说法 | 问题 | 正确理解 |
|---|---|---|
| “总 token 数可达 256^7” | 把组合空间当成 token 数 | 一条动作 7 个 token；`256^7` 是关节组合空间 |
| “10 × 7 × 4 = 280” | 把时间步和 chunk 混了 | 一个 chunk 28 个整数；10 个 chunk 才 280 |
| “tokenizer 用 FFT” | FAST 用 DCT | DCT 边界更平滑；FFT 有周期延拓跳变 |
| “低（2-3 个 token/序列）” | 与论文不符 | 典型约 30 token/手臂/chunk，随任务/频率变化 |
| “BPE token = 关节 bin” | 混淆 token 含义 | BPE token 是量化 DCT 系数的模式 |
| “DC 20.5 = 平均 20.5” | 归一化 | 20.5 是 orthonormal DC；平均值 10.25 |

---

## 13. 一个正确的走一遍

```text
输入：H=10 时间步，D=7 关节
DCT：每个关节 → 10 个系数
保留：K=4 个低频系数
形状：4 × 7 = 28 个整数
展平：28 个整数的序列
BPE：合并常见相邻对 → 可能 10~30 个 token
策略：自回归预测 token
解码：BPE 展开 → 反量化 → 逆 DCT → [10, 7] 动作
```

---

## 14. 面试问答

**Q1：FAST 的核心思想是什么？**

A：把动作 tokenization 当成压缩问题：DCT 压时间冗余，量化控制保真度，BPE 压符号冗余，解码逆 DCT。它不训练神经码本，是固定变换 + 学出来的词表。

**Q2：为什么不直接每个关节分 256 bin？**

A：序列太长（50Hz × 14 维 = 700 token/秒），相邻 token 重复，模型学 copying shortcut；高频细节被 bin 宽吃掉。FAST 压到约 30 token/手臂/chunk，并保留高频。

**Q3：DCT 和 FFT 的区别？**

A：FFT 假设首尾相接，首尾不一致会在接缝处产生跳变，截断高频后边缘振铃；DCT 先镜像，接缝连续，边缘 artifacts 更小，适合平滑动作。

**Q4：BPE 在 FAST 里做什么？**

A：把量化后的 DCT 系数序列中常见的相邻模式合并成新 token，缩短序列、建立动作词表；一个 token 不是关节 bin，而是系数模式。

**Q5：FAST 为什么训练快、推理慢？**

A：训练是 next-token 交叉熵，简单并行；推理要逐 token 自回归生成，序列越长越慢；diffusion 可以并行去噪。

**Q6：FAST+ 是什么？**

A：在 1M 真实机器人动作轨迹上训练的通用 tokenizer，跨形态/频率当黑盒使用；仍是 DCT + 量化 + BPE。

**Q7：FAST 的局限？**

A：推理延迟、固定 chunk、高频截断、flatten 顺序敏感、量化误差、词表分布依赖、重建不等于控制、token 无人类语义。

---

## 15. 精读建议

**值得精读**：

1. 做自回归 VLA / 动作 tokenization 的人：§3 方法、§4 实验、BPE/flatten 细节；
2. 做高频灵巧操作的人：T-shirt folding、laundry folding 的结果；
3. 做通用 tokenizer 的人：FAST+ 的 1M 轨迹训练和跨形态迁移。

**建议路径**：

- 先读摘要 + 图 1 → §3 FAST 方法（DCT/量化/BPE）→ §4 π0-FAST 实验 → §5 局限；
- 代码：`huggingface.co/physical-intelligence/fast`。

**可以跳过**：

- 只做连续 diffusion/flow、不关心离散 token 的人；
- 只关心小数据单任务、用 ACT/DP 更合适的人。

---

## 16. 参考

- 论文：https://arxiv.org/abs/2501.09747（RSS 2025）
- 项目/代码：https://huggingface.co/physical-intelligence/fast
- RSS：https://roboticsproceedings.org/rss21/p012.html
- 相关：[[20261028-Compression Gap]]、[[20261023-OpenVLA]]、[[20261022-RT-2 — VLA 概念的诞生]]、[[20261103-RDT-1B]]
