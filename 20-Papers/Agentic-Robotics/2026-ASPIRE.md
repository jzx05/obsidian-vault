---
title: "ASPIRE: Agentic Skills Discovery for Robotics"
authors: [Runyu Lu, Yubo Wu, Ethan Kou, Letian Fu, Wenli Xiao, Ajay Mandlekar, Yinzhen Xu, Guanya Shi, Ken Goldberg, Ang Chen, Mosharaf Chowdhury, Yuke Zhu, Linxi "Jim" Fan, Guanzhi Wang]
year: 2026
venue: NVIDIA Tech Report / arXiv
tags: [paper, code-as-policies, coding-agent, skill-library, continual-learning, evolutionary-search, sim-to-real, embodied-AI, LLM-agent, robotic-manipulation]
status: reading
rating: 
date-read: 2026-07-20
url: https://research.nvidia.com/labs/gear/aspire/
pdf: "C:\\Users\\huawei\\Zotero\\storage\\SXANC279\\Lu 等 - ASPIRE Agentic Skills Discovery for Robotics.pdf"
priority: high
---

# ASPIRE: Agentic Skills Discovery for Robotics

## TL;DR
> ASPIRE (**A**gentic **S**kill **P**rogramming through **I**terative **R**obot **E**xploration) 是 NVIDIA GEAR 提出的持续学习机器人系统：让编码智能体在 code-as-policy 范式下自主写、执行、诊断和修复机器人程序，并把每次验证过的失败修复方案沉淀为可复用的技能，积累到一个跨任务、跨仿真/真机、跨形态可复用的技能库。相比前作 CaP-Agent0，ASPIRE 把"每个任务从零 debug"升级为"越做越会做"，在 LIBERO-Pro、Robosuite、BEHAVIOR-1K 上大幅超越 VLA 和现有 coding agent 基线，并在真机 bimanual YAM 上初步验证 sim-to-real 技能迁移。

---

## 核心问题与动机

### 问题：机器人编码 agent 的两个结构性缺陷
1. **粗粒度反馈**：现有 code-as-policy 系统只给 agent "任务成功/失败" 这一层反馈，agent 无法定位失败根源（感知？抓取？规划？接触动力学？下游恢复？）
2. **零经验积累**：任务完成后，修复过程中发现的诊断策略和恢复方案被丢弃，第 100 个任务和第 1 个任务的 agent 一样"新手"

### 人类机器人工程师如何工作
回放执行 → 检查感知输出和运动轨迹 → 定位失败子系统 → 修改实现 → 把可复用的修复模式（抓取恢复启发式、导航策略、prompt 配方、程序化 fix）内化为长期知识。ASPIRE 明确以此为蓝本。

### 已有方法的不足
- **VLA 系列（OpenVLA / π₀ / π₀.5）**：端到端策略缺可解释性，任务扰动下鲁棒性差
- **CaP-Agent0 [[2026-CaP-X]]**：多轮 + VDM + 一次性合成的技能库，但技能库是**离线一次合成**，且反馈仍是任务级
- **软件工程 agent（Claude Code / SWE-agent）**：write-execute-debug 循环已成熟，但缺少物理执行痕迹和跨任务技能沉淀

---

## 系统架构：coordinator–actor + 三大组件

### 顶层架构
```text
Coordinator（协调器）
  │  ─────────  管理共享技能库，审计并接纳新技能
  ├─ Actor Agent 1  →  任务 1（并行学习）
  ├─ Actor Agent 2  →  任务 2
  └─ Actor Agent N  →  任务 N
```

- **Coordinator**：维护共享技能库，把 actor 分派到各任务
- **Actor Agent**：独立的 coding agent，负责单个任务的写代码 → 执行 → 诊断 → 修复
- **通信协议**：actor 之间**不交换完整聊天历史或原始 rollout**，只把结构化的 findings 上报给 coordinator，由 coordinator 提炼进技能库。这样每个 actor 的上下文窗口只专注于当前任务、当前程序、当前失败 trace

### 三大核心组件
| 组件 | 作用 | 输出 |
|------|------|------|
| **Robot Execution Engine** | 暴露 per-primitive 多模态执行痕迹（perception overlay、抓取候选、运动轨迹、collision feedback），让 agent 能够选择性检查、逐步定位、闭环验证 | 结构化 trace bundle |
| **Skill Library** | 持续增长，蒸馏 validated fix 为可复用 in-context guidance | 异构技能条目 |
| **Evolutionary Search** | 生成多样的候选程序，避免陷入局部修复循环 | 广度探索的程序种群 |

---

## 组件一：Robot Execution Engine（机器人执行引擎）

### 从"任务级反馈"到"per-primitive 多模态痕迹"

**先前方法的困境**：
- 反馈太少：隐藏失败原语，agent 无法定位
- 原始视频全塞进去：淹没 causal chain，agent 被分散

**ASPIRE 的执行引擎**为每次 primitive 调用（感知、规划、抓取、控制）都记录：
- 调用的 API 名称
- 输入、输出、返回状态
- 多模态证据：RGB 关键帧、感知 overlay、抓取候选可视化、物体位姿、运动规划结果

关键设计：**只保留 primitive 调用前后的关键帧和相关 overlay**，不塞完整视频。agent 可以像人一样"翻日志、看现场照片"，把注意力放在失败附近的证据。

### 案例：BEHAVIOR-1K 拿收音机

论文 Figure 2 详细拆了一个 navigate-and-pick-up-radio 的调试 episode：
1. **Ego view + overlay**：机器人找到收音机，但反复靠不过去
2. **Trace 定位**：`find_object_base_rotate` 成功返回 radio 位姿 → 但 `navigate_to_pose` 反复 `PLANNING_ERROR` → 检查日志发现候选目标位姿全在桌子的碰撞缓冲区内（离桌边 <20 cm）
3. **诊断**：不是感知错、也不是抓取错，而是 cuRobo 因目标位姿不可行拒绝规划
4. **修复**：写一段 multi-angle approach，绕物体尝试 5 个角度（180°、±90°、±45°），找到可行方向后重新感知抓取
5. **蒸馏**：这个 fix 被抽象为 **Multi-Angle Approach 技能**入库

> **核心**：执行引擎把"失败"从一个 0/1 信号变成一个可以 grep、可以看的"事故现场"，从而支持 agent 自主诊断和 targeted repair。

---

## 组件二：Skill Library（技能库）

### 什么是"技能"

**不是**：完整任务程序、整段策略
**是**：从 validated repair 中蒸馏出的 heterogeneous repair knowledge，每条技能包含：
1. **Problem** — 从触发它的失败 trace 中抽取的问题描述
2. **When to apply** — 检索条件 / 触发守卫（situational guard）
3. **Strategy** — 修复策略的自然语言描述
4. **Code sketch** — 可选的代表性代码片段
5. **Origin task** — 出处任务

### 六大技能类别（agent 自主发现，非人工预设）

| 类别 | 例子 |
|------|------|
| **Localization**（定位） | Multi-Object Disambiguation（front/back/left/right 语义排序）、SAM3 prompt registry、置信度过滤 |
| **Navigation**（导航） | Multi-Angle Approach（旋转靠近向量）、Waypoint Hopping、Free-Space Exploration |
| **Motion Primitives**（运动原语） | Linear Push（地面平面推物）、Pull、Turning、Handover |
| **Object-Level Grasping**（物体特化抓取） | Wine（OBB-aligned tall-cylinder grasp）、Bowl、Can、Pan handle、Box、Mug |
| **Scene Understanding**（场景理解） | OBB major/minor axis、Table-aware planning |
| **Debugging Workflow**（调试工作流） | Failure Pattern（同一 symptom 出现 ≥2 次才立库）、Trace Analysis、Negative Constraint |

### 典型技能示例

#### Multi-Object Disambiguation（定位）
- **Problem**："front bowl" / "left bowl" 对 SAM3 无意义，只会返回所有 bowl，`masks[0]` 就是错的
- **When**：场景里有 ≥2 个同类物体，且指令带方位词
- **Strategy**：按方位词隐含的坐标轴（前后 = X，左右 = Y）排序候选，然后取 index

```python
masks = sam3(rgb, "bowl")
candidates = [mask_to_world_point(m) for m in masks]
if "front" in instruction:
    candidates.sort(key=lambda p: p[0])       # 按 X
elif "back" in instruction:
    candidates.sort(key=lambda p: p[0], reverse=True)
elif "left" in instruction:
    candidates.sort(key=lambda p: p[1])       # 按 Y
target = candidates[0]
```

#### Multi-Angle Approach（导航）
- **Problem**：直接靠近物体经常 `PLANNING_ERROR`，planner 退到 >1 m 外的 waypoint
- **When**：`navigate_to_pose` 名义上成功但机器人 >1 m 于目标 / 日志出现 `PLANNING_ERROR` / 物体在墙、家具、房间边界附近
- **Strategy**：尝试 5 个角度（0°、±90°、±45°），旋转靠近向量绕过障碍

#### Bottle（物体特化抓取）
- **Problem**：酒瓶被随机 yaw 夹爪抓时，会以切向角度接触侧壁而滚出
- **When**：高瘦圆柱形物体，aspect ratio > 2.0
- **Strategy**：把夹爪 yaw 对齐 OBB 主轴，两段闭合（scout 50% → seat 70%），慢抬 (dz=0.15, speed=0.05)

#### Failure Pattern（元技能：调试工作流）
- **Problem**：单 trial 失败可能是噪声；跨 trial 同 symptom 才是模式
- **When**：≥2 个 trial 同 symptom / 失败按物体类、场景几何、动作类聚簇
- **Strategy**：按 `(symptom, applicability_class)` 分组，出现 ≥2 次的对才提为库条目候选

```python
patterns = defaultdict(list)
for trial in attributed_trials:
    key = (trial.symptom, trial.applicability_class)
    patterns[key].append(trial)
for key, trials in patterns.items():
    if len(trials) >= 2:
        propose_library_entry(symptom=key[0], applies_to=key[1], evidence=trials)
```

### 提交流程：actor 报告 → coordinator 审计
1. Actor 上报 structured findings（failure mode、validated fix、可迁移模式）
2. Coordinator 检查是否符合 API policy、是否已通过 debug 验证
3. 只有通过审计的 pattern 才被 promote 到共享库

> 论文 Appendix E.1 给出了 actor 报告和 coordinator 接纳的完整 prompt。

---

## 组件三：Evolutionary Search（进化搜索）

### 为什么需要进化搜索

Trace-guided debugging 单独可能陷入 **局部修复循环** —— agent 一直微调同一个失败策略，而不去尝试根本不同的解法。

### 算法（论文 Algorithm 1）

```text
Require: 任务 τ、初始程序 π₀、调试/验证 seed 集 D_dbg / D_val、
         技能库 L、agent A、budget (K, N)、成功阈值 s*

1: 用 π₀ 在 D_dbg 上执行，得到 (score s₀, trace bundle T₀)
2: 记为当前最优 π*
3: for k = 1..K:
4:     基于 L 和 top-3 历史种群，采样 N 个候选 {π_i}
5:     每个候选在 D_dbg 上执行，得到 (s_i, T_i)
6:     更新 π* 为得分最高者
7:     if s* ≥ threshold: break
8: 在 D_val 上验证 π* → (s_val, T_val)
9: ExtractValidatedPatterns(...) 抽取入库
10: return (π*, s_val, L)
```

- **搜索对象**：机器人程序本身（不是奖励、不是权重）
- **选择信号**：闭环执行分数
- **入库条件**：不仅 debug 通过，还要在 D_val 上跨 seed / 跨任务泛化

### 与 CaP-Agent0 并行推理的区别
| | CaP-Agent0 (+3M) | ASPIRE 进化搜索 |
|---|---|---|
| 生成模式 | 一轮 9 候选，中央模型合成 | 多轮迭代，每轮 N 候选 |
| 探索深度 | 单轮 diversity | 多轮 evolution，条件于历史 top-K |
| 上下文 | 中央模型看所有候选 | 每轮只带最好候选 + 残余失败 trace |
| 目标 | 提升单次生成的稳健性 | 广度探索 + 逐步逼近全局最优 |

---

## 实验设置

### 编码模型与 API
- **仿真**：Claude Code + **Claude Opus 4.6**（1M-token context window）
- **真机**：**OpenAI Codex GPT-5.5** in reasoning-xhigh mode
- **代码框架**：CaP-X [[2026-CaP-X]]（基于 MuJoCo Playground），感知/几何/运动规划 API 全程一致

### 三大 benchmark
1. **LIBERO-Pro**（短程 + 扰动）：Object / Goal / Spatial 三个扰动 suite，每 suite 10 任务 × 50 held-out seed。ASPIRE 在 seed 51–65 学习、seed 1–50 评估
2. **Robosuite**（接触密集 + 双臂）：Lift, Stack, Restack, Wipe, Insert, 2A-Lift, 2A-Handover。每任务 100 held-out trial
3. **BEHAVIOR-1K**（长时程移动操作）：navigate-and-pick-up-soda-can、navigate-and-pick-up-radio。ASPIRE 采用**增量式 block 执行**，逐 block 生成，基于当前多模态 trace

### 评估协议关键点
- **Disjoint debug / eval seed**
- **ASPIRE：一份程序跑所有 eval seed**（不做 per-seed 重生成，不做 test-time 重试）
- **CaP-Agent0：per-seed 重新生成 + test-time 推理和重试**（对基线更有利，但仍被 ASPIRE 大幅超过）

### Baseline
- **端到端 VLA**：OpenVLA / π₀ / π₀.5
- **Coding agent**：CaP-Agent0 [[2026-CaP-X]]
- **人类专家**：手写程序（"Human"列）

---

## 主要实验结果

### LIBERO-Pro（Figure 4a，短程扰动）

| 扰动 suite | 相对最强基线的绝对提升（Pos & Task 平均） |
|---|:---:|
| Object | **+77 pt** |
| Goal | +41.5 pt |
| Spatial | +42.5 pt |

- π₀.5 在部分位置扰动上强于 OpenVLA / π₀，但对**任务改写（task paraphrase）大面积崩掉**
- ASPIRE 在扰动下**接近饱和**，多任务甚至超过人类专家

### Robosuite（Figure 4b，接触密集）

关键突破：**双臂 Handover 从 20% 提升到 92%（+72 pt）**

其他任务（Lift / Stack / Restack / Wipe）几乎全部接近 100%。**Peg Insert 依然是最难 case**，与 CaP-Agent0 一样体现视觉+代码难以替代高频接触反馈。

### BEHAVIOR-1K（Figure 4c，长时程 + 移动操作）

| 任务 | Nav 成功率 | Task 成功率（相对 CaP-Agent0） |
|---|:---:|:---:|
| Nav & Pick up Radio | 88 / 80 → 100 | **56 → 88 (+32 pt)** |
| Nav & Pick up Soda Can | 100 / 99 → 100 | 72 → 88 (+16 pt) |

ASPIRE 在导航和任务成功率上**同时超过人类专家和 CaP-Agent0**。

### Zero-Shot 迁移（Figure 5，LIBERO-Pro Long）

**训练**在 LIBERO-90（90 个短程任务）积累技能库，**测试**在 held-out 长程任务，**不做**额外 debug、不做重试、不更新库。

| Library 大小 N | Pos 成功率 | Task 成功率 |
|---|:---:|:---:|
| N=0（空库） | 低 | 低 |
| N=25 | 上升 | 上升 |
| N=50 | 上升 | 上升 |
| **N=90（满库）** | **23%** | **38%** |
| CaP-Agent0 | 低（个位数） | 低 |
| π₀.5 | 低 | 低 |
| **对比**：CaP-Agent0 的对应 Task 只有 **4%** | | |

**关键发现**：Skill library 规模 → zero-shot 成功率**近乎单调上升**，说明短程任务发现的 repair pattern 在长程任务组合中**真的复用**。

### Sim-to-Real 迁移（Table 1，双臂 YAM 真机）

从仿真库选三个技能（soda-can pickup / bowl-on-plate / drawer push-pull），作为 in-context guidance 塞进真机 coding agent（GPT-5.5）。

| 任务 | Total Tokens w/o Skills → w/ Skills | Success Rate w/o → w/ |
|---|---|---|
| Put bowl on plate | 8.65 M → **5.11 M** | 20/20 → 20/20 |
| Lift soda can | 61.94 M → **6.58 M**（~10×↓） | 13/20 → **19/20** |
| Open/push drawer | 334.9 M → **81.7 M** | **0/20 → 11/20** |

**核心结论**：
- Bowl 已经容易，主要节省 token
- Soda can 一致性提升 + token 大幅下降
- Drawer 是"没有技能就干不成，有技能就能做"的典型
- 迁移的**不是策略**，而是**在不同 embodiment、不同 API 上都成立的失败恢复模式**

### 消融（Figure 6，LIBERO-Pro）

| 配置 | 平均成功率 |
|---|:---:|
| Base（无 Engine 无 Evo） | **14%** |
| + Robot Execution Engine | **62%**（+48 pt） |
| + Evolutionary Search | **72%**（+10 pt） |

- **Execution Engine 是最大杠杆**（+48 pt）：per-primitive trace 是让 agent 从瞎猜到定位诊断的关键
- **Evolutionary Search 处理残余硬样本**（+10 pt）：进化搜索的收益随迭代轮数 **单调但边际递减**（Figure 6c）

---

## ASPIRE vs CaP-Agent0：定位对比

| 维度 | CaP-Agent0 | **ASPIRE** |
|------|:---:|:---:|
| 反馈粒度 | 任务级 stdout/stderr + VDM | **per-primitive 多模态 trace** |
| 技能库 | 一次性离线合成、任务无关的几何工具 | **持续增长、开放异构、含调试工作流** |
| 探索方式 | 单轮 9 候选 + 中央综合 | **多轮进化搜索 + 条件采样** |
| 跨任务经验 | 无（每任务从零重推） | **共享 skill library，越做越会** |
| 每 seed 生成 | 每 seed 重生成 + 重试 | **一份程序打所有 seed** |
| Sim-to-Real | 未系统验证 | **验证 3 个技能、token 大幅节省** |
| 位置 | test-time compute scaling | **continual learning + open-ended exploration** |

**一句话**：CaP-Agent0 让固定模型"当次任务多试几遍"；ASPIRE 让 agent 在多个任务之间"越做越会"。

---

## 核心洞察

### 1. **Per-primitive trace = 让 agent 拥有"事故现场"**
从"任务 0/1 反馈"升级到"每个 API 调用的输入输出 + 关键帧 + overlay"，是从消融看出 **+48 pt** 的最大杠杆。类比软件工程：从"整个测试挂了"到"stack trace + 前后变量 dump"

### 2. **技能不是完整程序，而是可复用的失败恢复模式**
论文有意区分：**skill ≠ complete task program**。skill 是"当遇到 X 时试 Y"这种带触发条件的修复知识。这与人类工程师的经验积累最贴近

### 3. **开放式技能类别 > 预设的技能分类学**
六大类别（localization / navigation / motion / grasping / scene / debugging）不是人工先验，而是**从 agent 自己发现的修复中归纳**。甚至"如何调试"（Failure Pattern、Trace Analysis、Negative Constraint）本身也是技能，形成**元技能层次**

### 4. **进化搜索 + trace 调试的正交价值**
两者共存收益最大：trace 调试擅长"看现场想 fix"（局部），进化搜索擅长"跳出错误策略"（全局）。单独 trace 已经 62%，加进化只多 10 pt，但主要发生在**残余硬样本**上

### 5. **Sim-to-real 迁移 = code as embodiment-agnostic action space**
迁移的**不是像素→电机**，而是**"在什么条件下选什么策略"的元知识**。当仿真和真机在 API 名称上一致（`sample_grasp_pose`、`navigate_to_pose`）而 API 后端可以完全不同时，跨形态迁移 skill 是自然的

### 6. **Coordinator-Actor 架构避免 context 爆炸**
Actor 之间不交换聊天历史，只由 coordinator 通过 skill library 中转"经验"。这既是工程决策也是**信息瓶颈式的抽象强制**：只有能被写成 skill 的东西才值得跨任务保留

---

## 局限（论文 §5）

1. **不是完全自主真机 lifelong learner**：真机需要 robust success detection、安全 reset、calibration 维护，仿真里免费的东西真机里都不便宜
2. **依赖前沿 LLM**：Claude Opus 4.6 + 1M 上下文，小模型能否维持同样调试循环未验证
3. **API 是预设的**：agent 只能组合现有原语，无法自己扩 API。感知/控制能力受 API 边界限制
4. **技能库的长期治理未解决**：会有 stale / 过度具体 / 冗余 / 误导条目，零样本迁移在个别 N 值上非单调就是证据。需要更强的 retrieval / pruning / ranking / re-validation
5. **Compute 昂贵**：每任务大量 LLM 调用 + 大量仿真/真机 rollout，扩到极大任务库需要更便宜推理或更 sample-efficient 搜索

---

## 关键量化对比一览

| 数字 | 说明 |
|---|---|
| **+77 pt** | LIBERO-Pro Object 上超越最强基线 |
| **20% → 92%** | Robosuite 双臂 Handover |
| **56% → 88%** | BEHAVIOR-1K 拿收音机 |
| **4% → 31%** | LIBERO-Pro Long 零样本任务扰动 |
| **14% → 62% → 72%** | Engine / Evo 消融 |
| **61.94M → 6.58M** | 真机 Lift soda can 的 total tokens |
| **0/20 → 11/20** | 真机 Open/push drawer 成功率 |

---

## 与 [[2026-CaP-X]] 的关系

ASPIRE 直接建立在 CaP-X 之上：
- **共享底盘**：CaP-X 提供 MuJoCo Playground 环境、API、CaP-Agent0 作为基线
- **代际关系**：CaP-Agent0 → ASPIRE 是"离线一次合成的技能库" → "持续增长的开放技能库" + "任务级反馈" → "per-primitive trace" + "单轮并行采样" → "多轮进化搜索"
- **实验对比**：ASPIRE 在 CaP-X 的原生 benchmark（Robosuite）上把 CaP-Agent0 的短板（双臂 Handover 20%）补到 92%

**方法论意义**：CaP-X 给出了"编码 agent 需要什么样的评测基础设施"，ASPIRE 给出了"编码 agent 如何真正持续变强"。前者是**能力基线**，后者是**学习范式**。

---

## 我的思考

- [ ] **技能库 = LLM 时代的机器人技能图谱**？如果技能条目足够多、检索足够准，未来的"机器人编程"是否就变成"给定任务 → 检索合适技能 → 组合"的过程？这实质上是把机器人程序合成变成 in-context learning + 组合优化
- [ ] **元调试技能**（Failure Pattern / Trace Analysis）能否迁移到非机器人领域？—— 这些东西本质上是"软件调试哲学"，可能在通用 SWE agent 上也成立
- [ ] **Coordinator 的审计只依赖 API policy 检查**，如果技能之间有语义冲突（同一 trigger 两条 fix 但方向相反）如何解决？论文提到但没深入讨论 retrieval / ranking
- [ ] **进化搜索的种群多样性**：Algorithm 1 里 top-3 采样保留最多 3 条历史，是否会陷入局部最优？和 novelty search / MAP-Elites 结合会不会更强？
- [ ] **Sim-to-real 只验了 3 个技能**，且都是相对短程操作。长程移动 + 双臂协作的技能能否迁移是开放问题
- [ ] **和 [[2026-Qwen-RobotManip]] 类 VLA 的关系**：ASPIRE 展示了 code-as-policy + skill accumulation 的强大扩展性，但依然依赖强 LLM。未来可能是"VLA 做低层执行 + skill library 做高层调度"的混合系统

---

## 相关论文

- [[2026-CaP-X]] — 前作，提供环境、API 和 CaP-Agent0 基线
- Voyager (Wang et al., 2023) — Minecraft 中的开放式技能库，ASPIRE 引用其为 skill-library 谱系源头
- SWE-agent (Yang et al., 2024) — 软件工程 agent 的 write-execute-debug 循环，ASPIRE 的方法论直接借鉴
- Eureka (Ma et al., 2024a) — LLM 生成奖励函数，与 ASPIRE 的 "LLM 直接生成程序 + validate" 对比
- Code as Policies (Liang et al., 2023) — code-as-policy 谱系源头
- LIBERO / LIBERO-Pro / BEHAVIOR-1K / Robosuite — 三大 benchmark 的原文
- AlphaEvolve (Novikov et al., 2025) — LLM 驱动的进化式代码搜索，ASPIRE 进化搜索的近邻工作

## 引用与延伸阅读
- 项目页：https://research.nvidia.com/labs/gear/aspire/
- 论文重点附录：A（技能库详细条目）、E.1（actor/coordinator prompt）、E.4（进化搜索 pipeline 细节）
