---
title: "What Matters in Orchestrating Robot Policies: A Systematic Study of Hierarchical VLA Agents"
authors:
  - Jiaheng Hu
  - Mohit Shridhar
  - Caden Lu
  - Dhruv Shah
  - Hao-Tien Lewis Chiang
  - Jie Tan
  - Annie Xie
year: 2026
venue: arXiv (Google DeepMind)
tags:
  - paper
  - VLA
  - hierarchical-policy
  - robot-manipulation
  - VLM
status: read
rating: 4
date-read: 2026-07-22
url: https://jiahenghu.github.io/hi-vla
pdf: "[[Hu 等 - 2026 - What Matters in Orchestrating Robot Policies A Systematic Study of Hierarchical VLA Agents.pdf]]"
---

## TL;DR
> 系统研究分层 VLA(Hi-VLA)系统的关键设计选择,用 options 框架把不同实现统一起来,发现 VLM 的**思考(reasoning)能力**、VLA 的**指令跟随能力**、以及**终止条件、观测表示、跨 episode 记忆**是决定整体性能的关键;三者组合起来,可以显著超过 flat VLA 和朴素分层。

## 问题与动机
- **要解决的问题**:现有分层 VLA(高层 VLM 规划 + 低层 VLA 执行)在实现上各不相同——用什么 VLM、什么 VLA、如何切换控制权、如何组织观测和记忆,都没有统一的设计原则。缺乏对"什么组件真正重要"的系统性理解。
- **为什么重要**:单体(flat) VLA 在长程和抽象推理任务上受限——一是训练数据大多是短片段轨迹,二是 VLM 微调到 action 会灾难性遗忘掉推理能力。分层是缓解这些问题的自然思路,但要真正把分层做好,需要知道每个组件的作用。
- **已有方法的不足**:Pi-0.5、Hi-Robot、Gemini Robotics 1.5、HAMSTER、Helix、G0、RoboOS 等各自采用不同实现,难以横向对比,也不清楚哪种设计更适合哪类任务。

## 核心方法
### 统一控制回路(基于 options 框架)
把 Hi-VLA 视为一个统一的分层策略:

$$\pi_{\text{HiVLA}}(a \mid [o_i]_{i\le t}, I) = \int_l \pi_{\text{VLA}}(a\mid o_t, l)\, \pi_{\text{VLM}}(l\mid o, I)\, dl,\quad o = \text{mem}([\phi(o_i)]_{i\le t})$$

- $\pi_{\text{VLA}}$:低层 intra-option 策略,把语言指令 $l$ + 观测映射到动作
- $\pi_{\text{VLM}}$:高层 option 选择策略,基于观测 $o$ 和任务 $I$ 生成语言子目标 $l$
- $\text{mem}(\cdot)$:记忆模块,处理历史观测
- $\phi(\cdot)$:观测表示模块,对原始图像做后处理
- $\beta(o, t)$:终止条件,决定何时把控制权交还 VLM

VLM 频率远低于 VLA,同一个 $l$ 会被 VLA 反复执行直到 $\beta$ 触发。

### 五个被解剖的组件
1. **高层 VLM 策略**(Gemini 2.5 Lite / Flash / Pro,是否开启 thinking)
2. **低层 VLA 策略**(GROD 1B、1B+sim 微调、3B)
3. **终止条件**(固定频率 / 成功检测器 / VLM 自估执行时长)
4. **观测表示**(纯图像 / +文本描述 / +bounding box / +接触先验)
5. **记忆机制**(1/3/5 步窗口 / 全 episode / 本 episode 摘要 / 跨 episode 摘要)

### 评测设置
- MuJoCo ALOHA 仿真 + 真实 ALOHA 机器人
- 任务分三类:short-horizon、long-horizon、reasoning——分别对应原子执行、技能组合、语义推理三种能力
- 每类 5 个任务、每任务 200 次独立试验

## 实验与结论
### 各组件的关键发现
- **VLM**:开启 thinking(推理)带来一致提升,尤其对 long-horizon 任务(Flash-Lite 提升 +9.5 pp,long-horizon)。**模型规模影响不大**——Lite、Flash、Pro 在开启 thinking 后表现相近,现有基准分数并不能直接预测 Hi-VLA 性能。
- **VLA**:低层模型规模影响很大(GROD-3B 显著优于 1B)。**用 sim 数据微调 1B 反而变差**(long-horizon 从 41% 掉到 7.5%),因为微调破坏了 VLA 对指令重述的鲁棒性(steerability)。
- **终止条件**:成功检测器 > 固定频率 > VLM 自估。成功检测器对 10% 的检测错误依然稳健;假阳性错误比假阴性更伤性能;固定频率时,4–8 秒执行时长是较好的折中。
- **观测表示**:在图像之外补充文本描述、bounding box、甚至接触先验,能显著提升性能——即使图像已经"包含所有信息"。这印证了 VLM 在任务变难时会忽略图像的现象。
- **记忆**:**本 episode 内的原始或摘要记忆几乎没用**;但**跨 episode 摘要**(从多次 rollout 中蒸馏出 affordance)带来一致正向提升(short/long/reasoning 均 +7 pp 左右)。

### 组合结果(Table 1)
| 系统 | Short-Horizon | Long-Horizon | Reasoning | Real ALOHA |
| --- | --- | --- | --- | --- |
| Best Hierarchy | 78.2% | 67.1% | 80.9% | 12/15 |
| Naive Hierarchy | 69.6% | 40.6% | 66.5% | 9/15 |
| Flat VLA | 69.6% | 25.3% | 50.9% | 3/15 |

- 朴素分层就已优于 flat,但**用好的组件组合(cross-episode memory + thinking VLM + 接触先验观测 + 成功检测终止)可以带来数量级差别**,尤其在 long-horizon 和 reasoning 上。
- 真实 ALOHA 上验证结论迁移:best hierarchy 完成 12/15 果实归类,flat VLA 只有 3/15。
- 附录 A 用"完美脚本 VLA"做反事实实验:即使低层几乎完美,拿掉某个分层组件性能会从 95% 掉到接近 0%,说明分层设计不会因低层进步而失效。

## 我的思考
### 优点
- **框架优雅**:options 视角把混乱的 Hi-VLA 设计空间收敛到 5 个可控组件,便于消融。
- **发现有反直觉性**:VLM 大小不重要、但 thinking 很重要;in-episode 记忆没用、但跨 episode 摘要有用;sim 微调 VLA 反而变差。这些都是社区容易踩坑的地方。
- **仿真+真机双验证**,结论可信度高。

### 局限
- 静态环境假设,没考虑延迟敏感和动态场景——高层 VLM 频繁调用在真实系统中开销很大,论文只给了 4-8 秒的经验值。
- Success detector 的错误在这里是独立采样的;真实检测器往往在相邻状态间高度相关,一次假阴性可能长时间卡住命令,这个问题被作者自己指出但未量化。
- 所有 VLM 都是 Gemini 家族,结论对其他 VLM(GPT、Claude、Qwen-VL 等)是否成立未验证。
- 跨 episode 摘要目前只是 prompt 化的 affordance;更强的方式(VLM RL / SFT)是明确的 future work,但没有做。

### 可以延伸到哪里?
- **跨 episode 学习**是最有潜力的方向:能不能把 rollout 里成功/失败的经验蒸馏为 VLM 后训练信号?对应文献 [34](作者自己的 continual RL 工作)。
- **VLA steerability 保持**是另一条重要线:sim 微调导致指令跟随退化,如何在保持 steerability 的同时注入领域知识?
- **观测表示**里"接触先验带来最大增益"暗示 spatial understanding 是瓶颈,可能需要专门的 3D-aware VLM 或额外传感器。
- 关联笔记:[[2026-ASPIRE]]、[[WAM]]、[[world-models]]

## 深度解读
详见 [[2026-Hi-VLA-Deep-Read]]——按论文原顺序逐节展开,包含每个组件的完整消融表格、反直觉点的解释、以及批判性评价。

## 引用与延伸阅读
- Pi-0.5(Physical Intelligence, 2025)—— 最直接的对标 Hi-VLA 系统
- Hi-Robot(Shi et al., 2025)—— 分层指令跟随
- HAMSTER(Li et al., 2025)—— 层级动作模型
- RT-H(Belkhale et al., 2024)—— 语言动作层次
- Gemini Robotics 1.5(2025)—— thinking + motion transfer
- Options 框架:Sutton, Precup & Singh (1999)
- VLA-OS(Gao et al., 2025)—— 类似方向的规划表示研究
