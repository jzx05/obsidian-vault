---
title: "Hi-VLA 深度解读"
tags:
  - paper-deep-read
  - VLA
  - hierarchical-policy
  - robot-manipulation
related:
  - "[[2026-Hi-VLA-Orchestration]]"
date: 2026-07-22
---

> 主笔记见 [[2026-Hi-VLA-Orchestration]]。这一篇按论文原始逻辑走一遍,每一节都点明 **做了什么 → 发现什么 → 为什么值得关注**。

## 0. 一句话把握全篇

**这不是"我们做了个新系统"的论文,是"我们把这类系统的设计空间划清楚,然后一格一格做消融"的论文。**

它把 Pi-0.5、Hi-Robot、HAMSTER、Helix 等看起来五花八门的分层 VLA 收敛到一个 **5 参数的统一控制回路**,然后用一系列受控实验回答"哪个参数值得投入研发精力"。

工程价值 > 学术新颖性。这类论文在一个方向刚开始爆发时最有价值——避免所有人各自重复踩坑。

## 1. 为什么要分层

单体(flat) VLA 有两个结构性问题:

**问题一:训练数据分布 vs. 部署任务分布不匹配**
- VLA 训练数据基本是"短片段、单步操作"的遥控示教(pick、place、pour)
- 但用户想让机器人干的是"摆桌子""做早餐"这种长程组合任务
- 直接让 VLA 端到端搞定,就像让一个只学过单词的人去写论文

**问题二:VLM → VLA 的微调会破坏推理能力**
- VLA 是把 VLM 拿过来在 action 数据上 finetune 而成
- 这个 finetune 是灾难性遗忘的重灾区(catastrophic forgetting)
- 微调完的 VLA 会**丢掉原 VLM 的语义组合和抽象推理能力**

分层的核心 idea 就是**分工**:
- 高层 VLM(保留原始推理能力)→ 负责"想"
- 低层 VLA(微调过、擅长物理执行)→ 负责"做"
- 中间用**自然语言**做接口——通用、可解释、开销可控

这个 idea 早就有了(SayCan, Ahn et al. 2022),但每家实现都不一样,导致没人能说清哪些设计选择是本质、哪些是偶然。

## 2. 统一控制回路(§3,关键理论贡献)

作者观察到:**所有 Hi-VLA 都能被 options framework(Sutton & Precup 1999)套住**。

**Options framework 是什么**:强化学习里的"半马尔可夫决策过程",一个 option 由三样东西定义——
- $I$:启动条件(什么时候可以选这个 option)
- $\pi$:intra-option policy(选中后怎么执行)
- $\beta$:终止条件(什么时候结束这个 option)

对照到 Hi-VLA:

| Options | Hi-VLA 里对应 |
| --- | --- |
| Option 选择策略 | 高层 VLM:$\pi_{\text{VLM}}(l \mid o, I)$,输出语言指令 $l$ |
| Intra-option policy | 低层 VLA:$\pi_{\text{VLA}}(a \mid o_t, l)$,输出动作 |
| Option 集合 | 所有可能的自然语言指令 $\mathcal{L}$ |
| 终止条件 $\beta$ | 决定何时把控制权交还 VLM |

再加两个 Hi-VLA 特有的模块:
- **记忆** $\text{mem}(\cdot)$:处理历史观测
- **观测表示** $\phi(\cdot)$:对原始图像后处理

整个策略写成一个积分:

$$\pi_{\text{HiVLA}}(a \mid [o_i]_{i\le t}, I) = \int_l \pi_{\text{VLA}}(a \mid o_t, l)\,\pi_{\text{VLM}}(l \mid o, I)\,dl$$

$$o = \text{mem}([\phi(o_i)]_{i \le t})$$

**这个式子的重要性不在数学,在于把设计空间显式化**——五个可以独立替换的函数(VLM、VLA、β、mem、φ)正好对应后面 §4 的五个消融小节。

**关键频率不对称**:VLM 慢(推理开销大,秒级),VLA 快(控制环,10-100Hz)。同一条 $l$ 会被 VLA 反复执行,直到 $\beta$ 触发才切。

## 3. 实验设置

**环境**:MuJoCo ALOHA suite(仿真)+ 真实 ALOHA 双臂(验证)
- 选仿真是因为可以并行跑 200 trials × 5 tasks × 6 setups = 6000 次评估拿到统计显著性;真机做 sanity check。

**任务分三类**,每类对应一个想验证的能力:

| 类别 | 长度 | 例子 | 考验什么 |
| --- | --- | --- | --- |
| Short-horizon | 与 VLA 训练轨迹相当 | "Put the banana in the bowl" | 原子执行 |
| Long-horizon | 明显更长 | "Put the banana in the bowl and the mug on the plate" | 技能组合 |
| Reasoning | 需要间接推理 | "Put the object you pour coffee in on the plate"(→ mug) | 语义理解 |

**这个三分很关键**——它让作者能回答"这个设计选择帮到了哪种能力"而不是"整体好不好"。

**默认配置**(消融某个组件时,其他保持不变):
- VLM = Gemini 2.5 Flash + thinking
- VLA = GROD 3B(只用真机数据)
- β = 固定频率 8 秒
- φ = 图像 + 接触先验
- mem = 窗口 3,无摘要

## 4. 五个组件的消融

按"发现的意外程度"排序,而非论文顺序。

### 4.1 VLM 的规模 vs. 推理能力(§4.2)—— 最反直觉

**实验**:Gemini 2.5 Lite / Flash / Pro,各自 with/without thinking(Pro 强制开)。

**结果**:

| VLM | Short | Long | Reasoning |
| --- | --- | --- | --- |
| Flash-Lite | 70.5 | 48.7 | 58.5 |
| **Flash-Lite (thinking)** | 74.4 | 58.2 | 75.2 |
| Flash | 72.6 | 47.0 | 71.8 |
| **Flash (thinking)** | 75.8 | 52.4 | 72.6 |
| **Pro (thinking)** | 70.1 | 53.1 | 74.4 |

**两个惊人发现**:
1. **开 thinking 全面碾压不开 thinking**,long-horizon 上 Lite 提升 +9.5pp,reasoning 上 +16.7pp
2. **模型规模几乎没影响**——Lite (thinking) 和 Pro (thinking) 表现相近,Pro 在 long-horizon 上甚至输给自己的 Flash 版

**为什么**:
- Hi-VLA 场景下 VLM 主要在"整合场景信息 + 推断下一步该说什么话",是**推理密集**而非知识密集
- 大模型多的是**世界知识**,但摆碗摆桌不需要多深世界知识
- 而"多步 self-refine 优化输出"的 thinking 机制带来的收益,和模型规模关系不大

**实践含义**:
- 别再上来就 pick 最贵的 VLM,Lite 级别 + thinking 就够
- 现在评 VLM 用的那些 benchmark(MMLU、GPQA 之类)**不能预测 Hi-VLA 性能**——机器人社区可能需要自己的 VLM 评测集
- 这也解释了为什么 Pi-0.5 用小 VLM(PaliGemma 3B)也能 work

### 4.2 VLA 微调的陷阱(§4.3)—— 反直觉第二名

**实验**:GROD-1B、GROD-1B(用 sim 数据微调)、GROD-3B。

**结果**:

| VLA | Short | Long | Reasoning |
| --- | --- | --- | --- |
| GROD-1B | 63.4 | 41.3 | 66.9 |
| **GROD-1B + sim FT** | **54.6** | **7.5** | **43.0** |
| GROD-3B | 75.8 | 52.4 | 72.6 |

**震惊点**:用**同一个仿真环境**的示教数据微调,long-horizon 性能从 41% 掉到 7.5%,**掉了 5 倍不止**。这在"用领域数据微调更好"的直觉下是反的。

**为什么**:
- Sim FT 让 VLA 在 sim 里跑得更准,但**破坏了它对语言指令重述(paraphrase)的鲁棒性**
- 高层 VLM 生成的指令措辞五花八门("pick up X" / "grab X" / "get the X")
- 微调后的 VLA 只认训练数据里那几种固定说法,一变就懵
- 论文把这个叫 **steerability**(可控性/听话程度)

**实践含义**:
- **低层 VLA 越大越好**(instruction following 更强)——和高层 VLM 完全相反
- 微调 VLA 时必须**特别小心保持 steerability**,不能光看 sim 任务成功率
- Pi-0.5 团队后续的 Pi-0.7 标题就是 "steerable generalist robotic foundation model",可见这个问题已经成为共识

### 4.3 终止条件(§4.4)—— 工程细节里藏着大差别

**三种 β 的实现**:

| 方法 | 机制 |
| --- | --- |
| Fixed frequency | 每 $T$ 秒切一次 |
| Success detector | 另一个 VLM(拿到 privileged state)判断"完成了没" |
| VLM self-estimate | VLM 出命令时同时估个"预期执行秒数" |

**结果**:

| β | Short | Long | Reasoning |
| --- | --- | --- | --- |
| VLM self-estimate | 72.2 | 43.5 | 72.3 |
| Fixed (T=400 steps ≈ 8s) | 75.8 | 52.4 | 72.6 |
| **Success detector** | **74.7** | **57.4** | **80.9** |

**发现**:
- **Success detector 最好**,尤其 reasoning +8.3pp
- **VLM self-estimate 最烂**——因为 VLA 是随机的,提前预测执行时长不可行
- Short-horizon 对 β 不敏感(只有一步命令,做完就停)

**两个补充实验**:
1. **执行时长 sweep(Fig. 4)**:2s / 4s / 8s / 14s / 20s。太长会因为 VLA 卡住浪费步数,太短会 VLM 频繁调用变贵。**4-8 秒是甜点**。
2. **Success detector 有错怎么办(Fig. 5)**:注入 10% / 30% / 50% 的 false positive / false negative
    - **10% 错误几乎不影响甚至微升**——鲁棒性很好
    - **False positive 比 false negative 更伤**——因为会让 VLM 提前推进到下一步,导致连锁失败
    - 作者自己指出:实验里的错误是**独立采样**的,真实场景下检测错误在相邻状态间高度相关,一次 false negative 可能持续很长时间——这个问题没做量化

**实践含义**:
- 想做 Hi-VLA,**优先投资一个还算靠谱的 success detector**(错到 10% 都没事)
- 如果做不到,fixed 8s 是安全兜底
- 别信 VLM 自己估执行时长

### 4.4 观测表示(§4.5)—— "图像已经含全部信息"是错觉

**四种 φ**:

| φ | 内容 |
| --- | --- |
| Image only | 只给图 |
| Image + description | 让 VLM 先描述图,再送给决策 VLM |
| Image + description + bbox | 加上物体的 2D 包围框 |
| **Image + description + contact** | 加上"哪些物体互相接触"的仿真先验 |

**结果**:

| φ | Short | Long | Reasoning |
| --- | --- | --- | --- |
| Image | 67.6 | 38.8 | 69.2 |
| + Description | 67.9 | 35.7 | 62.8 |
| + BBox | 73.9 | 47.9 | 68.5 |
| **+ Contact** | **75.8** | **52.4** | **72.6** |

**关键发现**:
- **图 + 文本描述反而略降**(long-horizon 从 38.8 掉到 35.7)——朴素让 VLM 描述图会引入幻觉噪声
- **BBox 提升 +9pp**(long-horizon)——因为 VLM 空间理解本身弱,给它显式坐标
- **Contact 又提升 +5pp**——直接告诉它"gripper 现在在碰红方块",省去 VLM 猜测

**为什么图不够**:
- OpenEQA 的发现:**任务变难时 VLM 越来越倾向忽略图像输入**,只靠文字推理
- 补一段结构化文本能"逼"它看清场景

**尴尬的实践含义**:
- Contact 信息在 sim 里免费,**真机没有**——只能靠 tactile sensor 或视觉估计
- BBox 是真机也能拿的(VLM 自己就能出 detection)——这是最实用的可迁移收益
- 长期看这暗示:**VLM 的 spatial understanding 需要专门加强**(3D-aware VLM、depth、tactile fusion)

### 4.5 记忆(§4.6)—— 唯一"传统 agent 常识"失灵的地方

**A. 原始记忆窗口**:1步 / 3步 / 5步 / 全 episode

| 窗口 | Short | Long | Reasoning |
| --- | --- | --- | --- |
| 1 | 76.8 | 59.9 | 74.3 |
| 3 | 75.8 | 58.2 | 72.6 |
| 5 | 76.1 | 57.8 | 72.2 |
| Full | 76.5 | 59.0 | 72.8 |

**几乎完全平坦**——历史长度不影响。

**B. 摘要**:无摘要 / 上一步摘要 / 本 episode 摘要 / **跨 episode 摘要**

| 摘要 | Short | Long | Reasoning |
| --- | --- | --- | --- |
| None | 75.8 | 52.4 | 72.6 |
| Last step | 74.6 | 52.6 | 72.8 |
| Current episode | 71.7 | 50.1 | 75.7 |
| **Previous episodes** | **79.5** | **60.0** | **80.3** |

**跨 episode 摘要三个任务类别都涨 +7pp 左右**,是全篇除了 VLM thinking 之外最大的单点提升。

**为什么本 episode 记忆没用**:
- 本 episode 里 VLM 主要看到的是"这条指令没完成"的失败信号
- 一条失败不够 VLM 归纳出"应该改说啥"的规律,反而可能误导
- 本质上 VLM 缺乏 **in-context RL** 的能力——同一 episode 里边试边学做不到

**为什么跨 episode 有用**:
- 从 10 个 rollout 里蒸馏"哪些指令 VLA 能做、哪些不能",本质上是**离线归纳出 VLA 的 affordance**
- VLM 拿到这个 affordance 画像后,以后自动避开难指令、改用好指令

**这是全篇最有 vision 的发现**——它指向 continual learning + Hi-VLA 的结合,即 [[Simple-Recipe-Works-Continual-VLA-RL]] 那条路。

## 5. 组合起来(§4.7)

拿五个组件的最佳选择组一个 "Best Hierarchy":
- VLM = Flash + thinking
- VLA = GROD-3B
- β = success detector
- φ = 图 + 描述 + contact
- mem = 跨 episode 摘要

对比朴素分层(全部选最简单的实现)和 flat VLA(纯 VLA、无 VLM):

| 系统 | Short | Long | Reasoning | Real ALOHA |
| --- | --- | --- | --- | --- |
| Flat VLA | 69.6 | 25.3 | 50.9 | 3/15 |
| Naive Hierarchy | 69.6 | 40.6 | 66.5 | 9/15 |
| **Best Hierarchy** | **78.2** | **67.1** | **80.9** | **12/15** |

**几个关键点**:
- **Short-horizon 上 flat 和 naive 打平**(69.6% 对 69.6%)——因为短任务本来就是 VLA 训练数据的分布,不需要分层
- **Long-horizon 上朴素分层就已经 +15pp**,best 再叠 +27pp
- **真机结果吻合**:12/15 vs 3/15,4 倍差距,证明结论不是仿真人为

Fig. 7 展示了一个特别有意思的真机场景:第 6 步机器人放错了果实,**分层系统能识别错误并回滚重来**——这是 flat VLA 完全做不到的。这就是分层的杀手锏:**failure recovery**。

## 6. Appendix A —— "VLA 变强会不会让分层变没用"

作者预见到一个反驳:"你这些结论都是因为 VLA 不够好,等 VLA 强了这些分层技巧不就没用了?"

于是做了个巧妙的反事实实验:
- 用**脚本策略**(拿 privileged sim state 直接算动作)代替 VLA——**近乎完美的低层控制器**,只要指令清晰就 100% 执行成功
- 然后测 4 种分层配置

结果(Fig. 8):
- **完整 Hi-VLA + 完美 VLA:~95% 成功率**
- **去掉任何一个组件(观测表示 / 记忆 / 用朴素分层):性能塌到接近 0%**

**结论**:即使低层完美,高层如何组织信息和切换控制**依然决定了整体成败**。分层设计不会因低层进步而失效——反而随着 VLA 越强,分层的收益越集中在高层设计上。

这个反事实实验挺加分——它把结论从"当前 VLA 时代"推广到了"未来任何 VLA 时代"。

## 7. 批判性评价

### 论文的优点
1. **框架优雅**:options 视角把 mess 收敛到 5 个可控组件,后续同类研究几乎绕不开这个抽象
2. **反直觉发现密集**:VLM 大小无关、VLA 微调变差、in-episode 记忆无用——每一个都是社区易踩坑
3. **仿真 + 真机双验证**:结论可信度高
4. **反事实实验**:证明结论具有时间稳健性

### 论文的局限
1. **静态环境**:所有任务都是桌面静态操作,dynamic scenes(移动物体、人机协作)没考虑
2. **忽略延迟**:高层 VLM 每 8 秒调用一次,在真实部署里网络/推理开销是硬约束,论文没讨论
3. **Success detector 用了 privileged state**:真实场景不会有,得靠视觉,准确率显著下降
4. **VLM 家族单一**:全 Gemini,GPT/Claude/Qwen-VL 是否成立不清楚
5. **跨 episode 摘要只做了 prompt 层面**:最有 vision 的方向反而只做了最浅的实现

### 这篇论文在 Hi-VLA 研究图景里的定位
- **不是 SOTA 论文**——没打榜,没提新架构
- **是 "field-shaping" 论文**——给整个方向定了词汇表和评估协议
- 类似 CV 里 Li et al. 2024 "What matters in building VLA models" 的 Hi-VLA 版本
- 这种论文的引用量往往 1-2 年后才起来,但一旦起来就是长期高引

## 8. 三件事总结

如果只记住三件事:
1. **VLM 要开 thinking,但不用挑大的**;VLA 要挑大的,但**别乱微调**
2. **Success detector 是最值得投入的工程**,4-8 秒 fallback 也够用
3. **跨 episode 摘要是唯一有效的记忆**——顺着这条路走下去就是 continual RL,见 [[Simple-Recipe-Works-Continual-VLA-RL]]

## 关联
- 主笔记:[[2026-Hi-VLA-Orchestration]]
- 延伸阅读:[[Simple-Recipe-Works-Continual-VLA-RL]](Hu 等 2026,同作者的 continual RL 工作)
