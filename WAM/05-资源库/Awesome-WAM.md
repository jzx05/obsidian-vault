---
title: Awesome-WAM 资源库整理
date: 2026-06-04
---

# 📚 Awesome-WAM 资源库

> World Action Models (WAM) 相关资源的完整整理

---

## 📖 目录

- [论文和研究](#论文和研究)
- [代码实现](#代码实现)
- [数据集](#数据集)
- [工具和框架](#工具和框架)
- [博客和教程](#博客和教程)
- [相关领域](#相关领域)
- [贡献指南](#贡献指南)

---

## 论文和研究

### 核心论文

#### DreamZero: World Action Models are Zero-shot Policies
- **作者**: Seonghyeon Ye et al.
- **发表**: 2026年2月
- **链接**: 
  - [arXiv](https://arxiv.org/abs/2602.15922)
  - [OpenReview](https://openreview.net/forum?id=cd33uUB609)
  - [项目网站](https://dreamzero0.github.io/)
- **关键词**: Vision-Language-Action, 视频扩散模型, 零样本泛化, 机器人控制
- **核心贡献**:
  - World Action Models (WAM) 概念
  - 联合视频-动作预测
  - 物理先验约束
  - 7Hz 实时机器人控制

### 相关论文

#### Vision-Language-Action (VLA) 模型

- **Open X-Embodiment: Robotic Learning Datasets and RT-X Models**
  - 链接: [arXiv](https://arxiv.org/abs/2310.08864)
  - 关键词: 多具身学习, 大规模数据集

- **PaLM-E: An Embodied Multimodal Language Model**
  - 链接: [arXiv](https://arxiv.org/abs/2303.03378)
  - 关键词: 多模态语言模型, 机器人控制

- **Gato: A Generalist Agent**
  - 链接: [arXiv](https://arxiv.org/abs/2205.06175)
  - 关键词: 通用代理, 多任务学习

#### 视频预测和世界模型

- **Latent World Models For Intrinsically Motivated Exploration**
  - 关键词: 世界模型, 内在动机

- **Video Prediction by Watching Unlabeled Video**
  - 关键词: 无监督视频预测

- **Dreamer: Scalable Belief Exploration with World Models**
  - 关键词: 梦想家算法, 强化学习

#### 扩散模型在机器人中的应用

- **Diffusion Policy: Visuomotor Policy Learning via Action Diffusion**
  - 链接: [arXiv](https://arxiv.org/abs/2303.04137)
  - 关键词: 扩散策略, 视觉运动控制

- **Consistency Models**
  - 链接: [arXiv](https://arxiv.org/abs/2303.01469)
  - 关键词: 一致性模型, 快速采样

#### 零样本和少样本学习

- **Learning to Generalize to Unseen Domains via Adversarial Training with Domain Mixup**
  - 关键词: 域泛化, 零样本学习

- **Few-Shot Adaptation of Pre-Trained Networks for Imitation Learning**
  - 关键词: 少样本适应, 模仿学习

#### 具身 AI 和转移学习

- **Embodied AI with Implicit Representations**
  - 关键词: 具身 AI, 隐式表示

- **Cross-Embodiment Transfer Learning**
  - 关键词: 跨具身迁移, 知识转移

---

## 代码实现

### 官方实现

#### DreamZero
- **GitHub**: [dreamzero0/dreamzero](https://github.com/dreamzero0/dreamzero)
- **描述**: DreamZero 论文的官方实现
- **特点**:
  - 14B 自回归视频扩散模型
  - 7Hz 实时机器人控制
  - 零样本泛化演示
  - 跨具身迁移示例
- **依赖**: PyTorch, Transformers, Diffusers

### 相关实现

#### Diffusion Policy
- **GitHub**: [columbia-ai-robotics/diffusion_policy](https://github.com/columbia-ai-robotics/diffusion_policy)
- **描述**: 扩散策略的实现
- **特点**: 视觉运动控制, 多任务学习

#### Open X-Embodiment
- **GitHub**: [google-deepmind/open_x_embodiment](https://github.com/google-deepmind/open_x_embodiment)
- **描述**: 多具身学习框架
- **特点**: 大规模数据集, 统一接口

#### Dreamer
- **GitHub**: [danijar/dreamer](https://github.com/danijar/dreamer)
- **描述**: 梦想家算法实现
- **特点**: 世界模型, 强化学习

### 工具库

#### Hugging Face Transformers
- **描述**: 预训练模型库
- **用途**: 加载和使用预训练的视觉和语言模型

#### Diffusers
- **描述**: 扩散模型库
- **用途**: 实现和使用扩散模型

#### Jax/Flax
- **描述**: 高性能机器学习框架
- **用途**: 高效的模型训练和推理

---

## 数据集

### 机器人学习数据集

#### DROID (Distributed Robot Interaction Open Dataset)
- **描述**: 大规模机器人交互数据集
- **规模**: 100万+ 演示
- **特点**: 多具身, 多任务
- **链接**: [官方网站](https://droid-dataset.github.io/)

#### Open X-Embodiment Dataset
- **描述**: 多具身机器人数据集
- **规模**: 1000万+ 轨迹
- **特点**: 统一格式, 多机器人
- **链接**: [数据集页面](https://robotics-transformer-x.github.io/)

#### Robomimic
- **描述**: 机器人模仿学习数据集
- **规模**: 多个任务, 多个演示者
- **特点**: 高质量演示, 多模态数据

#### MetaWorld
- **描述**: 机器人操纵基准
- **特点**: 50个任务, 标准化评估

#### LIBERO
- **描述**: 长期具身学习基准
- **特点**: 长期任务, 复杂场景

### 视频数据集

#### Kinetics
- **描述**: 大规模视频理解数据集
- **规模**: 100万+ 视频
- **用途**: 预训练视频模型

#### Something-Something
- **描述**: 动作识别数据集
- **特点**: 细粒度动作标注

#### UCF101
- **描述**: 动作识别基准
- **特点**: 101个动作类别

---

## 工具和框架

### 机器人框架

#### PyBullet
- **描述**: 物理模拟引擎
- **用途**: 机器人仿真, 环境模拟

#### MuJoCo
- **描述**: 多体动力学引擎
- **用途**: 高精度仿真

#### Gym/Gymnasium
- **描述**: 强化学习环境框架
- **用途**: 标准化环境接口

#### ROS (Robot Operating System)
- **描述**: 机器人操作系统
- **用途**: 机器人软件开发

### 机器学习框架

#### PyTorch
- **描述**: 深度学习框架
- **特点**: 灵活, 易于调试

#### TensorFlow
- **描述**: 深度学习框架
- **特点**: 高性能, 生产就绪

#### JAX
- **描述**: 可组合变换框架
- **特点**: 自动微分, 高效编译

### 可视化工具

#### Wandb (Weights & Biases)
- **描述**: 机器学习实验追踪
- **用途**: 训练监控, 结果可视化

#### TensorBoard
- **描述**: TensorFlow 可视化工具
- **用途**: 训练过程可视化

#### Plotly
- **描述**: 交互式绘图库
- **用途**: 数据可视化

---

## 博客和教程

### 官方教程

#### DreamZero 官方博客
- **链接**: [项目网站博客](https://dreamzero0.github.io/)
- **内容**: 方法介绍, 实验结果, 使用指南

### 相关教程

#### Vision Transformers 入门
- **内容**: ViT 架构, 预训练, 微调
- **推荐**: Hugging Face 官方教程

#### 扩散模型详解
- **内容**: 扩散模型原理, 实现细节
- **推荐**: Lil'Log 博客

#### 机器人学习基础
- **内容**: 强化学习, 模仿学习, 多任务学习
- **推荐**: Berkeley 机器人学习课程

#### 多模态学习
- **内容**: 视觉-语言模型, 融合方法
- **推荐**: CLIP, ALIGN 论文和教程

### 视频教程

#### Stanford CS231N (计算机视觉)
- **内容**: CNN, 视觉任务, 预训练模型

#### UC Berkeley CS285 (深度强化学习)
- **内容**: 强化学习, 策略学习, 模型学习

#### CMU 机器人学习
- **内容**: 机器人控制, 学习方法

---

## 相关领域

### 相关研究方向

#### 1. 视觉语言模型
- CLIP, ALIGN, BLIP
- 多模态理解和生成

#### 2. 视频理解
- 视频分类, 动作识别
- 时间建模, 长期依赖

#### 3. 强化学习
- 策略梯度, 价值函数
- 模型学习, 世界模型

#### 4. 模仿学习
- 行为克隆, 逆强化学习
- 从演示学习

#### 5. 多任务学习
- 任务共享, 元学习
- 转移学习

#### 6. 具身 AI
- 视觉导航, 操纵
- 环境交互, 探索

#### 7. 物理理解
- 物理模拟, 物理推理
- 因果关系, 约束学习

---

## 会议和期刊

### 顶级会议

#### 机器人相关
- **ICRA** (IEEE International Conference on Robotics and Automation)
- **IROS** (IEEE/RSJ International Conference on Intelligent Robots and Systems)
- **RSS** (Robotics: Science and Systems)
- **CoRL** (Conference on Robot Learning)

#### 机器学习相关
- **NeurIPS** (Neural Information Processing Systems)
- **ICML** (International Conference on Machine Learning)
- **ICLR** (International Conference on Learning Representations)
- **CVPR** (IEEE/CVF Conference on Computer Vision and Pattern Recognition)
- **ICCV** (International Conference on Computer Vision)
- **ECCV** (European Conference on Computer Vision)

### 期刊
- **IEEE Transactions on Robotics**
- **International Journal of Robotics Research**
- **Journal of Machine Learning Research**

---

## 研究机构

### 主要研究机构

#### 学术机构
- **UC Berkeley** - 机器人学习实验室
- **Stanford University** - 人工智能实验室
- **MIT** - 计算机科学和人工智能实验室
- **CMU** - 机器人研究所
- **University of Toronto** - 机器学习小组

#### 工业研究机构
- **Google DeepMind** - 机器人和 AI 研究
- **OpenAI** - 多模态学习, 强化学习
- **Meta AI** - 计算机视觉, 机器学习
- **Microsoft Research** - 多模态学习
- **NVIDIA** - 深度学习, 机器人

---

## 贡献指南

### 如何贡献

1. **Fork 仓库**
   ```bash
   git clone https://github.com/YOUR_USERNAME/Awesome-WAM.git
   cd Awesome-WAM
   ```

2. **创建新分支**
   ```bash
   git checkout -b add-new-resource
   ```

3. **添加资源**
   - 遵循现有格式
   - 提供清晰的描述
   - 包含相关链接
   - 添加关键词标签

4. **提交 Pull Request**
   ```bash
   git add .
   git commit -m "Add: [资源类型] [资源名称]"
   git push origin add-new-resource
   ```

### 资源格式

#### 论文
```markdown
- **[论文名称]**
  - **作者**: 作者名称
  - **发表**: 发表年份/会议
  - **链接**: [arXiv](url) / [PDF](url)
  - **关键词**: 关键词1, 关键词2
  - **简介**: 简短描述
```

#### 代码
```markdown
- **[项目名称]**
  - **GitHub**: [链接](url)
  - **描述**: 项目描述
  - **特点**: 特点1, 特点2
  - **依赖**: 依赖库
```

#### 数据集
```markdown
- **[数据集名称]**
  - **描述**: 数据集描述
  - **规模**: 数据规模
  - **特点**: 特点1, 特点2
  - **链接**: [官方网站](url)
```

### 提交规范

- 使用清晰的提交信息
- 一次提交一个资源或相关资源组
- 提供有意义的 PR 描述
- 确保链接有效

---

## 许可证

本仓库采用 **MIT License**

---

## 联系方式

- **Issues**: 提交问题和建议
- **Discussions**: 讨论和交流
- **Email**: 联系方式

---

## 相关资源

- [[WAM/README.md]] - DreamZero 学习资源
- [[WAM/02-核心学习/论文解读.md]] - 论文详细解读
- [OpenMOSS/Awesome-WAM](https://github.com/OpenMOSS/Awesome-WAM) - 原始仓库

---

**最后更新**: 2026-06-04
**维护者**: OpenMOSS Community
**版本**: 1.0

