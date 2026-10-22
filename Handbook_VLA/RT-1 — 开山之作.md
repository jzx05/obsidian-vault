**核心思想**：把机器人控制建模为 Token 生成问题。

⚡ **Eureka Moment**：把机器人策略做成"小模型 + 少 token + 离散动作 + 并行解码"，就能在 3 Hz 的硬预算内吃下 700+ 任务；限制机器人 scaling 的往往不是模型不够大，而是推理预算和数据多样性。

6 × RGB (300×300)
      │
      ▼
EfficientNet-B3 (16M, ImageNet 预训练)
      ▲
      │ FiLM(USE(ℓ))  ← 自然语言指令
      ▼
81 tokens / frame
      │
      ▼
TokenLearner (K=8)  →  8 tokens / frame  →  6 帧拼接 = 48 tokens
      │
      ▼
decoder-only Transformer (8 层, 19M)
      │
      ▼
11 个并行分类头 × 256 bins
      │
      ▼
11 维动作  →  机器人 @ 3 Hz
