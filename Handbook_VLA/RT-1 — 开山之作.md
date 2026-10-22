**核心思想**：把机器人控制建模为 Token 生成问题。

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
