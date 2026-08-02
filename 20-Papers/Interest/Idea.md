1.  感知驱动的 skill 检索 
  现在 skill 检索基本都靠 planner 输出的语言 query，但具身场景更自然的是「看到 mug → 触发 grasp-mug」。可做：
  - VLM affordance heatmap 直接触发 skill 而不经过语言
  - Skill 索引结构从文本 embedding 换成视觉-动作联合 embedding
  - 对小模型尤其友好——省掉整个语言 retrieval 环节

2.  Skill 路由走 MoE 而不是走 context
  换一个思路：不把 skill 塞 prompt，也不把 skill 塞权重，而是把 skill 做成 MoE expert 或 adapter，靠路由激活。可做：
  - Skill-expert MoE 架构
  - 只激活 top-k skill 的 adapter，attention 开销和 skill 库大小解耦
  - 对 4B/7B 特别有意义，因为你可以做「小 base + 大 skill bank」