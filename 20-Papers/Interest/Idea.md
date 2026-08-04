%%1.  感知驱动的 skill 检索 
  现在 skill 检索基本都靠 planner 输出的语言 query，但具身场景更自然的是「看到 mug → 触发 grasp-mug」。可做：
  - VLM affordance heatmap 直接触发 skill 而不经过语言
  - Skill 索引结构从文本 embedding 换成视觉-动作联合 embedding
  - 对小模型尤其友好——省掉整个语言 retrieval 环节

2.  Skill 路由走 MoE 而不是走 context
  换一个思路：不把 skill 塞 prompt，也不把 skill 塞权重，而是把 skill 做成 MoE expert 或 adapter，靠路由激活。可做：
  - Skill-expert MoE 架构
  - 只激活 top-k skill 的 adapter，attention 开销和 skill 库大小解耦
  - 对 4B/7B 特别有意义，因为你可以做「小 base + 大 skill bank」%%


2. 通过训练期间重组 Harness，制造多样化的感知、工具、Skill、上下文和执行条件，迫使 Qwen3.5-9B 学到更通用的任务理解、状态判断、长时序规划、工具组合、失败恢复和结果验证能力，最终提高具身任务上的推理与执行泛化。训练阶段通过 Harness 重组训练模型的通用规划能力；推理阶段使用一个稳定的部署 Harness，给模型提供可靠感知、Affordance、Skills、Memory 和执行工具，让已经训练好的规划能力发挥出来