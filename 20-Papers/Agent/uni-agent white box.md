核心React代码：
	问模型一次
	解析模型返回的 tool_calls
	执行工具
	把工具结果写回 transcript
	判断是否结束

迁移libero -> uni-agent

|SWE-Bench|LIBERO|
|---|---|
|`/testbed` 代码仓库|LIBERO 仿真世界|
|shell/editor 工具|robot primitive tools|
|pytest eval|success / termination eval|
|git patch|robot action sequence / code policy|
|instance_id|suite + task_id + seed|
|prompt issue|task language goal|
|sandbox filesystem|simulator state|