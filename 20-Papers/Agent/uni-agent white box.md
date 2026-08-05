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

当前 SWE 风格执行链路是：

```text
parallel_infer_api.py
  -> TaskConfigResolver
  -> get_task(config).run()
  -> Task.build_sandbox()
  -> Task.build_agent()
  -> Agent.run(sandbox, messages)
  -> Toolbox.call(...)
  -> Task reward
```

==目标：LIBERO 应该作为一个新的 task family 和一组 task-specific tools 接入，而不是在 `react` 里写 LIBERO 特判==



LiberoTask(runtime_backend="libero_local")
  -> start real LIBERO runtime server
  -> /reset 调 env.reset()
  -> /observe 返回真实 task_language/state/render summary
  -> /run_skill 接一个最小 scripted skill backend
  -> /success 读 official success / libero_terminated
  -> /close 调 env.close()