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


`view_driver_state`  --> 主观测口

   误差类型    当前表现
  ━━━━━━━━━━  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   感知误差    SAM3 选中了相似但错误的碗
  ──────────  ────────────────────────────────────────────
   几何误差    mask 中心不一定是可抓取点
  ──────────  ────────────────────────────────────────────
   状态误差    模型没有确认物体是否真的被抓住
  ──────────  ────────────────────────────────────────────
   记忆误差    多次工具调用后，模型只保留“已经定位”的摘要



(pi0_pick) 内部每执行一个 chunk 后都会检查是否满足“抓住并抬起物体”的条件：


- `checkpoints/contact_graspnet/checkpoints/model.pt` — Contact-GraspNet,抓取位姿预测
- 5 个 PointNet++ 的 `best_model.pth`(classification / part_seg / sem_seg)—— 点云网络,是 Contact-GraspNet 的骨干