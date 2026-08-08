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
Contact-GraspNet 的输入是场景/目标的 3D 几何信息，输出是带评分的 6D 夹爪抓取位姿和接触点；它是“抓哪里、以什么方向抓”的模型，不是“识别什么”和“执行完整动作轨迹”的模型。



  |方法|输入|输出/作用|
|---|---|---|
|`get_observation()`|无|获取场景 RGB、深度、相机内外参、腕部相机、末端位姿和关节状态|
|`point_prompt_molmo(image, text_prompt)`|RGB 图像、目标描述|Molmo 根据语言返回目标像素坐标|
|`segment_sam3_text_prompt(rgb, text_prompt)`|RGB、目标文本|SAM3 根据文字生成目标 mask、框和分数|
|`segment_sam3_point_prompt(rgb, point_coords)`|RGB、像素 `(x,y)`|SAM3 根据 Molmo 给出的点生成目标 mask|


plan_grasp
select_top_down_grasp


goto_pose(position, quaternion)





get_observation
point_prompt_molmo
segment_sam3_point_prompt
segment_sam3_text_prompt
plan_grasp
select_top_down_grasp

goto_pose （内置solve_ik，move_to_joints ？）
open_gripper
close_gripper

(decompose_transform内置有需要的用)