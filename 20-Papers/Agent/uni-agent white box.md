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


| 你整理的方法                      | `uni2-agent` 当前对应                                  | 结论                         |     |
| --------------------------- | -------------------------------------------------- | -------------------------- | --- |
| `get_observation`           | `libero_inspection.view_driver_state` + 返回的场景/腕部图像 | 有类似能力，但不是同名单一函数            |     |
| `point_prompt_molmo`        | 没有                                                 | 需要新增                       |     |
| `segment_sam3_point_prompt` | `libero_perception.segment`，但当前主要通过 SAM3 grounding | 部分对应，需要扩展为 point prompt    |     |
| `segment_sam3_text_prompt`  | `libero_perception.segment`                        | 基本对应                       |     |
| `plan_grasp`                | 没有 Contact-GraspNet 版本                             | 需要新增                       |     |
| `select_top_down_grasp`     | 没有                                                 | 需要新增，或作为 `plan_grasp` 内部逻辑 |     |
| `goto_pose`                 | `libero_motion.move_to` / `move_pose`              | 只有部分对应，底层机制不同              |     |
| `open_gripper`              | `libero_motion.set_gripper(gripper="open")`        | 对应                         |     |
| `close_gripper`             | `libero_motion.set_gripper(gripper="close")`       | 对应                         |     |
| `decompose_transform`       | 没有对 agent 暴露的对应函数                                  | 应该内部使用，不暴露                 |     |


get_observation  和     view_driver_state
    view_camera_meta  功能差不多吗  可以集成变成get_observation吗？     rotate_wrist
    rotate_pitch  这两个我决定删掉  move_to` / `move_pose` 换成 `goto_pose` 应该就行

  保留 
get_observation
point_prompt_molmo
segment_sam3_point_prompt
segment_sam3_text_prompt
plan_grasp
goto_pose
open_gripper
close_gripper
release
vla_pick
vla_doubled
finish

////////////////////////////////////////////////////////
观察层
└── get_observation

感知层
├── point_prompt_molmo
├── segment_sam3_point_prompt
└── segment_sam3_text_prompt

抓取规划层
└── plan_grasp
└──plan_grasp_from_point_clouds

运动执行层
├── goto_pose
├── open_gripper
├── close_gripper
└── release

学习策略
├── vla_pick
├── vla_doubled


结束层 
├── finish
////////////////////////////////////////////////////////


get_observation
point_prompt_molmo
segment_sam3_point_prompt
segment_sam3_text_prompt
plan_grasp
goto_pose
open_gripper
close_gripper
release
vla_pick
vla_doubled
finish




需求文档：




尚未动过的按文件简化候选：tools/libero.py、backends/base.py、task.py、recorder.py、artifacts.py、preprocess.py、model_servers/*。要我继续往下做

2. /root/rivermind-data/codes/uni2-agent/logs/202608101208-external-gpt-5.6-sol-0-0-ee59e16469ab4d37bef68df6d0722c00/artifacts/agentview.mp4 视频需要及时更新刷新
3. 鼓励使用vla_pick，vla_double
4. graphNet 策略的结果会有一定的偏差  你查一下大概多少偏差  需要多少弥补
5. artifact_reader   改成 file_reader 名字，molmo 要改成 molmo2
6. 现在的代码结构完整  但是我后面需要用harness enginner agent 修改harness，所以我希望代码可不可以简单一点，当然代码的注释和说明要干净，不要过于冗余，同时不要缺失核心功能

o.yaml
2026-08-10 06:12:03,668 | WARNING | UNI_AGENT_DEBUG_INLINE=1: running rollouts inline without Ray
/root/.vscode-server/extensions/ms-python.debugpy-2026.6.0-linux-x64/bundled/libs/debugpy/adapter/../../debugpy/launcher/../../debugpy/../debugpy/_vendored/force_pydevd.py:18: UserWarning: incompatible copy of pydevd already imported:
 /root/rivermind-data/venvs/uni2-libero/lib/python3.10/site-packages/pydevd_plugins/extensions/pydevd_plugin_omegaconf.py
  warnings.warn(msg + ':\n {}'.format('\n  '.join(_unvendored)))
LIBERO model RPC listening on http://127.0.0.1:8081
Loading weights: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 199/199 [00:00<00:00, 6937.86it/s]
[transformers] BertModel LOAD REPORT from: /root/rivermind-data/codes/uni2-agent/checkpoints/bert-base-uncased
Key                                        | Status     | Details
-------------------------------------------+------------+--------
cls.predictions.transform.LayerNorm.bias   | UNEXPECTED |        
cls.predictions.bias                       | UNEXPECTED |        
cls.predictions.transform.dense.weight     | UNEXPECTED |        
cls.predictions.transform.dense.bias       | UNEXPECTED |        
cls.predictions.transform.LayerNorm.weight | UNEXPECTED |        
cls.seq_relationship.bias                  | UNEXPECTED |        
cls.seq_relationship.weight                | UNEXPECTED |        

Notes:
- UNEXPECTED:   can be ignored when loading from different task/architecture; not ok if you expect identical arch.
Loading weights: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 211/211 [00:00<00:00, 6810.82it/s]
LIBERO model RPC listening on http://127.0.0.1:8082
/root/.vscode-server/extensions/ms-python.debugpy-2026.6.0-linux-x64/bundled/libs/debugpy/adapter/../../debugpy/launcher/../../debugpy/../debugpy/_vendored/force_pydevd.py:18: UserWarning: incompatible copy of pydevd already imported:
 /root/rivermind-data/venvs/uni2-libero/lib/python3.10/site-packages/pydevd_plugins/extensions/pydevd_plugin_omegaconf.py
  warnings.warn(msg + ':\n {}'.format('\n  '.join(_unvendored)))
Using a slow image processor as `use_fast` is unset and a slow processor was saved with this model. `use_fast=True` will be the default behavior in v4.52, even if the model was saved with a slow processor. This will result in minor differences in outputs. You'll still be able to use a slow processor with `use_fast=False`.
Loading checkpoint shards: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 4/4 [00:26<00:00,  6.54s/it]
Molmo2 loaded from /root/rivermind-data/models/molmo2-4b on cuda
LIBERO model RPC listening on http://127.0.0.1:8084
/root/.vscode-server/extensions/ms-python.debugpy-2026.6.0-linux-x64/bundled/libs/debugpy/adapter/../../debugpy/launcher/../../debugpy/../debugpy/_vendored/force_pydevd.py:18: UserWarning: incompatible copy of pydevd already imported:
 /root/rivermind-data/venvs/uni2-libero/lib/python3.10/site-packages/pydevd_plugins/extensions/pydevd_plugin_omegaconf.py
  warnings.warn(msg + ':\n {}'.format('\n  '.join(_unvendored)))
model func:  <module 'contact_graspnet_pytorch.contact_graspnet' from '/root/rivermind-data/codes/uni2-agent/third_party/contact_graspnet_pytorch/contact_graspnet_pytorch/contact_graspnet.py'>
/root/rivermind-data/codes/uni2-agent/third_party/contact_graspnet_pytorch/checkpoints/contact_graspnet/checkpoints/model.pt  waring 消除



{"action":"plan_grasp","args":"{\"segment_ref\":\"seg_1\",\"top_k\":5}"}  何意为top_k？