# RPent 与 CaP-X 的 LIBERO Tool / Backend 对比

这份文档的目的不是决定“把哪个项目整体搬进 uni2-agent”，而是把两个项目里模型能看到的接口、背后的执行 backend、以及适不适合迁移成 uni2-agent 的高层 tool 说清楚。

## 先统一术语

- **planner-visible tool**：LLM / planner 直接调用的接口。uni2-agent 里应该保持少而高层，例如 `locate_object`、`move_to_object`、`grasp_object`、`place_on`、`wait`。
- **tool_backend**：实现 planner-visible tool 的内部 backend。它可以调用视觉模型、IK、VLA、仿真器 step、几何函数，但这些细节不应该全部暴露给 planner。
- **primitive / actuator**：真正会驱动机器人或仿真器状态变化的底层动作，例如 `env.step(action)`、`move_to_joints`、`open_gripper`、`close_gripper`。
- **Harness Skill**：以后 Harness 层沉淀出来的可复用恢复流程或任务片段。它不等于现在直接暴露给模型的底层 action tool。

当前 uni2-agent 的 LIBERO tool 边界已经比较干净：`uni_agent/tools/libero.py` 只暴露 `libero_observe`、`libero_run_tool`、`submit`，其中 `libero_run_tool` 的 `name` 限定为 `locate_object`、`move_to_object`、`grasp_object`、`place_on`、`wait`；协议常量在 `uni_agent/tasks/libero/protocol.py`。这意味着 RPent 和 CaP-X 最适合被接成 **tool_backend**，而不是把它们所有原始函数直接塞给 planner。

源码依据：

- uni2-agent planner-visible LIBERO tool：`/home/eren/codes/uni2-agent/uni_agent/tools/libero.py:50`
- uni2-agent 高层 tool 名列表：`/home/eren/codes/uni2-agent/uni_agent/tasks/libero/protocol.py:16`

## RPent：当前暴露的工具与 backend

RPent 的 LIBERO 入口是 `LiberoToolkit`。它把工具 schema 从 `robots.libero.tools.TOOLS_SPEC` 取出来，然后注册 inspection tools 和 primitive tools。

源码依据：

- `LiberoToolkit._register_libero_tools()`：`/home/eren/codes/RPent/robots/libero/toolkit.py:41`
- primitive tool 统一经 `_step()` 调用 `LiberoPrimitives.<name>`：`/home/eren/codes/RPent/robots/libero/toolkit.py:67`
- 工具 schema 列表 `TOOLS_SPEC`：`/home/eren/codes/RPent/robots/libero/tools.py:1253`

### RPent 工具表

| RPent tool 名 | RPent 是否暴露给 planner | 类型 | 背后 backend | 适合迁入 uni2-agent 的方式 |
|---|---:|---|---|---|
| `view_driver_state` | 是 | observation / state inspection | 读 `states.json`、图像路径、world map 路径、上一步 command/result | 可迁成 `libero_observe` 的内部实现，不建议作为单独 planner tool 暴露 |
| `view_camera_meta` | 是 | camera calibration inspection | 读取 agentview / wrist camera metadata | 内部给 `locate_object` 使用 |
| `back_project` | 是 | perception geometry | 从预计算 world map 里把像素或区域反投影成 world XYZ | 内部给 `locate_object` 使用 |
| `segment` | 是 | perception | `Sam3Client.segment(...)`，输出 mask、box、score、world_xyz、overlay artifact | 可直接作为 `locate_object` 的核心 backend |
| `move_to` | 是 | low-level motion primitive | 计算 7D OSC action，循环 `LiberoEnvClient.step(action)` | 不建议直接暴露；可隐藏在 `move_to_object` / `place_on` 内部 |
| `pi0_pick` | 是 | learned VLA primitive | `VLAClient.predict_action_batch(...)` 产生 action chunk，再 `env.chunk_step(actions)` | 可作为 `grasp_object` backend 候选 |
| `pi0_doubled` | 是 | learned VLA contact primitive | VLA action chunk，靠 LIBERO termination 判断任务是否成功 | 可作为接触类任务 backend，暂不放进第一版高层 ABI |
| `release` | 是 | gripper primitive | 循环发送打开夹爪的 7D action | 不建议直接暴露；放进 `place_on` 内部 |
| `set_gripper` | 是 | gripper primitive | 保持当前 EEF pose，发送夹爪 action | 不建议直接暴露；作为 backend 内部辅助 |
| `rotate_wrist` | 是 | wrist orientation primitive | 计算 yaw 误差，发送 action[5] | 不建议直接暴露；作为复杂放置 backend 内部辅助 |
| `rotate_pitch` | 是 | wrist orientation primitive | 计算 pitch 误差，发送 action[3] | 不建议直接暴露；作为复杂放置 backend 内部辅助 |
| `move_pose` | 是 | pose servo primitive | 同时控制 xyz、pitch、yaw，发送 7D action | 不建议直接暴露；可用于 `move_to_object` / `place_on` 的内部执行 |
| `place` | 否，当前 toolkit 未注册 | learned VLA place primitive | VLA 按 `place it on {target}` 执行动作 chunk | 值得迁入 backend；需要先在 RPent 侧确认稳定性 |

### RPent backend 链路

RPent 更像一个“工具化机器人 driver”。它的模型可见工具不少，而且不少工具已经接近低层控制。

典型链路：

```text
planner
  -> RPent LiberoToolkit tool
  -> LiberoPrimitives method
  -> LiberoEnvClient RPC
  -> env_server.py
  -> rlinf.envs.libero.LiberoEnv
  -> env.step(...) / env.chunk_step(...)
```

关键 backend：

- **环境 backend**：`LiberoEnvClient` 通过 RPC 调 `env.reset`、`env.step`、`env.chunk_step`、`env.raw_obs`、`env.render_camera` 等；server 端由 `env_server.py` 包装 `rlinf.envs.libero.libero_env.LiberoEnv`。
- **VLA backend**：`pi0_pick`、`pi0_doubled`、`place` 走 `VLAClient.predict_action_batch(...)`，然后用 `chunk_step` 执行动作块。
- **SAM3 backend**：`segment` 走 `Sam3Client`，并结合已保存的 image/world map artifact 给出 `world_xyz`。
- **scripted controller backend**：`move_to`、`move_pose`、`rotate_wrist`、`rotate_pitch`、`release`、`set_gripper` 直接生成底层 action。

源码依据：

- `pi0_pick`：`/home/eren/codes/RPent/robots/libero/tools.py:132`
- `pi0_doubled`：`/home/eren/codes/RPent/robots/libero/tools.py:207`
- `place`：`/home/eren/codes/RPent/robots/libero/tools.py:248`
- `move_to`：`/home/eren/codes/RPent/robots/libero/tools.py:293`
- `release`：`/home/eren/codes/RPent/robots/libero/tools.py:593`
- `segment`：`/home/eren/codes/RPent/robots/libero/tools.py:707`
- `view_driver_state`：`/home/eren/codes/RPent/robots/libero/tools.py:1679`
- `back_project`：`/home/eren/codes/RPent/robots/libero/tools.py:1867`
- `LiberoEnvClient.step/chunk_step`：`/home/eren/codes/RPent/robots/libero/env_client.py:59`
- `LiberoEnvFacade.step/chunk_step`：`/home/eren/codes/RPent/robots/libero/env_server.py:191`

### 对 RPent 的判断

RPent 的优点是“离能跑最近”：它已经有 LIBERO env RPC、Pi0/VLA action chunk、SAM3 artifact、状态 dump、视频和调试信息。

RPent 的问题是 planner-visible surface 太低层：`move_to(xyz)`、`move_pose(xyz, pitch, yaw)`、`rotate_*`、`set_gripper` 会诱导 planner 操心坐标、姿态、步长和控制器限制。对 uni2-agent 的具身 RL 来说，这些应该成为 backend 内部细节，否则训练出来的是“会调 action 参数的 planner”，不是“会做任务规划和恢复的 agent”。

推荐迁移方式：

```text
tool_backend: rpent_v0

locate_object
  -> segment / back_project / view_driver_state

move_to_object
  -> locate_object
  -> move_to or move_pose internally

grasp_object
  -> pi0_pick(prompt=...)
  -> optionally verify with view_driver_state / images

place_on
  -> locate target
  -> move_to or move_pose internally
  -> release
  -> optionally use RPent place(...) once validated

wait
  -> no-op or short env step
```

## CaP-X：API 函数与 backend

CaP-X 和 RPent 的抽象方式不一样。CaP-X 不是 OpenAI JSON function-call 风格的 tool registry，而是 **code generation API** 风格：每个 API 类通过 `functions()` 返回函数名到 callable 的映射，`combined_doc()` 把函数签名和 docstring 拼进 prompt，执行时这些函数会被注入 Python exec 的全局命名空间。

源码依据：

- `ApiBase.functions()`：`/home/eren/codes/cap-x/capx/integrations/base_api.py:92`
- `ApiBase.combined_doc()`：`/home/eren/codes/cap-x/capx/integrations/base_api.py:96`
- prompt 拼接 APIs：`/home/eren/codes/cap-x/capx/envs/tasks/base.py:139`
- API 函数注入 exec globals：`/home/eren/codes/cap-x/capx/envs/tasks/base.py:153`
- 添加 API 文档说明 docstring 是模型接口文档：`/home/eren/codes/cap-x/docs/adding-apis.md:16`

### CaP-X LIBERO API 注册名

| API 注册名 | 定位 | planner 可见方式 | 迁移建议 |
|---|---|---|---|
| `FrankaLiberoApi` | 视觉版较高层 API | 函数名直接出现在生成代码环境里 | 可作为 `capx_v0` backend 参考 |
| `FrankaLiberoApiReduced` | 更低层的视觉、grasp、IK、动作 API | 同上 | 可拆成 backend 内部模块，不建议原样暴露 |
| `FrankaLiberoApiReducedSkillLibrary` | Reduced API + 合成几何/视觉 utility | 同上 | utility 很有价值，但放 backend 内部 |
| `FrankaLiberoPrivilegedApi` | ground-truth object pose + IK | 同上 | 只用于 oracle、debug、regression，不用于正式 planner 训练 |

源码依据：

- LIBERO API 注册：`/home/eren/codes/cap-x/capx/integrations/__init__.py:125`
- docs 可用 API 表：`/home/eren/codes/cap-x/docs/libero-tasks.md:182`

### `FrankaLiberoApi`

| API 函数名 | 类型 | 背后 backend | 适合迁入 uni2-agent 的方式 |
|---|---|---|---|
| `get_observation` | observation | `FrankaLiberoEnv.get_observation()`，返回 agentview / wrist RGB、depth、intrinsics、pose、robot state | 对应 `libero_observe` 的 backend |
| `get_object_pose` | perception-derived pose | 内部调用 `get_object_3d_points_and_masks_from_language`，再做 OBB | 可作为 `locate_object` 实现 |
| `sample_grasp_pose` | grasp planning | segment object、构建点云、Contact-GraspNet 规划 grasp pose | 可作为 `grasp_object` 内部候选 |
| `goto_pose` | motion primitive | TCP offset、PyRoKi IK、`move_to_joints_blocking` | 不直接暴露；给 `move_to_object` / `place_on` 内部调用 |
| `open_gripper` | actuator primitive | `_set_gripper(1.0)` + `_step_once()` | 不直接暴露 |
| `close_gripper` | actuator primitive | `_set_gripper(0.0)` + `_step_once()` | 不直接暴露 |
| `get_oriented_bounding_box_from_3d_points` | geometry | Open3D OBB helper | backend 内部工具 |
| `get_object_3d_points_and_masks_from_language` | perception pipeline | observation + Molmo point prompt + SAM3 + depth point cloud + camera pose transform | `locate_object` 的强候选 |
| `goto_home_joint_position` | motion reset/helper | `env.home_joint_position` + `move_to_joints_blocking` | backend 内部恢复动作 |

源码依据：

- `FrankaLiberoApi.functions()`：`/home/eren/codes/cap-x/capx/integrations/franka/libero.py:74`
- 初始化 SAM3、Molmo、Contact-GraspNet、PyRoKi：`/home/eren/codes/cap-x/capx/integrations/franka/libero.py:57`
- `get_observation`：`/home/eren/codes/cap-x/capx/integrations/franka/libero.py:93`
- `goto_pose`：`/home/eren/codes/cap-x/capx/integrations/franka/libero.py:215`

### `FrankaLiberoApiReduced`

`Reduced` 更像给代码生成 agent 用的一组零件。它暴露了更多低层函数，所以直接迁成 planner-visible tools 会过度展开。

| API 函数名 | 类型 | 背后 backend | 迁移建议 |
|---|---|---|---|
| `get_observation` | observation | `FrankaLiberoEnv.get_observation()` | 可用 |
| `segment_sam3_text_prompt` | perception | SAM3 text prompt | 内部给 `locate_object` |
| `segment_sam3_point_prompt` | perception | SAM3 point prompt | 内部给 `locate_object` |
| `point_prompt_molmo` | perception / VLM pointing | Molmo 输出像素点 | 内部给 `locate_object` |
| `plan_grasp` | grasp planning | Contact-GraspNet depth + mask | 内部给 `grasp_object` |
| `plan_grasp_from_point_clouds` | grasp planning | Contact-GraspNet point cloud | 内部给 `grasp_object` |
| `get_oriented_bounding_box_from_3d_points` | geometry | Open3D OBB helper | backend 内部 |
| `solve_ik` | motion planning | PyRoKi IK | backend 内部 |
| `move_to_joints` | actuator primitive | `FrankaLiberoEnv.move_to_joints_blocking` | 不直接暴露 |
| `open_gripper` | actuator primitive | gripper helper + env step | 不直接暴露 |
| `close_gripper` | actuator primitive | gripper helper + env step | 不直接暴露 |
| `goto_pose` | motion primitive | `solve_ik` + `move_to_joints` | 不直接暴露 |
| `goto_home_joint_position` | motion helper | home joints + blocking move | backend 内部 recovery |
| `subsample_point_cloud` | geometry / performance | 点云采样 | backend 内部 |
| `filter_noise` | geometry / perception cleanup | DBSCAN noise filtering | backend 内部 |

源码依据：

- `FrankaLiberoApiReduced.functions()`：`/home/eren/codes/cap-x/capx/integrations/franka/libero_reduced.py:84`
- `plan_grasp`：`/home/eren/codes/cap-x/capx/integrations/franka/libero_reduced.py:239`
- `solve_ik`：`/home/eren/codes/cap-x/capx/integrations/franka/libero_reduced.py:339`
- `move_to_joints`：`/home/eren/codes/cap-x/capx/integrations/franka/libero_reduced.py:398`

### `FrankaLiberoApiReducedSkillLibrary`

这个类名里有 `SkillLibrary`，但在 uni2-agent 的设计里最好不要把它理解成 Harness Skill。它更像“从历史代码生成中提炼出来的几何/视觉 utility 函数包”。

它继承 `FrankaLiberoApiReduced`，然后额外加入：

- `rotation_matrix_to_quaternion`
- `decompose_transform`
- `depth_to_point_cloud`
- `mask_to_world_points`
- `pixel_to_world_point`
- `transform_points`
- `interpolate_segment`
- `normalize_vector`
- `select_top_down_grasp`

这些函数本身多为纯几何或相机数学，迁移价值很高，但建议放进 `tool_backend` 内部库，而不是直接作为 planner-visible tool。planner 不应该自己写“mask 转世界坐标再选 top-down grasp 再 IK”的长程序；这些应收敛成 `locate_object`、`grasp_object` 的内部实现。

源码依据：

- 追加 utility 的 `functions()`：`/home/eren/codes/cap-x/capx/integrations/franka/libero_reduced_skill_library.py:43`
- 文件注释说明来自 reduced API / exampleless 生成结果：`/home/eren/codes/cap-x/capx/integrations/franka/libero_reduced_skill_library.py:57`

### `FrankaLiberoPrivilegedApi`

| API 函数名 | 类型 | 背后 backend | 迁移建议 |
|---|---|---|---|
| `get_observation` | observation | `FrankaLiberoEnv.get_observation()` | 可用于 debug |
| `get_object_pose` | privileged state | `_env._get_object_pose(object_name)` | 只用于 oracle / regression |
| `get_all_object_poses` | privileged state | `_env._get_all_object_poses()` | 只用于 oracle / regression |
| `sample_grasp_pose` | privileged grasp hint | `_get_object_pose` + 固定 top-down quaternion | 只用于 oracle / regression |
| `goto_pose` | motion primitive | PyRoKi IK + `move_to_joints_blocking` | backend 内部 |
| `open_gripper` | actuator primitive | gripper helper + env step | backend 内部 |
| `close_gripper` | actuator primitive | gripper helper + env step | backend 内部 |
| `goto_pose_interactive_cartesian` | motion primitive | 重复读取 target predicate、IK、step | backend 内部或暂不迁 |

源码依据：

- `FrankaLiberoPrivilegedApi.functions()`：`/home/eren/codes/cap-x/capx/integrations/franka/libero_privileged.py:33`
- `get_object_pose` 直接读 env object pose：`/home/eren/codes/cap-x/capx/integrations/franka/libero_privileged.py:67`
- `sample_grasp_pose` 实际返回 `(pos, quat)`，但类型标注写的是 `None`：`/home/eren/codes/cap-x/capx/integrations/franka/libero_privileged.py:90`

### CaP-X low-level env 与服务依赖

CaP-X 的 LIBERO env 是 `FrankaLiberoEnv`，它包装 LIBERO 的 offscreen simulator，controller 使用 `JOINT_POSITION`。核心 primitive 是 `move_to_joints_blocking`，它把 7 个 joint target 转成 LIBERO action，再调用 `handle.step(action)`。夹爪通过 `_set_gripper` 和 `_step_once` 驱动。

源码依据：

- `FrankaLiberoEnv` 初始化：`/home/eren/codes/cap-x/capx/envs/simulators/libero.py:34`
- `load_libero_task(... controller="JOINT_POSITION")`：`/home/eren/codes/cap-x/capx/envs/simulators/libero.py:60`
- `move_to_joints_blocking`：`/home/eren/codes/cap-x/capx/envs/simulators/libero.py:212`
- `_set_gripper` / `_step_once`：`/home/eren/codes/cap-x/capx/envs/simulators/libero.py:267`
- `get_observation` 返回相机、深度、内参、pose、robot state：`/home/eren/codes/cap-x/capx/envs/simulators/libero.py:432`
- `task_completed()` 调 LIBERO success checker：`/home/eren/codes/cap-x/capx/envs/simulators/libero.py:527`

常规 config 中的外部服务：

| 服务 | 默认端口 | 用途 | 来源 |
|---|---:|---|---|
| PyRoKi | 8116 | IK / motion planning | `franka_libero_spatial_0.yaml` |
| Contact-GraspNet | 8115 | grasp pose planning | `franka_libero_spatial_0.yaml` |
| SAM3 | 8114 | text / point segmentation | `franka_libero_spatial_0.yaml` |
| Molmo | 8122 | VLM point prompt，默认在代码里 | `capx/integrations/vision/molmo.py` |

配置依据：

- 常规视觉版 config：`/home/eren/codes/cap-x/env_configs/libero/franka_libero_spatial_0.yaml:42`
- SAM3 配置为 `device: cuda`：`/home/eren/codes/cap-x/env_configs/libero/franka_libero_spatial_0.yaml:53`
- privileged config 也列出相同服务，但 API 是 `FrankaLiberoPrivilegedApi`：`/home/eren/codes/cap-x/env_configs/libero/franka_libero_spatial_0_privileged.yaml:14`

## 迁移到 uni2-agent 的统一映射

| uni2-agent 高层 tool | RPent backend 候选 | CaP-X backend 候选 | 说明 |
|---|---|---|---|
| `locate_object` | `segment` + `back_project` + state artifacts | `get_object_3d_points_and_masks_from_language`，或 Reduced 的 `segment_sam3_*` + `mask_to_world_points` | planner 只传 object name，不传 pixel / xyz |
| `move_to_object` | `locate_object` 后内部 `move_to` / `move_pose` | `locate_object` 后内部 `goto_pose` / `solve_ik` / `move_to_joints` | planner 不直接传 7D action，也尽量不传 world xyz |
| `grasp_object` | `pi0_pick(prompt=...)`，或 scripted move + gripper | `sample_grasp_pose` / `plan_grasp` + `goto_pose` + `close_gripper` | 成功后返回结构化结果和 observation |
| `place_on` | locate target + `move_to` / `move_pose` + `release`，或验证后的 `place` | locate target + pose plan + `goto_pose` + `open_gripper` | 放置失败要记录 failure，不要让 planner 调底层补动作 |
| `wait` | no-op / 短步进 | no-op / 短步进 | 已有 stub，可以保留 |

## 不建议直接暴露给 planner 的函数

这些函数可以迁入 backend，但不建议成为 planner-visible tool：

- RPent：`move_to`、`move_pose`、`rotate_wrist`、`rotate_pitch`、`set_gripper`、`release`
- CaP-X：`solve_ik`、`move_to_joints`、`goto_pose`、`open_gripper`、`close_gripper`
- CaP-X utility：`depth_to_point_cloud`、`mask_to_world_points`、`pixel_to_world_point`、`transform_points`、`select_top_down_grasp`
- Privileged API：`get_object_pose`、`get_all_object_poses`、privileged `sample_grasp_pose`

原因很简单：这些函数会把 planner 拉进坐标系、控制器、关节空间、像素点和深度图细节里。我们要训练的是高层任务 agent，而不是让 LLM 每一轮都重新写一个半吊子的机器人控制程序。

## 推荐选型

### 第一阶段：先做 `capx_v0` 风格的干净 backend 壳

优先参考 CaP-X 的模块化思路：observation、segmentation、grasp planning、IK、actuation 分得比较清楚，而且外部服务边界明确。不要照搬 CaP-X 的 code generation API；在 uni2-agent 里仍然保持 `libero_run_tool(name, args)` 这个高层工具。

建议第一版 backend 名：

```yaml
tool_backend: capx_v0
```

内部模块可以长这样：

```text
CapxLiberoToolBackend
  -> observe()
  -> locate_object(object)
  -> move_to_object(object)
  -> grasp_object(object)
  -> place_on(object, target)
  -> wait()

internal services:
  -> LiberoEnvAdapter
  -> SamBackend
  -> MolmoBackend optional
  -> GraspBackend
  -> MotionBackend
```

这里 `CapxLiberoToolBackend` 只是 uni2-agent 的 backend 命名，不代表依赖 cap-x 包本身。可以先把思想迁过来，代码写在自己项目下。

### 第二阶段：接 `rpent_v0` 作为快速可跑 backend

如果明天 GPU 和 VLA 服务能接上，RPent 很适合做第二个 backend，因为它离“能动起来”近。它有 Pi0/VLA chunk、env RPC、SAM3 artifact、视频和状态日志。接的时候要把 RPent 的低层 tool surface 包起来，不让 planner 直接看到。

建议第二版 backend 名：

```yaml
tool_backend: rpent_v0
```

### 第三阶段：`privileged_v0` 只做 oracle / regression

CaP-X 的 `FrankaLiberoPrivilegedApi` 很适合拿来做：

- regression test
- oracle upper bound
- failure label 生成
- backend sanity check

但不要混进正式训练环境，否则 agent 会学到真实部署时不存在的 object pose。

## 最终建议

我建议你现在选的 planner-visible tool 保持不变：

```text
locate_object
move_to_object
grasp_object
place_on
wait
```

第一版真正实现时，不要把 RPent / CaP-X 的全部函数搬出来。正确做法是：

```text
planner
  -> libero_run_tool(name, args)
  -> LiberoRuntimeClient
  -> runtime server /run_tool
  -> tool_backend = capx_v0 or rpent_v0
  -> backend internal perception / motion / primitive
  -> env.step(...) or env.chunk_step(...)
  -> observation + structured tool_result
  -> transcript
```

如果你要我拍板：先按 **CaP-X 的模块边界** 写 `capx_v0` backend 壳，用 CaP-X 的 Reduced / SkillLibrary 函数作为内部参考；同时保留 `rpent_v0` 接口位，因为 RPent 的 Pi0/VLA primitive 对“尽快跑通真实 LIBERO manipulation”很有价值。

一句话：**CaP-X 给干净结构，RPent 给可跑 primitive；uni2-agent 只暴露高层 tool，把两者都收进 tool_backend。**
