# RoboAgent

RoboAgent 是一个以 Python 实现的机器人 Agent 最小闭环。项目只使用一个
`agent_core.Agent` 作为 Agent Loop，让视觉语言模型通过工具调用控制 frozen VLA，
再由环境适配器把动作串行落到仿真或真实机器人。

当前阶段聚焦最小可验证链路：

- `create_task_plan(steps)`：首次动作前提交有序子任务及显式状态；
- `execute_vla(step_id)`：从 TaskPlan 读取子任务指令，让 VLA 推理并串行执行动作块；
- `restage_vla()`：沿当前腕部相机 `-Z` 方向固定回撤 5 cm，为 VLA 重试重整初始状态；
- `adjust_ee_pose(action, magnitude)`：VLA 尝试失败后执行单轴、无接触的局部末端重整；
- `open_gripper()`：失败后保持末端位姿并将夹爪恢复为开放状态；
- `close_gripper()`：对准目标或把手后保持末端位姿并明确闭合夹爪；
- `return_to_home()`：累计多次 VLA 尝试后，将机械臂恢复到 episode 初始 7 维关节位姿；
- `advance_task_plan(completed_step_id, next_step_id)`：完成当前步骤并显式选择下一步骤；
- `repair_task_plan(running_step_id, reopen_step_id)`：前置条件丢失时暂停当前步骤并重开已完成的前置步骤；
- `finish(success, summary)`：提交结束请求，并由可选环境终态验证器确认成功；
- `working_memory`：组合单次 VLA 调用内的视觉观察、夹爪关键帧图像和确定性 robot state；
- `glm-4.6v-flash + pi05_libero + LIBERO` 真实闭环；
- 终端工具 trace 和可选 JSON 落盘。

当前已实现单次 VLA 调用范围内的多模态工作记忆和受限局部恢复；跨 episode 的长期语义
Memory、通用运动规划和 HarnessVLA 的其他物理辅助原语暂不在本阶段实现。

官方 LIBERO 权重的独立推理环境、版本锁及 GPU2 加载验证方式见
[HarnessVLA 官方推理环境](benchmark/robomemarena/baselines/README.md#官方权重推理环境独立验证)。
RoboMemArena 的 `harness_official` 入口使用官方 `Pi05VLAClient`，连接官方
`RLinf-Pi05-LIBERO-130-fullshot-SFT` HTTP 推理服务，保留原始 Planner、动作工具和记忆流程。
观测按官方 LIBERO 约定双轴翻转并构造本体状态，服务按原配置返回 5×7 动作，逐步交给
RoboMemArena 执行并记录官方评分。旧分支的 `harness_eval` 仍独立保留。
提示 v3 更新权重身份并移除 v2 的自加推进要求；必要的环境、预算及记忆格式说明仍明确归档。
官方源码不修改；新权重实验使用独立、带策略身份标记的记忆目录，拒绝混入旧权重记忆。
`run_harness_exploration_round.sh` 在 GPU2 上逐任务运行一轮探索，每任务一个 episode，
跨任务复用官方记忆库；探索结果与正式评测分开记录。
官方隔离入口不挂载整个 `agent/`，仅暴露共同任务参考加载器及 JSON；RoboAgent 的
提示词、skills、演化目录及其他实验记录不在项目挂载范围内。修改只对新启动会话生效。
该探索脚本从独立的 `.env_claude` 读取 Claude Code 使用的 API Key、地址和模型，
不再读取其他基线的 `.env`；DeepSeek 根地址自动转换为其 `/anthropic` 兼容入口。
批次遇到非零退出码会停止并释放本批服务，保留出错记录供排查；正常结束的任务失败仍继续下一任务。

[HiMe-Same-Backbone 接入实验](benchmark/robomemarena/baselines/HIME.md) 复用原 HiMe
的 Planner/Sentry/Memory，适配 RoboMemArena primitive，并连接相同 DeepSeek、Qwen 和
冻结 VLA。独立入口保存官方评分与模型用量，每个 episode 初始化空记忆。
HiMe 的模型连接错误最多重试两次，所有尝试均计入请求预算并单独记录。
HiMe 接入层修正已知的 summary 标签嵌套/漏闭合及中文计划状态标签，保留原始响应并记录格式适配版本；不增加模型轮次或改写动作内容。
HiMe 上游位于 `baselines/HiMe`，使用独立 Python 3.13 环境。论文对照启用 900 秒和 2,500 步预算，
记录实际 Planner 阻塞区间与 VLA WebSocket 请求。环境与复跑方式见 [HiMe 评测说明](benchmark/robomemarena/baselines/HIME.md)。
RoboMemArena 自动启动脚本明确使用仿真 Conda 环境的 Python，支持在已激活项目 `.venv` 的终端运行。
环境构造和正式 reset 前均设置任务 seed，避免构造阶段使用未受控的随机摆放。

## 架构与目录

```text
视觉 Agent（GLM / DeepSeek / OpenAI / Anthropic）
        │ tool call
        ▼
agent/roboagent ── create plan / execute VLA / local recovery / return home / advance / finish
        │
        ├── VlaPolicyPort ── OpenPI pi05_libero（:8000）
        ├── RobotPort ───── LIBERO / MuJoCo（:8001）
        └── ObserverPort ─── Qwen3-VL-8B（:8002，只读观察）
```

```text
agent/
├── ai/                    # Provider、消息和多模态协议
├── agent-core/            # 唯一 Agent Loop
└── roboagent/             # runtime、tools、OpenPI adapter 和 Observer
benchmark/libero/          # LIBERO 环境服务、适配器、评测脚本和测试
benchmark/robomemarena/    # RoboMemArena 环境、官方评分与 RobotPort 适配

```

运行依赖方向保持为：

```text
roboagent → agent_core → ai
benchmark/libero → roboagent
benchmark/robomemarena → roboagent
```

## Plan-first 执行协议

每个 episode 的计划由上层 Agent 独占决策，runtime 只校验并原子保存显式状态转换：

```text
create_task_plan（恰好一个 running）
  → execute_vla（step_id 必须指向 running 步骤）
  → 检查返回画面
  → advance_task_plan（running → completed，指定 pending → running）
  → 前置条件丢失时 repair_task_plan（当前 running → pending，先前 completed → running）
  → ...
  → finish
```

`execute_vla` 的公共参数是 TaskPlan 的 `step_id`。单次调用默认最多执行 24 个 action chunk；达到预算会
安全返回 `status=running`、`stop_reason=chunk_budget`、对应 `step_id` 和
`chunks_used=24`，计划状态不会自动改变。Agent 可以继续调用同一 `step_id`，或在确认
视觉成功后显式推进计划。对于 RoboMemArena 官方任务，Runtime 会把与当前高层任务精确匹配的 π0.5 checkpoint
primitive labels 一并提供给 Planner，作为计划步骤边界、顺序和 VLA 指令措辞的重要参考；
当前任务文本始终优先于旧标签中的对象别名或版本差异。非 RoboMemArena 任务不注入该参考。
每个物理工具结果已经包含最新双视角和机器人状态，因此 Agent 不再暴露独立 `observe` 工具。
每个新
`running` 步骤都必须先实际执行自己的 `execute_vla`，
不能沿用上一步的动作记录直接推进。环境 `task_success=true` 仍拥有最高优先级，此时可直接结束整个
episode，runtime 不会据此替 Agent 改写步骤状态。最终 TaskPlan 写入
`episode.details.task_plan`。计划和执行工具结果还会携带精简的 `active_step` 与
`plan_progress`，让 Planner 读取当前状态而不重复整份计划历史。

`execute_vla`、`restage_vla`、`adjust_ee_pose`、`open_gripper`、`close_gripper`、`return_to_home` 和
`advance_task_plan`、`repair_task_plan` 支持简短的
`decision_summary`。Agent 在调用时填写其证据依据，Runtime 将其原样保存在 tool trace 中；
Runtime 不生成或改写该理由。

`repair_task_plan` 只处理最新物理证据表明先前前置条件已经丢失的回退场景。它要求当前
`running` 步骤已执行过 `execute_vla`，只允许重开计划中位置更早且状态为 `completed` 的步骤；
Runtime 原子地保留重开步骤之前的 `completed` 前缀、把该前置步骤改为 `running`，并把它之后
的所有步骤统一退回 `pending`。重开的步骤必须重新执行自己的 `execute_vla`。

所有计划始终满足 `[completed...][running][pending...]`（终态可全部为 `completed`）。任何会破坏
该连续性的创建、推进或回退请求都不会修改计划，而会返回 `status=rejected`、当前完整计划、合法
候选步骤和重试提示；Agent 可在下一轮按工具错误恢复方式修正参数后再次调用。

Runtime 还会在当前 action chunk 内检测 `action[6]` 的 open/close 命令边沿，并用
`gripper_qpos` 变化确认真实夹爪转换，默认变化阈值为 `0.01`。转换事件只用于执行诊断和
多模态关键帧记录，不会中断 VLA action chunk 循环或提前返回 Agent；该事件本身不代表抓取
或放置成功。若证据显示抓取或放置可能失败，Agent 可根据各工具声明的边界选择继续执行或局部
调整；Runtime 不替 Agent 推导完成状态或恢复策略。
工作记忆和工具 observation 会在保留原始 `gripper_command`、`gripper_qpos` 的同时，确定性补充
`gripper_command_state=open|close|neutral` 与两指实际间距 `gripper_width_m`。前者只解释控制命令，
后者只表示物理开度；二者均不单独推导物体是否抓牢。Recovery 工具的 Reflection 还会保留工具
真实 `status`，因此发生部分位移但最终 `blocked/controller_failed` 时不会被描述为完整成功。
RobotPort 若返回终态
`failure`、`timeout` 或 `cancelled`，Runtime 会保留当前工具结果并直接结束 Agent loop，
不再请求模型执行无效重试。

## 局部末端恢复

`restage_vla` 是优先级较高的固定恢复原语：当前步骤至少执行过一次 `execute_vla` 后，它沿
当前腕部相机 `-Z_camera` 回撤 0.05 m，保持当前 `gripper_command`，并复用与
`adjust_ee_pose` 相同的闭环 OSC、工作区、收敛、取消和环境终态保护。Agent 不提供方向、幅度
或绝对位姿。结果包含请求距离、实际位移、前后 EEF pose，以及带 `RESTAGE START` 和
`RESTAGE END/latest` 标签的 external/wrist 图像。

独立的 `RecoveryState` 只机械记录当前 running 步骤自开始以来累计执行的 VLA 调用次数，不生成
失败标签，也不控制工具许可。`execute_vla` 每次实际执行后加一；`restage_vla`、
`adjust_ee_pose`、`open_gripper` 和 `return_to_home` 均不清零。第 3 次起返回轻量提醒，第 5 次起
提示重新考虑当前状态或策略，第 8 次起提示避免无证据的重复行为；这些 notice 都只是正常决策
上下文中的统计信息，Runtime 不会仅凭次数阻止 `execute_vla`，也不会强制选择某个恢复工具。
只有推进计划或开始新 episode 时才清除旧步骤计数。

`return_to_home` 不接受 Agent 提供的方向、幅度或绝对位姿。RoboMemArena 环境在 episode reset
完成 warmup 后记录 Panda 的 7 维关节位置；调用时临时使用 `JOINT_POSITION` 控制器逐步回到该固定
目标，随后恢复 VLA 使用的 `OSC_POSE` 控制器。环境中的抽屉、微波炉和物体状态不会被 reset，当前
`gripper_command` 也保持不变。每个关节控制步通过独立环境 RPC 返回并写入录像/Observer；若实际
EEF 离开工作区，或从越界位置继续远离安全区，客户端立即终止轨迹。若 480 个控制步内没有进入
`0.02 rad` 最大关节误差，结果返回 `controller_failed`。该结果不解锁或限制其他工具；Agent
根据最新物理证据自行决定后续动作。结果包含前后及目标关节位置、最大关节误差、是否到达和执行
步数，以及带 `RETURN-TO-HOME START` 和 `RETURN-TO-HOME END/latest` 标签的 external/wrist 图像。
成功的 `repair_task_plan` 会额外允许重开步骤在执行 VLA 前调用一次 `return_to_home`；该授权在
Home 调用或新一次 `execute_vla` 开始时失效，不放宽其他恢复工具的调用条件。

`adjust_ee_pose` 只用于一次 `execute_vla` 已尝试当前步骤后的小范围 re-staging。它通过
环境现有的 7D `OSC_POSE` action 改变下一次 VLA 调用的初始状态，不承担抓取、放置、插入、
开门或主动接近目标等接触操作：

```text
execute_vla 尝试失败
  → adjust_ee_pose（一个方向、一个连续幅度）
  → 检查最新 external / wrist 图像和 EEF state
  → 再调整一次，或重试同一 running step 的 execute_vla
```

支持的平移动作为 `world_z_up`、`camera_z_retract`、`camera_x_left` 和
`camera_x_right`，`magnitude` 单位为米，默认最大值为 `0.08`。支持的旋转动作为
`camera_rx_pos/neg` 和 `camera_ry_pos/neg`，单位为度，默认最大值为 `30`。世界系动作只有
`+Z_world`；相机系采用 `+X` 图像向右、`+Y` 图像向下、`+Z` 沿腕部相机视线向前，
`camera_z_retract` 对应 `-Z_cam`。第一版不提供向下、向前、camera roll、任意 SE(3)、
Agent 指定的任意关节控制或任意夹爪力控制；固定关节 home 仅由 `return_to_home` 提供，失败恢复所需的明确开合分别由 `open_gripper` 和
`close_gripper` 提供。

Runtime 将目标拆成闭环 OSC 小步，并保持调用前的 EEF 位置或朝向以及最近一次有效的
`gripper_command`。工具返回请求动作、实际平移或旋转、调整前后 EEF pose、最新 robot state
和双视角图像。配置 AABB 时，结果还包含精确目标和工作区上下界。状态为 `success`、
`blocked`、`out_of_workspace` 或 `controller_failed`。Runtime 在执行前检查目标，并在每个
OSC 控制步后检查实际 EEF；从安全区内越界或在已越界时继续远离安全区会立即停止并把拒绝
原因返回 Agent。已越界 EEF 仍允许通过 `return_to_home` 或明确缩短到安全区距离的动作恢复。
默认限制局部幅度；需要绝对工作区时可通过 `RoboAgentOptions.ee_pose_adjustment` 配置 AABB。

配置 AABB 后，`execute_vla` 也会保留 policy 生成的 action 顺序并逐 action 执行。Runtime 在每个
真实 post-step observation 后检查 EEF；首次越界立即丢弃当前 chunk 的剩余动作并返回
`status=out_of_workspace`、实际 EEF pose 和工作区上下界，并设置跨工具的
`workspace_violation` 安全锁。锁定后只有 `return_to_home` 可以执行；其他工具均被 Runtime
拒绝。home 成功、`home_pose_reached=true` 且最终 EEF 位于 AABB 内才解除锁；否则 Episode 立即以
`safety_stop` 结束。该检查不包含关节 home 中间轨迹的逐步 EEF 验证。

调整期间不会启动 Observer 推理，也不会产生新的 VLA `working_memory`。环境适配器仍按原有
路径发布物理 observation，调整后的最后一帧会成为下一次 `execute_vla` 的 Observer anchor，
用于建立恢复后的视觉起点。核心层无法在不读取环境内部状态的前提下完成完整碰撞、关节限位或
IK 预检，因此该工具只提供受限方向、幅度检查、可选工作区、闭环收敛和停滞终止。

`open_gripper` 和 `close_gripper` 同样只用于当前步骤已经执行过 `execute_vla` 后的恢复。它们
发送 EEF 零增量和明确的开合命令，并在每个真实控制步后检查 EEF 工作区；一旦越界就截断剩余
夹爪控制步。结果返回动作前后的 `gripper_qpos`、最新 robot state 和双视角；零增量不保证实际
EEF 绝对不动，这两个工具也不负责判断对准、接触、目标身份、抓取或放置，不会产生 Observer
context。

所有物理工具结果保留兼容字段 `status`，并同时返回 `execution_status` 和
`task_effect_status`。前者描述控制调用本身，后者只有在环境明确报告 `task_success=true` 时才为
`verified_task_success`，否则为 `unverified`。因此 `close_gripper` 或 `adjust_ee_pose` 的
`execution_status=success` 不能作为抓稳、抽屉打开或当前计划步骤完成的证据，Agent 必须依据后续
观测再判断任务效果。

## 单次 VLA 多模态工作记忆

`working_memory` 是一次 `execute_vla` 调用范围内的只读多模态工作记忆，不是 episode 级
世界模型。视觉哨兵和执行哨兵只读消费同一 observation stream；动作循环结束后，独立的
`VlaWorkingMemory` 将两者的本次调用结果确定性组合并随工具结果返回 Agent：

```text
env.step(action)
  → post-step 图像、robot state、gripper command
  ├→ VisualObserver：视觉文本和关键帧图像
  └→ ExecutionDiagnosticsRunner：起点、夹爪事件和终点 robot state
  → VlaWorkingMemory
  → execute_vla 工具结果
  → Agent 判断继续、只读验证或推进 TaskPlan
```

工作记忆包含三种模态：

- **文本**：保留本次调用内全部非空视觉 observation；只做完全相同文本去重，重复项保留
  最后位置，不做模糊匹配、语义合并、条数截断或进度推断。
- **图像**：Working Memory 保留调用开始和调用结束的双视角原图，并标注 sequence 和 external
  eye-to-hand、wrist eye-in-hand 视角顺序。物理工具返回后，Visual Evidence Adapter 对 external
  和 wrist 分别组织 `START → change highlight → END`；Reflection 与随后的正常决策复用同一组
  证据，OpenCV 可用时通常共六张图，不可用时退化为四张 START/END 原图。途中夹爪关键帧图像
  继续保存在 Observer 日志，不进入模型上下文。
- **机器人状态**：执行部分记录本次调用的起点、每个夹爪事件和终点，并携带
  `eef_position`、`eef_axis_angle`、`gripper_qpos` 和 `gripper_command`。

夹爪关键帧不由 VLM 自由选择。Adapter 在每个真实 `env.step(action)` 后保留本步
`action[6]`，sampler 根据 open/close 命令区间和累计 `gripper_qpos` 变化，确定性选择变化
最大的实际帧。`working_memory` 的可序列化摘要结构为：

```json
{
  "visual": {
    "observations": ["..."],
    "keyframes": [
      {
        "id": "gripper-keyframe-241",
        "sequence": 241,
        "event": "gripper_close",
        "observation": null,
        "views": ["external", "wrist"]
      }
    ]
  },
  "execution": {
    "start": {"sequence": 180, "robot_state": {}},
    "gripper_events": [
      {
        "id": "gripper-keyframe-241",
        "sequence": 241,
        "command_sequence": 180,
        "event": "gripper_close",
        "command": 0.995,
        "robot_state": {
          "eef_position": [0.0, 0.0, 0.0],
          "eef_axis_angle": [0.0, 0.0, 0.0],
          "gripper_qpos": [0.03, -0.03],
          "gripper_command": 0.997
        }
      }
    ],
    "end": {"sequence": 250, "robot_state": {}},
    "chunks_used": 3,
    "stop_reason": "chunk_budget"
  }
}
```

JSON 只保存文本和关键帧元数据；关键帧图片不进入 Agent 或 trace，继续保存在 Observer
磁盘日志。视觉关键帧和执行事件是两个独立记录，仅通过相同的 `id` 与
`sequence` 关联；VisualObserver 不接收或保存机器人状态，ExecutionDiagnosticsRunner 不保存
图像或视觉文本。配置 `--observer-record-dir` 时，完整窗口 PNG 和 `assessments.jsonl` 继续
写入本地日志，供离线分析。

Observer 使用**单并发动态窗口**：每次推理期间继续收集 observation，推理返回后立即将
尚未消费的帧作为下一窗口，不按 action chunk 建立请求队列。首个窗口等待
`min_new_observations` 个新视觉帧；后续只要有新帧就开始评估。窗口起点沿用上次输入
末帧，而不是上次推理完成时刻，因此推理期间的动作仍落在下一窗口内。
采样保持 START/END 双视角、可选夹爪转变帧和 View A 差分高亮；相邻窗口共享边界图。
一次 `execute_vla` 的评估次数由模型处理速度和执行时长决定，不固定为 20 次，也不限于两次。
调用结束时先等待进行中的评估，再补齐截至结束时刻的尾部窗口，结果进入同轮 working memory
和 Reflection。窗口越长，首尾采样越可能漏掉中间短暂事件；本次改动没有增加内部过程帧采样。
Observer 新描述要求一两句、目标不超过 40 个英文词，重点描述动作过程和状态变化；
历史描述仍成立时**原样复述优先于长度目标**。输出上限为 256 tokens，包含 JSON 格式。
`observer_trace` 和磁盘 `assessments.jsonl` 的 assessment 中记录 `finish_reason`、
`error_stage`、`error_type`：正常空观察的错误字段为空，截断标为 `truncation`，
解析失败为 `parse`，请求异常为 `request`。截断文本不进入视觉记忆。
使用 `input_sequences`、`range_start/range_end` 和 `latency_ms` 核对窗口范围与实际延迟。
夹爪关键帧不再单独发起描述评估，描述并入窗口评估的 `keyframe` 字段。
窗口文本过滤空值并做完全相同文本去重后，作为低优先级且可能出错的时序参考进入 Agent；
最新双视角和调用结束时的机器人状态始终具有更高优先级。`execute_vla` 的首个模型文本块只
包含当前步骤、确定性 execution diagnostics、执行结果、计划进度和最新 robot state，不包含
普通或关键帧 Observer 描述；完整视觉文本作为低优先级 advisory 进入同轮 Reflection，
并保存在工具 `details` 和 trace。消融实验可设置
`RoboAgentOptions(include_observer_text=False)`，只保留日志、关键帧、最新双视角和执行诊断。

一次 VLA 调用结束后该工作记忆即冻结；下一次调用从新的流位置重新收集，不跨调用累计。
Agent transcript 与 trace 保留完整历史；每次工具完成时，独立的 `EpisodeMemory` 账本会用纯
Python 机械追加 `tool`、`decision_summary`、`step_id`、`status`、`stop_reason`、`chunks_used`
和恢复动作等确定性字段，不调用额外模型，也不从 transcript 重新推导事件。请求 Planner 时，
上下文变换只渲染已归档账本并删除旧图像；最新一次 `execute_vla` 的完整 working memory、最新
工具结果以及最新物理结果仍进入模型。原始数据继续完整保存在 transcript、trace 和 Observer
磁盘日志中。未配置视觉 Observer 或视觉推理异常时，视觉部分
为空；执行部分仍返回当前可用的调用起点和终点，不影响 VLA 动作结果。

职责边界保持单向：`VisualObserver` 只描述看到了什么；`ExecutionDiagnosticsRunner` 只记录
机器人实际做了什么，不输出失败标签或恢复建议；`VlaWorkingMemory` 只组合两类证据，不修改
TaskPlan；是否完成、失败或需要恢复仍由 Agent 决定。

## Grounded State Reflection

`execute_vla`、`restage_vla`、`adjust_ee_pose` 或 `return_to_home` 正常返回后，同一个
RoboAgent 会在下一次动作决策前插入独立的 Grounded State Reflection 回合。该回合仍使用同一个
Planner 模型和 Agent Loop，但不暴露工具。V1 的 `Visual Evidence Adapter` 为 Reflection
压缩最新物理工具图像：保留确定性
diagnostics、带标签的 START/END external 与 wrist 双视角，并用 OpenCV 分别在两个 END 视角上
生成不带任务语义的像素变化高亮；不做 ROI、目标检测、成功判断或额外模型调用。OpenCV 不可用
时仍返回带标签的 START/END 双视角。Reflection 专用 diagnostics 只包含当前步骤的 id/instruction、
起点/夹爪事件/终点，并明确 `gripper_command > 0` 表示闭合、`< 0` 表示打开；不展示 Plan 的
`running` 状态、chunk budget 或 plan progress，避免这些执行边界先验锚定视觉判断；Recovery
结果保留顶层工具状态，用于区分完整成功与部分位移。完整 VlaWorkingMemory 仍保存在原始工具
结果和 trace 中。

```text
physical tool
  → latest VlaWorkingMemory
  → Visual Evidence Adapter（逐视角 START / change highlight / END，逐图标签）
  → tools=[] 的独立 Reflection LLM 请求
  → 2–4 句自然语言状态判断并写入 state_reflections
  → 恢复正常系统提示词和工具集合
  → 独立的下一动作决策 LLM 请求
```

Reflection 按“前后发生了什么变化 → 当前 running step 是 complete、incomplete 还是 uncertain
→ 是否存在重要冲突或不确定性”的顺序检查证据，只描述当前物理状态，禁止选择或推荐下一工具。
该引导由 `agent/roboagent/prompts/state_reflection.md` 独立管理。正常 Planner 使用
`tool_choice=required` 时，Runtime 只在无工具 Reflection 回合临时清除该参数，随后恢复。自然语言
结果保留在原始 Assistant transcript，同时写入 episode 结果的 `details.state_reflections`，并作为
`Latest Grounded State Reflection` 明确提供给后续正常决策回合。

V1 触发白名单仅包含 `execute_vla`、`restage_vla`、`adjust_ee_pose` 和
`return_to_home`。`create_task_plan`、`advance_task_plan`、`repair_task_plan`、`finish`、`open_gripper` 和
`close_gripper` 不触发
Reflection；环境已经进入终态时也不会为
已结束的循环额外请求模型。

迁移来源、协议边界和新增机器人适配方法见 [PORTING.md](PORTING.md)。LIBERO 的安装、
可视化和官方 baseline 见 [benchmark/libero/README.md](benchmark/libero/README.md)；
RoboMemArena 的独立环境接口见
[benchmark/robomemarena/README.md](benchmark/robomemarena/README.md)。

## 环境准备

RoboAgent 客户端使用 Python 3.11 或更高版本：

```bash
cd /path/to/robo-mem-agent
uv sync --extra libero-client --extra dev
test -f .env || cp .env.example .env
# 在 .env 中填写 ZHIPU_API_KEY；该文件已被 Git 忽略
```

LIBERO/MuJoCo 必须使用独立的 Python 3.8.13 环境：

```bash
conda activate roboagent-libero
cd /path/to/robo-mem-agent
python -m pip install -r benchmark/libero/requirements.txt
```

RoboMemArena 使用另一个独立的 Python 3.8.13 环境，不复用 `roboagent-libero`：

```bash
conda create -n roboagent-robomemarena python=3.8.13 -y
conda activate roboagent-robomemarena
python -m pip install -r benchmark/robomemarena/requirements.txt
```

本地需要存在：

```text
benchmark/.external/LIBERO
benchmark/.external/RoboMemArena
models/openpi/checkpoints/pi05_libero
models/openpi/checkpoints/pi05_robomemarena
models/Qwen3-VL-8B-Instruct
```

这些目录是独立上游源码、资产或权重，已被 `.gitignore` 排除。首次使用时按照
[benchmark/libero/README.md](benchmark/libero/README.md) 和 OpenPI 上游说明分别准备，
不要把 checkpoint、虚拟环境或完整上游仓库提交到 RoboAgent 仓库。

RoboMemArena 同样保持三进程边界：RoboAgent Python 3.11、OpenPI 自己的 policy 环境、
`roboagent-robomemarena` Python 3.8.13 环境服务；可选的只读 Observer 是第四个进程，
生成窗口图片与结构化日志，并在每次 VLA 调用结束后回传去重视觉文本。其单任务 runner 与完整命令见
[benchmark/robomemarena/README.md](benchmark/robomemarena/README.md)，且不改变下面现有
LIBERO 启动方式。

## LIBERO 完整启动

以下命令分别在四个终端运行。Observer 是可选的诊断组件，未启动时原有三进程流程不变。

终端 1，启动 frozen `pi05_libero`：

```bash
cd /path/to/robo-mem-agent/openpi
CUDA_VISIBLE_DEVICES=0 uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=pi05_libero \
  --policy.dir="$PWD/checkpoints/pi05_libero"
```

看到下面的日志表示权重和 normalization stats 已加载：

```text
Loaded norm stats from .../checkpoints/pi05_libero/assets/physical-intelligence/libero
server listening on 0.0.0.0:8000
```

终端 2，启动持有 MuJoCo 的 LIBERO 环境服务：

```bash
conda activate roboagent-libero
cd /path/to/robo-mem-agent
export LIBERO_CONFIG_PATH="$HOME/.libero"
export MUJOCO_GL=egl
python -m benchmark.libero.scripts.serve_env --host 127.0.0.1 --port 8001
```

终端 3，在项目 `.venv` 中安装 vLLM 并启动本地 Observer：

```bash
cd /path/to/robo-mem-agent
UV_CACHE_DIR=/tmp/roboagent-vllm-cache \
uv pip install --python .venv/bin/python vllm --torch-backend=auto

CUDA_VISIBLE_DEVICES=1 .venv/bin/vllm serve \
  "$PWD/models/Qwen3-VL-8B-Instruct" \
  --served-model-name robo-observer \
  --host 127.0.0.1 --port 8002 \
  --dtype bfloat16 --max-model-len 16384 \
  --gpu-memory-utilization 0.65
```

终端 4，运行视觉 Agent 闭环并保存工具 trace：

```bash
cd /path/to/robo-mem-agent
uv run --env-file .env --extra libero-client \
  python -m benchmark.libero.scripts.roboagent_eval \
  --policy-port 8000 \
  --env-port 8001 \
  --suite libero_spatial \
  --task 0 \
  --seed 0 \
  --planner glm \
  --max-vla-chunks-per-call 24 \
  --planner-retries 12 \
  --planner-retry-delay-ms 15000 \
  --observer-base-url http://127.0.0.1:8002/v1 \
  --observer-model robo-observer \
  --observer-timesteps 5 \
  --observer-min-new-observations 10 \
  --observer-gripper-threshold 0.01 \
  --observer-record-dir benchmark/libero/logs/observer-task0 \
  --trace-file benchmark/libero/logs/glm-task0.json
```

终端会展示 episode 结果和完整 `tool_trace`。日志包含工具参数、执行/拦截状态、结果、
observation 摘要和图像数量，不包含 API Key 或图像 base64；`benchmark/libero/logs/`
已被 Git 忽略。

LIBERO adapter 在每个串行 `env.step()` 返回后，把 observation 发布到只读视觉流；Observer
不持有机器人端口、不调用 `observe/env.step`，也不知道 action chunk。`VisualObserver` 在积累至少
`--observer-min-new-observations` 个未消费 observation 后独立启动单个 Qwen 请求；推理期间
的新帧进入下一非重叠区间，同时最多只有一个请求。默认窗口为一个上一区间末帧锚点加
最多四个新 timestep，始终保留新区间末帧；配置夹爪阈值后，会根据 VLA 夹爪命令区间选择
累计 `gripper_qpos` 变化最大的实际帧。输入明确按 oldest→newest 排列，每个 timestep 依次发送外部固定
eye-to-hand 视角和腕部 eye-in-hand 视角。相同 TaskStep 的下一窗口会携带上一条待验证的
视觉 observation；步骤切换时额外标明上一条和当前 VLA instruction。上一条 instruction
只发送给切换后的首个有效窗口；`VisualObserver` 只原样转发和替换 observation，不解析或合并语义。
现有 sampler 继续根据实际 `gripper_qpos` 检测夹爪开合变化，只有该变化帧可以成为关键帧。

区间、输入 sequence、夹爪关键帧 sequence、视觉 observation 和延迟会写入 episode 级
`observer_trace`。Observer 不会停止动作、宣告成功或修改规划；Runtime 独立使用动作命令与
`gripper_qpos` 确认 chunk 边界检查点。每次 `execute_vla` 的动作
循环结束后，Runtime 只负责结束两个哨兵的调用边界；`VlaWorkingMemory` 把非空文本、关键帧
图像和确定性 robot state 统一放入工具结果的 `working_memory`。trace 只保存元数据和图像数量，
不保存 base64。`--observer-record-dir` 额外保存供本地标注使用的 PNG 窗口和 JSONL；
不需要离线数据时可以省略该参数。

## RoboMemArena 单任务启动

VoLoAgent 的独立依赖安装、SAM3＋GraspGen 抓取（放置使用 VLA）、可选 RGB-D
接口、可选官方任务 primitive 参考和单任务评测入口见
[VoLo 接入说明](benchmark/robomemarena/baselines/VOLO.md)。安装与静态检查不启动 GPU 服务。
该说明同时记录 DeepSeek 短响应兼容补丁、单 GPU 共用配置及容器驱动库挂载方式；
代理启动保存实际源码补丁，中断回合标记为无效验证。
启用官方参考时，VoLo 分解、监控和重规划共用该参考；监控完成判断保留固定上游原文，
包括“目标物体到达目的地”条件，不再应用完成语义修改补丁。
后续运行在 episode 的 `vlm_requests/` 保存实际 VLM 请求与响应，不记录客户端凭据。
VoLo 监控沿用上游 LIBERO 的 40 个控制步间隔；在当前 20 Hz 环境下为每 2 秒仿真时间一次
（0.5 Hz），可通过 `VOLO_CHECK_INTERVAL` 覆盖，实际耗时取决于推理与仿真速度。
论文对照从 `baselines/VoLoAgent` 的独立环境启动，使用 DeepSeek 规划、Qwen3.8-27B 监控和共享 π₀.₅。
评测预算、请求时序和恢复入口见 [VoLo 论文对照](benchmark/robomemarena/baselines/VOLO.md#论文对照)。

RoboMemArena 使用相同的只读 Observer 契约，但环境和 policy 配置独立。以下命令分别在四个
终端运行；不需要 Observer 日志时可省略终端 3 和终端 4 的 `--observer-*` 参数。

终端 1，在 GPU 0 启动 `pi05_robomemarena`：

```bash
cd /path/to/robo-mem-agent
CUDA_VISIBLE_DEVICES=0 models/openpi/.venv/bin/python \
  benchmark/.external/RoboMemArena/third_party/openpi_minimal/scripts/serve_policy.py \
  --port 8000 \
  policy:checkpoint \
  --policy.config=pi05_robomemarena \
  --policy.dir="$PWD/models/openpi/checkpoints/pi05_robomemarena"
```

终端 2，启动固定上游 commit 对应的 Python 3.8.13 环境和官方评分服务：

```bash
conda activate roboagent-robomemarena
cd /path/to/robo-mem-agent
export ROBOMEMARENA_ROOT="$PWD/benchmark/.external/RoboMemArena"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export NUMBA_CACHE_DIR=/tmp/roboagent-robomemarena-numba
export MPLCONFIGDIR=/tmp/roboagent-robomemarena-matplotlib
python -m benchmark.robomemarena.scripts.serve_env --host 127.0.0.1 --port 8001
```

终端 3，在 GPU 1 启动只读 Observer：

```bash
cd /path/to/robo-mem-agent
CUDA_VISIBLE_DEVICES=1 .venv/bin/vllm serve \
  "$PWD/models/Qwen3-VL-8B-Instruct" \
  --served-model-name robo-observer \
  --host 127.0.0.1 --port 8002 \
  --dtype bfloat16 --max-model-len 16384 \
  --gpu-memory-utilization 0.65
```

终端 4，运行 Task 1、seed 100，并保存 Agent、Observer 和官方评分日志：

```bash
cd /path/to/robo-mem-agent
RUN_DIR="benchmark/robomemarena/logs/task1-seed100-$(date +%Y%m%d-%H%M)"
mkdir -p "$RUN_DIR"
.venv/bin/python -m benchmark.robomemarena.scripts.roboagent_eval \
  --policy-host 127.0.0.1 --policy-port 8000 \
  --env-host 127.0.0.1 --env-port 8001 \
  --task 1 --seed 100 \
  --planner faux \
  --max-vla-chunks-per-call 24 \
  --max-agent-turns 24 \
  --observer-base-url http://127.0.0.1:8002/v1 \
  --observer-model robo-observer \
  --observer-timesteps 5 \
  --observer-min-new-observations 10 \
  --observer-gripper-threshold 0.01 \
  --observer-record-dir "$RUN_DIR/observer" \
  --trace-file "$RUN_DIR/trace.json"
```

一次实验的所有输出统一放在同一个时间戳目录中，不把 Agent trace 和 Observer 窗口拆成
两个并列路径：

```text
benchmark/robomemarena/logs/task1-seed100-20260902-0316/
├── trace.json
└── observer/
    ├── assessments.jsonl
    └── window-0001/
        ├── t01-external.png
        └── t01-wrist.png
```

后续真实联调沿用 `--trace-file "$RUN_DIR/trace.json"` 和
`--observer-record-dir "$RUN_DIR/observer"`；已有日志不自动迁移，以免破坏历史引用。

日常评测可用统一脚本启动 policy、环境和 Observer，并将一次实验的配置、日志、视频、trace
和官方格式汇总写入同一个时间戳目录：

```bash
bash benchmark/robomemarena/scripts/run_eval.sh
# 或指定一份非密钥配置
bash benchmark/robomemarena/scripts/run_eval.sh \
  benchmark/robomemarena/configs/qwen_recovery.env
```

默认配置使用 Qwen3.8 Max Planner、20-step action chunk 和三个独立 GPU；端口、GPU、任务、
seed 与预算均在 `benchmark/robomemarena/configs/qwen_recovery.env` 管理。API 凭据仍只从本地
`.env` 读取，不复制到运行目录。默认不传 `--observer-record-dir`，因此 Observer 继续在线参与
工作记忆，但不保存逐窗口 PNG；设置 `SAVE_OBSERVER_WINDOWS=1` 可恢复这些离线窗口。

`deepseek.env` 使用 `deepseek-flash` Planner 和本地 `Qwen/Qwen3.8-27B` Observer。
Observer 从 `models/Qwen3.8-27B` 加载 BF16 权重，通过 `models/.venv-observer/bin/vllm`
在两张 GPU 上做 tensor parallel 推理。本机使用 vLLM 0.24.0 的 CUDA 12.9 构建。
需要转发网络请求时，可设置
`ROBOAGENT_HTTP_PROXY=http://127.0.0.1:<代理端口>`，本地模型服务仍通过 loopback 直连。

完整评测使用常驻服务批跑脚本。Policy、环境和 Observer 只启动一次，环境会在每个任务开始时重置。
必需服务启动超时或重启后仍不可用时，批跑退出，不继续执行缺少服务的任务。
修复服务后可用同一批次 ID 续跑；服务缺失期间产生的旧运行需要重跑。

```bash
# deepseek 配置跑 task 1..26，或传 CONFIG [START] [END]
bash benchmark/robomemarena/scripts/run_batch_persistent.sh \
  benchmark/robomemarena/configs/deepseek.env
```

并行使用多个常驻批次时，可让它们指向同一个已启动的 Observer，并在批次配置中设置
`OBSERVER_EXTERNAL=1`。各批次使用自己的 policy 和环境端口，同时检查共享 Observer 的健康状态。

在同一个 DeepSeek 和 Qwen 配置上评测已保存的演化 Skill 时，可以传入 `TASK_EXECUTION_SKILL`。

```bash
TASK_EXECUTION_SKILL=/path/to/SKILL.md \
  bash benchmark/robomemarena/scripts/run_batch_persistent.sh \
    benchmark/robomemarena/configs/deepseek.env 1 26 deepseek-evolved
```

单任务失败不中断，逐任务状态与产物目录追加写入
`benchmark/robomemarena/logs/batch/current.status`。

重新采集 Task 1–26、统一 seed 50 的第一批无 Skill 经验时，传入第四个 `BATCH_ID`：

```bash
bash benchmark/robomemarena/scripts/run_batch_persistent.sh \
  benchmark/robomemarena/configs/deepseek_evolution_v3_seed50.env \
  1 26 evolve-v3-collection-deepseek-seed50
```

全部产物会集中写入 `benchmark/robomemarena/logs/batches/evolve-v3-collection-deepseek-seed50/`；
其中 `runs/` 保存逐任务实验，`services/` 保存每次启动的服务日志，`manifest.jsonl` 记录任务、
seed、相对产物路径和完整性。使用相同命令续跑时会跳过已有完整产物的任务；配置快照发生变化时
会拒绝混入同一个 batch。

```text
benchmark/robomemarena/logs/task3-seed100-qwen3.8-max-YYYYMMDD-HHMM/
├── trace.json
├── evaluation.json
├── episodes.tsv
├── task_summary.tsv
├── summary.json
├── aggregate.json
├── run_config.json
├── eval_config.env
├── run.log
├── services/
└── videos/
    ├── task3_success|failure_ep0.mp4
    └── task3_success|failure_ep0_wrist.mp4
```

`external` 视频对应固定 eye-to-hand/agentview，`wrist` 视频对应 eye-in-hand。TSV 和汇总 JSON
沿用官方 batch evaluator 的文件名与百分制 TSR 表达；评分值仍直接来自环境的官方
`metrics()`，本地 writer 不重新推断阶段是否成功。`trace.json` 保留 Agent 的完整工具、工作
记忆、Reflection 与原始 evaluation，便于逐轮审计。

高层视觉 Planner 也可通过 DashScope OpenAI-compatible API 使用 Qwen3.8 Max：在 `.env`
配置 `LLM_API_KEY`、`LLM_MODEL_NAME` 和 `LLM_BASE_URL` 后传入 `--planner qwen`。若设置
`LLM_DISABLE_THINKING=true`，评测入口会发送 DashScope 对应的
`enable_thinking=false`。
OpenAI-compatible Provider 会按 `--planner-retries` 重试发生在首个有效模型输出之前的
连接或超时错误；已经产生文本或工具调用增量后不会重放请求，以免重复执行动作。

`faux` 只替代高层工具调用：它先创建一个运行中的 smoke-test 步骤，再按单次预算重复调用
VLA，最后显式完成计划；低层动作仍来自真实 `pi05_robomemarena`。它用于验证协议接线，不
评价真实任务拆分质量。Task 1 默认最多执行 2500 个有效 action steps；满足官方成功和延迟
结束条件时会提前停止。trace 文件包含
`episode.details.tool_trace`、`episode.details.observer_trace` 和独立的 `evaluation`；窗口目录
保存双相机 PNG 与 `assessments.jsonl`。Observer 结果不会改写 Agent observation、feedback
或规划，但单次 VLA 的文本、关键帧 robot state 和关键帧图像会进入对应工具结果。更完整的环境安装、评分语义和 smoke 命令见
[benchmark/robomemarena/README.md](benchmark/robomemarena/README.md)。

## 已验证结果

底层真实链路在引入 Plan-first 协议前已经验证：

```text
glm-4.6v-flash → RoboAgent → execute_vla → pi05_libero
→ LIBERO/MuJoCo task_success=true → finish(success=true)
```

当前 Plan-first 工具顺序、非法调用拦截、显式两步推进和 24-chunk 边界由 runtime 回归测试
覆盖。2026-08-31 使用 DeepSeek 视觉 Planner、真实 `pi05_robomemarena` 和 Qwen Observer
完成 RoboMemArena Task 1、seed 100 联调；Planner 生成了两个可视觉验证的完整步骤：先将
cookies 放入篮子，再将 tomato sauce 放入同一篮子。实际工具顺序为：

```text
create_task_plan
→ execute_vla × 2
→ advance_task_plan
→ execute_vla × 2
→ advance_task_plan
→ finish
```

官方评分和运行摘要为：

```text
status=success
agent_turns=8
tool_calls=8
official_step=1096
CSR=100.0
TSR=1
cookies_in_basket=true
tomato_sauce_in_basket=true
observer_windows=59
gripper_keyframes=4
```

Observer 当前只描述视觉 observation，不判断步骤是否完成，也不直接终止动作；Runtime 会在
每次 VLA 调用结束时把本次多模态 observation context 回传 Agent。
本次完整本地 trace 位于被 `.gitignore` 排除的
`benchmark/robomemarena/logs/task1-seed100-goal-state-prompt-v2-deepseek-20260831.json`。

## RoboAgent-Evolve V3

V3 将事实、规律和行动规则严格分层：

```text
Episode → 高价值 Experience → Wiki Pattern → 单一 Skill 修改 → Replay/Gate
```

新工作区是 `evolution_workspaces/robomemarena-v3/`。旧
`evolution_workspaces/robomemarena/` 仅作为 legacy 审计数据；初始化和编译命令不会迁移或改写它。

```bash
.venv/bin/python -m roboagent.evolution init \
  --workspace evolution_workspaces/robomemarena-v3

.venv/bin/python -m roboagent.evolution compile \
  --workspace evolution_workspaces/robomemarena-v3 \
  --batch-manifest benchmark/robomemarena/logs/batches/<batch-id>/manifest.jsonl \
  --policy-role baseline

.venv/bin/python -m roboagent.evolution validate \
  --workspace evolution_workspaces/robomemarena-v3
```

V3 只接受包含 `runtime_revision`、`evaluator_revision` 和 `data_role` 的新运行；同时保存
`cohort_id`、Skill 版本/哈希、Planner/Observer 模型、seed 和 task。三个固定数据角色是
`experience_collection`、`candidate_gate` 和 `final_evaluation`，但角色不再绑定某个 seed。
第一版可以固定 seed 50 重复运行；不同角色必须使用新的 run/Episode。修改 Runtime 或 evaluator
后必须更新相应 revision，不能复用旧标识。

Experience Unit 是高价值事件，不再与每个 ToolCall 一一对应。编译器自动保留官方 stage 首次变化、
明确 failure/controller failure/rejection，以及待人工确认的成功 plan repair；普通 `running`
`execute_vla` 和无明显效果的调用只保留在 Episode。掉落恢复窗口、副作用和决策矛盾通过 JSONL
审查清单导入：

```bash
.venv/bin/python -m roboagent.evolution compile-events \
  --workspace evolution_workspaces/robomemarena-v3 \
  --manifest evolution_workspaces/robomemarena-v3/experiences/reviewed_events/<review>.jsonl

.venv/bin/python -m roboagent.evolution compile-pairs \
  --workspace evolution_workspaces/robomemarena-v3 \
  --manifest evolution_workspaces/robomemarena-v3/experiences/paired_comparisons/<pairs>.jsonl
```

事件 Unit 使用 schema 3，并把环境证据与模型/维护者解释分开：

```json
{
  "schema_version": 3,
  "unit_kind": "event",
  "event_type": "successful_state_change",
  "episode_id": "episode-...",
  "call_span": {"start_call_id": "...", "end_call_id": "...", "call_ids": ["..."]},
  "provenance": {},
  "evidence": {},
  "interpretation": {"failure_domain": null, "confidence": "high"},
  "learning": {"learning_status": "active", "skill_eligible": true, "exclusion_reasons": []}
}
```

派生 comparison 固定为 `derived=true`、`independent_evidence=false`。Wiki 成熟度按可用于 Skill 的
唯一来源 Episode 计数：`single-shot` 至少 1 个，`probable` 至少 2 个，`verified` 还必须覆盖
至少 2 个 task。Skill manifest 只能引用带 maturity 和内容 SHA-256 的 Wiki Pattern，不能直接
引用 Experience；Candidate 相对 Parent 只能修改一个 `SKILL.md` 二级章节。

```json
{
  "parent_version": "vNNN",
  "change_summary": "一个逻辑修改",
  "changed_section": "一个 SKILL.md 二级章节",
  "source_patterns": [
    {"pattern_id": "...", "maturity": "probable", "content_sha256": "..."}
  ]
}
```

`single-shot` Candidate 可运行 Historical Replay，但不能进入正式 rollout Gate。正式 Candidate
必须由 `probable` 或 `verified` Pattern 支持，并依次通过：

Replay 使用与在线决策一致的工具 schema，包括 `repair_task_plan` 和
`return_to_home`，但不会执行真实机器人动作。

为 Planner 配置重试时，单次 Assistant 流会在完整结束后才提交给 Agent；中途连接
断开会丢弃未提交的 partial 并安全重试，不会重复已经执行的机器人工具。

```bash
.venv/bin/python -m roboagent.evolution replay \
  --workspace evolution_workspaces/robomemarena-v3 \
  --candidate evolution_workspaces/robomemarena-v3/skills/candidates/v001 \
  --planner deepseek --model deepseek-v4-flash-vision-exp

.venv/bin/python -m roboagent.evolution compare-rollouts \
  --phase candidate-gate \
  --workspace evolution_workspaces/robomemarena-v3 \
  --candidate evolution_workspaces/robomemarena-v3/skills/candidates/v001 \
  --manifest evolution_workspaces/robomemarena-v3/runs/v001-seed51-pairs.json

.venv/bin/python -m roboagent.evolution promote \
  --workspace evolution_workspaces/robomemarena-v3 \
  --candidate evolution_workspaces/robomemarena-v3/skills/candidates/v001
```

Gate 会从 Wiki Evidence 递归取得来源 Episode 的运行目录并拒绝复用。同一 pair 还必须匹配
task、seed、模型、Runtime/evaluator revision 和运行参数。Candidate Gate 使用至少一个 Parent
锚点和两个不同 Candidate rollout，要求 Candidate 严格更优且不允许 Parent 成功任务退化；通过
Replay 和 Candidate Gate 后即可晋升。Final Evaluation 保留为激活后的可选审计，不阻塞晋升。
被拒 Candidate 保留在 history；结果、
Pattern 关联和 Skill diff 追加到 `wiki/skill-impact.md`。

## 测试

```bash
cd /path/to/robo-mem-agent
uv run --extra libero-client --extra robomemarena-client --extra dev \
  pytest agent benchmark/libero/tests benchmark/robomemarena/tests -q
uv run --extra libero-client --extra dev mypy agent/roboagent
```

测试只验证代码和适配边界；真实端到端验证仍需同时启动 OpenPI、LIBERO 和模型 Provider。

RoboMemArena 启动脚本对本地 Observer 的健康检查最多等待 900 秒，其他服务为 180 秒。
Observer 本地健康检查绕过代理，每次 HTTP 探测最多等待 5 秒。
