# RoboMemArena Benchmark

## 仓库介绍

本目录提供 RoboMemArena 的环境服务、官方阶段评分包装、Python 3.11 RPC 客户端和
`RobotPort` 适配。运行时直接依赖上游官方 LIBERO fork、BDDL 与阶段检查器，不复制评分
实现。完整 runner 可以连接已下载的 `pi05_robomemarena`，不需要再次微调。

上游仓库固定为 commit：

```text
cc156e519990ae43cf3b64281a548724f428fbbd
```

本地目录结构：

```text
benchmark/
├── .external/RoboMemArena/       # 固定 commit 的上游仓库，不提交
└── robomemarena/
    ├── config.py                  # 共享：任务与评分参数（3.8 可导入）
    ├── environment.py             # 共享：最小环境契约类型
    ├── preprocessing.py           # 共享：官方 flipud、INTER_AREA 和 state 语义
    ├── server/                    # Python 3.8 环境服务进程专用
    │   ├── upstream.py            # 定位并加载固定 commit 官方 fork
    │   ├── evaluation.py          # 增量调用上游阶段检查器
    │   └── service.py             # 持有 MuJoCo 与官方评分状态（原 env_server）
    ├── client/                    # Python 3.11 RoboAgent 侧（只被 >=3.11 导入）
    │   ├── adapter.py             # RoboMemArena → RobotPort
    │   ├── remote_env.py          # 环境 RPC 客户端
    │   ├── observation.py         # 观测构造与 OpenPI payload
    │   ├── video_recording.py     # 双视角 MP4 流式写入
    │   ├── observer_recording.py  # 双视角窗口与 assessment 本地记录
    │   └── artifacts.py           # 官方格式 TSV/JSON 评测产物
    ├── scripts/
    │   ├── serve_env.py           # 启动环境服务
    │   ├── smoke_env.py           # 环境接口 smoke，不调用权重
    │   └── roboagent_eval.py      # RoboAgent + OpenPI + 环境闭环
    └── tests/                     # 镜像 server/、client/ 拆分
```

`server/` 与根目录共享模块必须保持 Python 3.8 可导入；`client/` 依赖 RoboAgent
（Python 3.11），任何服务端模块不得引用它。

## 独立环境安装

RoboMemArena 必须使用独立环境，不复用或修改 `roboagent-libero`：

```bash
cd "$(git rev-parse --show-toplevel)"
conda create -n roboagent-robomemarena python=3.8.13 -y
conda activate roboagent-robomemarena
python -m pip install -r benchmark/robomemarena/requirements.txt
```

Python 3.11 客户端依赖：

```bash
uv sync --extra robomemarena-client --extra dev
```

## 启动环境服务

```bash
cd "$(git rev-parse --show-toplevel)"
conda activate roboagent-robomemarena
export ROBOMEMARENA_ROOT="$PWD/benchmark/.external/RoboMemArena"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export NUMBA_CACHE_DIR=/tmp/roboagent-robomemarena-numba
export MPLCONFIGDIR=/tmp/roboagent-robomemarena-matplotlib

python -m benchmark.robomemarena.scripts.serve_env \
  --host 127.0.0.1 \
  --port 8001
```

服务端使用官方 model-agnostic adapter 参数：Task 1–26、默认 seed 100、warmup 10 步、
最多 2500 个有效动作步、相机 `480×640`、`ignore_done=True`、
`reward_shaping=True`、`control_freq=20` 和 `initialization_noise=None`。

## Python 3.11 客户端 smoke

保持服务端运行，在项目 Python 3.11 终端执行：

```bash
uv run --extra robomemarena-client \
  python -m benchmark.robomemarena.scripts.smoke_env \
  --host 127.0.0.1 \
  --port 8001 \
  --task-id 1 \
  --seed 100
```

该命令依次执行 `reset → observe → dummy step → metrics → close`。`observe()` 不推进仿真，
每次 `step()` 只接收一个有限 7D action；动作块由 `RoboMemArenaRobotPort` 逐条串行执行。

## Observation 与评分边界

图像预处理与上游 recorder 保持一致：只做 `np.flipud`，不水平翻转，然后使用 OpenCV
`INTER_AREA` 强制缩放为 `256×256`。state 由末端位置、四元数转换后的 axis-angle 和
gripper state 拼接。`build_openpi_input()` 构造与 `pi05_robomemarena` 匹配的 OpenPI payload。

默认 RoboAgent 侧只能获得：

- 主相机与腕部相机图像；
- robot state；
- episode 终止时的 `task_success`。

VoLo 抓取客户端可通过 `RoboMemArenaConfig(enable_depth=True)` 额外申请同帧
agentview 米制深度、相机内参和 OpenCV camera-to-world 外参。
该选项默认关闭，不改变现有 RGB/VLA 预处理；它不导出物体位姿、分割真值或阶段状态。
SAM3＋GraspGen 的依赖、图像坐标约定和启动入口见 [VoLo 接入说明](baselines/VOLO.md)。

CSR、TSR、阶段名、阶段完成状态和计数诊断只能通过 `metrics()` evaluation 通道读取，绝不
写入 `Observation.metadata` 或 `ExecutionFeedback.details`。

`metrics()` 还会为离线 RoboAgent-Evolve 编译器保存 `stage_events`，记录官方顺序阶段首次
通过检查的 `environment_step`。该字段只随 episode 结束后的 evaluation artifact 返回，不会在
执行中暴露给 Agent，也不改变任何官方成功条件。

固定上游版本的 Task 8、9、15、22 stage 实现曾与任务定义和官方测试不一致。本地上游补丁
恢复 Task 8/9/15 的 frypan 前置放置阶段，并恢复 Task 22 在两次倾倒后的放下酱瓶、打开微波炉
和放入 cookies 阶段。Task 22 的最终 `Close_Microwave` 保留为 optional，不计入 CSR/TSR；
各 counting task 最后的容器归位动作仍保持该版本官方测试口径，不额外计入 TSR。
在固定 checkout 上依次应用上游修订
[`a1099e0`](https://github.com/OpenHelix-Team/RoboMemArena/commit/a1099e0b8e329022f671215f2822cf58f22906c4)
中的 stage 检查器和 Task 22 BDDL 修改，以及
[`2057e5b`](https://github.com/OpenHelix-Team/RoboMemArena/commit/2057e5b50ec170f876549ebcb23454e4155bc6a7)
中的 stage 检查器修改，即可恢复上述计分条件。

Task 11 的两份 reference JSON 也已从过时的 chocolate 描述同步为 BDDL、主服务和 evaluator
实际使用的 butter，确保 reference planner 与正式评测对象一致。

官方结束语义保持如下：

- 非计数任务完成 required stages 后继续执行 200 步，再结束；
- 计数倾倒任务完成第二次倾倒后继续监控 30 步；
- 监控窗口内检测到第三次倾倒立即失败；
- 到 2500 个有效 action steps 时终止并按当前官方阶段状态评分。

当 Agent 完成全部 TaskPlan 步骤并调用 `finish(success=true)` 时，通用 Runtime 会委托
RoboMemArena adapter 执行一次终态收敛。环境服务使用官方 dummy action 推进上述
post-goal 或 extra-pour 窗口，直到 evaluator `done=true` 或收敛预算耗尽；LIBERO 和未配置
该接口的机器人不会执行这一步。CSR、TSR 和阶段状态仍不会进入 Agent 上下文。

## 单任务闭环与 Observer 日志

Observer 未启用时仍是原有三个进程；需要观察器日志时，以下命令分别在四个终端运行。
权重已经位于 `models/openpi/checkpoints/pi05_robomemarena`。

终端 1，使用 RoboMemArena 上游附带的 `pi05_robomemarena` 配置启动 policy server：

```bash
cd "$(git rev-parse --show-toplevel)"
models/openpi/.venv/bin/python \
  benchmark/.external/RoboMemArena/third_party/openpi_minimal/scripts/serve_policy.py \
  --port 8000 \
  policy:checkpoint \
  --policy.config=pi05_robomemarena \
  --policy.dir="$PWD/models/openpi/checkpoints/pi05_robomemarena"
```

终端 2，按上文“启动环境服务”运行 Python 3.8.13 环境服务。

终端 3，在项目 Python 3.11 环境中启动只读 Qwen Observer：

```bash
cd "$(git rev-parse --show-toplevel)"
CUDA_VISIBLE_DEVICES=1 .venv/bin/vllm serve \
  "$PWD/models/Qwen3-VL-8B-Instruct" \
  --served-model-name robo-observer \
  --host 127.0.0.1 --port 8002 \
  --dtype bfloat16 --max-model-len 16384 \
  --gpu-memory-utilization 0.65
```

终端 4，运行 Task 1 的真实 VLA 权重、faux 高层 Planner，并保存完整日志：

```bash
cd "$(git rev-parse --show-toplevel)"
.venv/bin/python -m benchmark.robomemarena.scripts.roboagent_eval \
  --policy-host 127.0.0.1 \
  --policy-port 8000 \
  --env-host 127.0.0.1 \
  --env-port 8001 \
  --task 1 \
  --seed 100 \
  --planner faux \
  --max-vla-chunks-per-call 24 \
  --observer-base-url http://127.0.0.1:8002/v1 \
  --observer-model robo-observer \
  --observer-timesteps 5 \
  --observer-min-new-observations 10 \
  --observer-gripper-threshold 0.01 \
  --observer-record-dir benchmark/robomemarena/logs/observer-task1-seed100 \
  --trace-file benchmark/robomemarena/logs/task1-seed100.json
```

`faux` 只替代高层模型的工具调用，低层动作仍由真实 `pi05_robomemarena` 推理。它先创建
一个运行中的 smoke-test 计划步骤，再按预算重复调用 VLA，最后显式推进计划并结束；这只
验证 Plan-first 协议，不评价任务拆分质量。评测脚本统一令每个 VLA action chunk 执行 20 步；
每次 `execute_vla` 最多使用
`--max-vla-chunks-per-call` 个 chunk，默认 24，达到预算只把控制权返回 Planner。只验证接线
时可加 `--max-steps 10` 快速结束，完整 Task 1 使用默认 2500 步上限。

需要真实视觉 Planner 时可将 `--planner` 改为 `glm`、`deepseek`、`qwen`、`openai` 或
`anthropic`，并按对应
Provider 配置密钥。真实 Planner 默认最多运行 24 个 Agent turn；复杂任务或免费接口重试时
可以显式传入 `--max-agent-turns 30`。文本回复不会提前结束 episode，runtime 会持续要求
模型调用合法工具，直到 `finish` 或达到该上限。episode 结束后 runner 才从独立
`metrics()` 通道打印官方 evaluation。

DeepSeek 官方视觉模型使用 `.env` 中的 `LLM_API_KEY` 和 `LLM_MODEL_NAME`，启动参数为
`--planner deepseek`；评测入口固定使用官方 OpenAI-compatible 地址
`https://api.deepseek.com/v1`。

Qwen3.8 Max 通过 DashScope OpenAI-compatible API 接入，启动参数为 `--planner qwen`；读取
`.env` 中的 `LLM_API_KEY`、`LLM_MODEL_NAME` 和 `LLM_BASE_URL`。当
`LLM_DISABLE_THINKING=true` 时，入口会按 DashScope 契约发送
`enable_thinking=false`。
`--planner-retries` 同时覆盖请求建立和首个有效模型输出之前的流式连接错误；若响应已经输出
文本或工具调用增量，则不会重放请求。

`--trace-file` 保存一个 JSON 对象：`episode` 包含 Agent tool trace 和结构化
`observer_trace`，`evaluation` 包含 episode 结束后单独读取的官方 CSR/TSR。Observer 图片
不会写成 base64；`--observer-record-dir` 会按 `window-0001/` 保存每个时序的
`external` eye-to-hand、`wrist` eye-in-hand PNG，并把上一条视觉 observation、切换前
instruction、夹爪变化候选帧和当前视觉记录追加到 `assessments.jsonl`。这些文件位于已忽略的
`benchmark/robomemarena/logs/`，便于离线检查观察器的视觉描述和关键帧。

RoboMemArena adapter 只在 `reset` 和每个真实 `env.step()` 后向只读 observation stream
发布帧，并在 post-step observation 中保留本步 `action[6]` 夹爪命令。普通 `observe()` 不发布
也不推进仿真。Observer 不持有 RobotPort，不调用环境，其判断不会直接修改 Agent
observation、feedback 或规划。Observer 使用单并发动态窗口：推理期间持续收集新帧，返回后
立即评估自上次输入末帧起尚未消费的区间，不按 action chunk 排队。首窗等待
`--observer-min-new-observations` 个新视觉帧，后续有新帧即可评估；短调用不足阈值也会在
结束时评估尾部。采样仍为 START/END 双视角、可选夹爪转变帧与 View A 差分高亮，
相邻窗口共享边界图，历史连续性经上一条观察文本传递。调用结束时等待当前评估并补齐尾部，
再将结果写入本次 working memory 和 Reflection。窗口次数随处理速度变化，没有每调用
两次或每 chunk 一次的固定限制；较长窗口的首尾采样仍可能漏掉中间短暂事件。
新描述要求一两句、目标不超过 40 个英文词，历史描述仍成立时保留原样复述，优先于长度目标。
输出上限为 256 tokens；trace 和 `assessments.jsonl` 的 assessment 中新增
`finish_reason`、`error_stage`、`error_type`，区分正常空观察与截断、解析、请求错误。
可用 `input_sequences` 与 `latency_ms` 检查实际观察范围和延迟。夹爪关键帧描述不单独评估，
并入窗口评估的 `keyframe` 字段。按命令符号区间选择累计 qpos 变化最大的 open/close
关键帧，并把去重文本、关键帧 robot state 和带标签的双视角图像统一作为
`observation_context` 返回 Agent。trace 不保存图像 base64，窗口 PNG 仍保存在本地日志目录。
最终 TaskPlan 单独保存在 `episode.details.task_plan`。

RoboMemArena runner 默认给局部 EEF 恢复配置世界坐标 AABB：最小值
`[-0.45, -0.40, 0.30] m`，最大值 `[0.35, 0.65, 1.20] m`。该范围覆盖已有完整成功轨迹并
保留余量，约束 `adjust_ee_pose`、`restage_vla`，并检查 `execute_vla`、`open_gripper` 和
`close_gripper` 每个 action 执行后的真实 EEF；Runtime 不改写 VLA action。可用
`--eef-workspace-min-m X Y Z` 和
`--eef-workspace-max-m X Y Z` 显式覆盖。目标越界会在执行前返回 `out_of_workspace`；控制期间
实际 EEF 越界或继续远离安全区也会停止。VLA 首次出现真实越界时会立即丢弃当前 action chunk
的剩余动作，并返回越界 EEF pose、工作区上下界和拒绝原因，供
Agent 改选固定的关节空间 `return_to_home` 或朝安全区内部的恢复动作。`return_to_home` 使用
episode warmup 后记录的 7 维关节位置作为目标，通过临时 `JOINT_POSITION` 控制器逐步回归，
最多执行 480 个控制步，并在调用环境前把有限夹爪命令裁剪到 `[-1, 1]`。它不会 reset 场景或
任务评分状态。每个 home 控制步通过独立 RPC 发布到录像和 Observer；实际 EEF 离开 AABB，或从
越界位置没有继续靠近 AABB 时立即中止。真实 EEF 越界后 Runtime 设置 `workspace_violation` 锁，只允许
`return_to_home`；home 成功且最终 EEF 回到 AABB 才解除。home 失败或最终仍越界会直接产生
`safety_stop` Episode，不再请求 Planner 执行其他工具。批量评测不等待人工批准。

物理工具结果中的兼容字段 `status` 只描述原有运行状态；新增 `execution_status` 明确表示控制
执行结果，`task_effect_status` 则只有在官方环境报告 `task_success=true` 时才是
`verified_task_success`，其余均为 `unverified`。夹爪闭合、局部调整或回到 home 的执行成功
不能直接证明抓住目标、打开抽屉或完成当前计划步骤。

困难任务探索时可给单任务评测命令增加 `--interactive-plan-approval`。启用后，只有已经通过
Runtime 计划一致性检查的 `advance_task_plan` 会在真正修改 TaskPlan 前等待终端输入：`y`
批准推进，`n` 拒绝并可输入一句反馈。拒绝结果会返回 Main Agent 重新决策；Episode trace
记录 `human_intervention` 和 `supervision_events`。该开关默认关闭，不应加入正式自主批测；
带人工确认的 Episode 可作为 Experience，但不作为自主成绩。

## 统一评测目录与官方格式产物

日常单任务测试可直接运行：

```bash
cd "$(git rev-parse --show-toplevel)"
bash benchmark/robomemarena/scripts/run_eval.sh
```

脚本默认读取 `benchmark/robomemarena/configs/qwen_recovery.env`，也可以把其他配置文件作为
第一个参数传入。该配置只保存任务、GPU、端口、模型名与预算等非密钥参数；Planner 的 API
Key 和地址仍从本地 `.env` 读取。每次运行创建一个独立时间戳目录，并自动启动和回收 policy、
环境及 Observer 服务。默认 Observer 权重为 `models/Qwen3-VL-32B-Instruct`；可通过配置中的
`OBSERVER_MODEL_PATH`、`OBSERVER_SERVED_MODEL_NAME` 和
`OBSERVER_GPU_MEMORY_UTILIZATION` 切换本地权重及显存比例。

一次跑完多个任务时用常驻服务批跑脚本：policy/环境/Observer 只启动一次，每个任务仅
`reset` 环境后执行一次 `roboagent_eval`，省去逐任务 3–5 分钟的冷启动开销：

```bash
bash benchmark/robomemarena/scripts/run_batch_persistent.sh \
  benchmark/robomemarena/configs/deepseek.env 1 26
```

服务日志写入 `logs/batch/services-<时间戳>/`，逐任务状态与产物目录追加写入
`logs/batch/current.status`；单任务失败不中断后续任务，服务异常死亡时自动重启。
必需服务首次启动超时或重启后仍不可用时，批跑以非零状态退出，不启动后续任务；
修复服务后可使用同一批次 ID 续跑。已在服务缺失期间执行的旧任务需要单独重跑。

RoboAgent-Evolve V3 的 Task 1–26 原始经验统一使用 seed 50、禁用 Skill，并保存 Observer 窗口：

```bash
bash benchmark/robomemarena/scripts/run_batch_persistent.sh \
  benchmark/robomemarena/configs/deepseek_evolution_v3_seed50.env \
  1 26 evolve-v3-collection-deepseek-seed50
```

第四个参数是可选 `BATCH_ID`。传入后，产物集中到
`logs/batches/<BATCH_ID>/`，包括 `batch_config.env`、`status.log`、`manifest.jsonl`、
`services/` 和 `runs/`。相同命令可以断点续跑：退出码为 0 且产物完整的任务会被跳过；
退出码非 0 或缺失关键产物的任务会保留旧 attempt 并创建新 attempt。未传 `BATCH_ID` 时仍保持原有 `logs/batch/` 和
`logs/task*-<时间戳>/` 布局。

需要在同一 batch 中有意追加某个已完成任务的新 attempt 时，显式设置 `FORCE_RERUN=1`，
并把起止任务设为同一个 ID；默认断点续跑行为不变：

```bash
FORCE_RERUN=1 bash benchmark/robomemarena/scripts/run_batch_persistent.sh \
  <config> 26 26 <existing-batch-id>
```

完成 batch 后按 manifest 导入 V3。每个 run 必须带 Runtime/evaluator revision、数据角色、模型、
seed 和 task；cohort ID 取自 batch manifest。普通无阶段变化的调用不会产生 Experience Unit，
但 Episode 仍完整保留。旧工作区不迁移：

```bash
.venv/bin/python -m roboagent.evolution compile \
  --workspace evolution_workspaces/robomemarena-v3 \
  --batch-manifest \
    benchmark/robomemarena/logs/batches/evolve-v3-collection-deepseek-seed50/manifest.jsonl \
  --policy-role baseline
```

结果清单位于 `evolution_workspaces/robomemarena-v3/experiences/cohorts/<batch-id>.json`。
自动编译只提取官方 stage 状态变化和明确失败；成功的 `repair_task_plan` 作为待审查恢复事件。
事件记录固定为 `schema_version=3`、`unit_kind=event`，并包含 `event_type`、`episode_id`、
`call_span`、`provenance`、`evidence`、`interpretation` 和 `learning`；事实只进入 `evidence`，
Observer、Reflection 和维护者归因只进入 `interpretation`。
人工确认的多调用恢复、副作用或 Reflection/ToolCall 矛盾用 `compile-events` 导入：

```bash
.venv/bin/python -m roboagent.evolution compile-events \
  --workspace evolution_workspaces/robomemarena-v3 \
  --manifest evolution_workspaces/robomemarena-v3/experiences/reviewed_events/<review>.jsonl
```

完成成对人工审查后，用同一 Experience Store 编译派生比较证据：

```bash
.venv/bin/python -m roboagent.evolution compile-pairs \
  --workspace evolution_workspaces/robomemarena-v3 \
  --manifest \
    evolution_workspaces/robomemarena-v3/experiences/paired_comparisons/<pairs>.jsonl
```

配对比较是派生 Experience，不增加独立证据数。Wiki maturity 按可用于 Skill 的唯一 Episode 统计；
`single-shot` 只能运行 Replay，`probable/verified` 才能进入正式 Gate。Candidate manifest 只引用
Wiki Pattern，并且相对 Parent 只允许修改一个二级章节。先运行不启动仿真的 Replay：

Replay 暴露在线决策使用的恢复工具 schema（包括 `repair_task_plan` 与
`return_to_home`），但不会执行模拟器动作。

`PLANNER_RETRIES>0` 时，Planner 的单次流式响应会在完整结束后再提交给 Agent；中途断线
只重试尚未产生工具副作用的 Assistant 响应。

```bash
.venv/bin/python -m roboagent.evolution replay \
  --workspace evolution_workspaces/robomemarena-v3 \
  --candidate evolution_workspaces/robomemarena-v3/skills/candidates/v001 \
  --planner deepseek \
  --model deepseek-v4-flash-vision-exp
```

正式验证必须使用没有生成 Candidate 的新独立 rollout。可以固定 seed 50：运行至少一个
Parent/Active 锚点，并对同一 Candidate 内容运行至少两个不同 `run_id/attempt_id`。Candidate
Gate 必须严格更优，且不允许 Parent 成功任务退化：

```bash
bash benchmark/robomemarena/scripts/run_batch_persistent.sh \
  benchmark/robomemarena/configs/deepseek_candidate_gate_v3_seed50.env \
  1 26 gate-v3-parent-seed50-r1

TASK_EXECUTION_SKILL=evolution_workspaces/robomemarena-v3/skills/candidates/v001/SKILL.md \
bash benchmark/robomemarena/scripts/run_batch_persistent.sh \
  benchmark/robomemarena/configs/deepseek_candidate_gate_v3_seed50.env \
  1 26 gate-v3-candidate-v001-seed50-r1

TASK_EXECUTION_SKILL=evolution_workspaces/robomemarena-v3/skills/candidates/v001/SKILL.md \
bash benchmark/robomemarena/scripts/run_batch_persistent.sh \
  benchmark/robomemarena/configs/deepseek_candidate_gate_v3_seed50.env \
  1 26 gate-v3-candidate-v001-seed50-r2
```

把 Parent 锚点和两个 Candidate 运行写入 Candidate Gate manifest；Replay 与 Candidate Gate
通过后即可 `promote`。`compare-rollouts --phase final-evaluation` 保留为激活后的可选审计，
不再阻塞晋升。配置中的 revision 是不可变实验标签；Runtime 或 evaluator 行为改变时必须更新。

```text
logs/task3-seed100-qwen3.8-max-YYYYMMDD-HHMM/
├── trace.json              # Agent、Working Memory、Reflection 和官方 evaluation
├── evaluation.json         # 环境 metrics 的便捷副本
├── episodes.tsv            # 官方 batch evaluator 同名格式
├── task_summary.tsv        # 官方 batch evaluator 同名格式
├── summary.json            # 单任务结果列表
├── aggregate.json          # 单任务/单 episode 汇总
├── run_config.json         # 本次解析后的非密钥运行参数
├── eval_config.env         # 启动配置快照
├── run.log
├── services/
│   ├── policy.log
│   ├── environment.log
│   └── observer.log
└── videos/
    ├── task3_success|failure_ep0.mp4
    └── task3_success|failure_ep0_wrist.mp4
```

external rollout 视频来自固定 eye-to-hand/agentview，wrist rollout 视频来自 eye-in-hand；视频
按 adapter 发布的真实物理状态流式写入，不在内存中缓存完整 episode。录制结束后统一用系统
ffmpeg 转码为 H.264（libx264 + faststart），可直接在 VS Code 和浏览器中播放；找不到 ffmpeg
或转码失败时保留 OpenCV mp4v 原文件。`episodes.tsv`、
`task_summary.tsv`、`summary.json` 和 `aggregate.json` 遵循上游 batch evaluator 的命名及百分制
TSR 表达，但不复制评分实现：CSR、TSR、阶段状态与失败原因全部直接取环境服务的官方
`metrics()`。计划完成后的终态收敛仍执行官方 post-goal/extra-pour 等待；若多个有序 stage
在最后一次物理调用中同时满足，会额外为每个尚未登记的 stage 保留一次 evaluator update。

默认 `SAVE_OBSERVER_WINDOWS=0`：Observer 仍然运行并将结果送入 Agent 工作记忆，但不保存
每个窗口的双视角 PNG。需要离线检查窗口时，在配置中设为 `1`；图片会写入同一运行目录下的
`observer/`。也可绕过脚本直接给 `roboagent_eval` 传 `--run-dir PATH`，由入口生成上述 trace、
视频和汇总文件。

## 常见问题

- **找不到 LIBERO fork**：确认 `ROBOMEMARENA_ROOT` 指向
  `benchmark/.external/RoboMemArena`，不要指向普通 LIBERO。
- **EGL/OpenGL 初始化失败**：启动前同时设置 `MUJOCO_GL=egl` 和
  `PYOPENGL_PLATFORM=egl`，并确认机器 NVIDIA/EGL 驱动可用。
- **Numba 报 cache locator 错误**：设置可写的 `NUMBA_CACHE_DIR`；Matplotlib 同理使用
  `MPLCONFIGDIR`。
- **端口占用**：用其他 `--port` 启动，并让 smoke 使用相同端口。
- **依赖导入异常**：确认当前解释器为 `roboagent-robomemarena` 的 Python 3.8.13，并按本
  目录固定的 `requirements.txt` 安装，不要混用 `roboagent-libero`。

## 本期明确不包含

没有论文频率近似或额外策略调度；`agent/roboagent` 和 `pi05_robomemarena` 权重本身均保持
不动。当前提供单任务 episode、双视角 rollout 视频、官方 evaluation 和官方格式的单任务
汇总文件；尚未实现 26 个任务的批量调度与跨 episode 聚合。

RoboMemArena 启动脚本对 32B Observer 的健康检查最多等待 900 秒，其他服务为 180 秒。
Observer 本地健康检查绕过代理，每次 HTTP 探测最多等待 5 秒。
