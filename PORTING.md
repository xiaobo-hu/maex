# RoboAgent 迁移与适配说明

本文记录当前实现从参考仓库吸收了什么、哪些代码由本项目维护，以及将 RoboAgent 接到
其他 VLA、benchmark 或真实机器人时必须保持的边界。日常启动命令见根
[README.md](README.md)。

## 代码来源与所有权

| 路径 | 角色 | 当前运行是否依赖 |
| --- | --- | --- |
| `agent/ai` | Python 模型 Provider、多模态消息和工具协议 | 是 |
| `agent/agent-core` | Python 唯一 Agent Loop | 是 |
| `agent/roboagent` | 本项目机器人 runtime、ports 和 tools | 是 |
| `benchmark/libero` | 本项目维护的 LIBERO benchmark 集成 | 是 |
| `benchmark/robomemarena` | RoboMemArena 环境、官方评分和 RobotPort 适配 | 环境接口可用 |
| `benchmark/.external/RoboMemArena` | 固定 commit 的官方 LIBERO fork、BDDL 与评分器 | 是，环境进程 |
| `openpi` | frozen VLA policy server 和权重 | 是，独立进程 |
| `pi`、`pi-py` | Agent/AI 组织与行为参考 | 否 |
| `HarnessVLA`、`HiMe` | Harness、reposition 等方案参考 | 否 |

`benchmark/libero/_vendor/openpi_client` 从 OpenPI client 协议迁入，使 benchmark 客户端
不需要 import `openpi` Python 包。该目录保留 Apache-2.0 `LICENSE` 和 `NOTICE.md`。
这只消除了客户端源码依赖；运行 `pi05_libero` 仍然需要 `openpi` policy server。

## 为什么拆成三个环境

```text
RoboAgent / Provider       Python >= 3.11
OpenPI policy server       openpi 自己的 uv 环境
LIBERO / robosuite         conda Python 3.8.13
```

LIBERO 上游依赖较旧，不能与 RoboAgent 的 Python 3.11+ 依赖直接混装。当前通过两个本地
WebSocket 边界隔离：

- `8000`：OpenPI client → `pi05_libero` policy server；
- `8001`：`RemoteLiberoEnv` → Python 3.8 LIBERO/MuJoCo server。

LIBERO server 持有唯一环境实例，并串行处理 `reset / observe / step / close`。这不是分布式
执行设计，而是依赖隔离设计。

RoboMemArena 沿用相同的三进程职责边界，但使用独立的
`roboagent-robomemarena`（Python 3.8.13）环境，不能复用 `roboagent-libero`：

```text
RoboAgent / Provider       Python >= 3.11
OpenPI policy server       OpenPI 自己的环境，pi05_robomemarena
RoboMemArena / MuJoCo      conda roboagent-robomemarena, Python 3.8.13
```

`RemoteRoboMemArenaEnv` 的环境 RPC 为 `reset / observe / step / metrics / close`。其中
`metrics` 是 evaluation-only 通道：CSR、TSR、阶段名和中间评分不得进入 Agent 可见的
`Observation` 或 `ExecutionFeedback`。物理 action 仍严格串行，`observe` 不推进仿真。
可选 Observer 作为第四个只读进程消费每步 observation；当前只写 episode trace、窗口图片
和 assessment JSONL，不参与规划或向 Agent 回传判断。

## 核心契约

### RobotPort

实现位于 `agent/roboagent/ports.py`：

```python
async def observe(...) -> Observation: ...
async def execute_action_chunk(...) -> ExecutionFeedback: ...
```

异步签名用于统一 Agent 生命周期和取消信号，不代表物理动作可以并发。RoboAgent runtime
使用一把物理锁，LIBERO adapter 再逐条调用同步 `env.step(action)`；不要用 `gather`、后台
task 或并行动作队列操作同一机器人。

### VlaPolicyPort

```python
async def infer(
    instruction: str,
    observation: Observation,
    ...,
) -> ActionChunk: ...
```

VLA 只负责根据语言子任务和当前 observation 生成动作块。动作是否执行、何时停止以及任务
是否成功由 RobotPort/runtime 决定，不由 policy server 直接结束 episode。

### Observation 与动作

LIBERO 适配当前保持 OpenPI 官方 `pi05_libero` 输入语义：

```text
observation/image
observation/wrist_image
observation/state
prompt
```

`benchmark/libero/observation.py` 在原始 LIBERO observation、Provider 可见 PNG 图像和
OpenPI numpy 输入之间转换。`benchmark/libero/preprocessing.py` 负责官方图像/state 预处理；
不要把 benchmark 特有字段放进 `agent/roboagent`。

RoboMemArena 使用不同的官方图像语义：只做垂直 `flipud`，再用 OpenCV `INTER_AREA`
强制缩放到 `256×256`，不做水平翻转。其阶段评分在环境进程中直接调用固定 commit
`cc156e519990ae43cf3b64281a548724f428fbbd` 的上游 checker；本项目不复制评分代码。

### Observer 只读视觉流

Observer 与机器人执行使用能力分离的 observation stream：Robot adapter 只持有
`ObservationSink`，ObserverRunner 只持有 `ObservationSource`。adapter 应在每次真实产生
新 observation 后快速 `publish`；Observer 不得持有 `RobotPort`、调用 `observe` 或推进
环境。VLA action chunk 只影响低层控制，不能作为 Observer 的触发条件。

Runner 以当前 Agent 子任务为条件，单并发消费尚未处理的 sequence 区间。一次推理期间到达
的 observation 留给下一窗口；窗口可以保留上一区间末帧作为视觉锚点，但不能把锚点再次
选为关键帧。关键帧只使用采样器根据实际 `gripper_qpos` 检测到的夹爪开合变化帧；新
benchmark 若没有该状态，采样器会退化为末帧加均匀抽帧且关键帧为空，不应从 action 命令
伪造实际夹爪状态。

## 工具与安全边界

模型当前看见五个工具：`create_task_plan`、`execute_vla`、`advance_task_plan`、`observe`、
`finish`。所有工具按 sequential 模式执行。TaskPlan 只由模型创建和推进；runtime 只保存
通过校验的显式状态转换。runtime 还执行以下最小安全规则：

- 首次机器人动作前必须创建计划，且恰好一个步骤为 `running`；
- `execute_vla` 的 `step_id` 必须指向唯一的 `running` 步骤，instruction 由 Runtime 从计划读取；
- 可选 `EpisodeFinalizerPort` 只负责环境自身的终态验证，不包含任务规划或评分逻辑；
- 单次 `execute_vla` 默认最多执行 24 个 action chunk，达到预算不改变计划状态；
- 只有 `advance_task_plan` 能完成当前步骤并激活 Agent 指定的下一 pending 步骤；
- 初始消息已经携带当前 observation，重复的初始 `observe` 会被拦截；
- 同一 assistant turn 只允许第一个工具调用；
- 没有动作发生时，连续 `observe` 会被拦截；
- 最新 observation 为 `task_success=true` 后，禁止继续驱动或观察机器人，可直接 `finish`；
- episode 必须由 `finish` 给出结构化终态。

这些拦截也会写进 `tool_trace`，不会静默丢弃。

## 迁移到新 benchmark 或真实机器人

最小迁移顺序：

1. 实现 `RobotPort`，保证 `observe` 不产生动作，动作块严格串行执行；
2. 把环境原始 observation 转为通用 `Observation`，只在 adapter 内保留环境字段；
3. 实现或复用 `VlaInputBuilder`，匹配目标 VLA 的图像、state、动作维度和归一化统计；
4. 用 faux planner 验证 Agent Loop、工具次序和成功信号；
5. 用官方 policy client/baseline 验证 VLA 与环境；
6. 最后接真实视觉 Planner，保存 `--trace-file` 做端到端验证。

若目标环境依赖版本与主项目冲突，沿用 LIBERO 的独立进程模式；若依赖兼容，可以直接实现
进程内 RobotPort，但仍必须保持物理动作串行。

## 更新上游代码

更新 vendored OpenPI client 时：

1. 对照 `openpi/packages/openpi-client` 的 WebSocket、msgpack 和图像处理协议；
2. 只迁移 benchmark 实际需要的文件，不复制整个 OpenPI 包；
3. 保留许可证和来源说明；
4. 运行 `benchmark/libero/tests`；
5. 再运行官方 baseline 和一次真实 RoboAgent trace。

更新 LIBERO 或 robosuite 时，不要直接升级主项目依赖。先在 `roboagent-libero` 环境验证
`reset/step/success`、相机键名和 7D 动作语义，再调整 RPC adapter。

更新 RoboMemArena 时也必须先在独立环境验证 Task 1 的
`reset/observe/step/metrics/close`，并重新核对非计数任务 200 步延迟、计数任务 30 步监控
与第三次倾倒失败语义；不要把上游评分器复制到本仓库。

## 当前不迁移的内容

- Memory 的存储、检索和 prompt 注入；
- HarnessVLA reposition 和其他物理辅助原语；
- 并行机器人动作执行；
- LIBERO demonstration dataset 的训练流程；
- 自训练或微调 OpenPI policy。

这些内容只有在最小 `Agent + frozen VLA + environment` 闭环稳定后再独立设计，不提前写入
当前 runtime API。
