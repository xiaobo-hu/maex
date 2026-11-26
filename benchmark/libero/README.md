# LIBERO Benchmark

## 仓库介绍

本目录把 LIBERO（MuJoCo + robosuite）作为 RoboAgent 的仿真 benchmark，并提供隔离的
LIBERO 环境服务供 `execute_vla` 使用。仿真不需要 demonstration dataset。

## 仓库构成

```text
benchmark/libero/
├── _vendor/openpi_client/  # Apache-2.0 OpenPI WebSocket/图像协议
├── env_server.py           # Python 3.8 持有 MuJoCo，串行 reset/step
├── remote_env.py           # RoboAgent Python 的环境客户端
├── preprocessing.py       # pi05_libero 官方 observation/state 预处理
├── adapter.py              # LIBERO → RoboAgent RobotPort
├── observer_recording.py   # 可选的本地 Observer 窗口记录
└── scripts/
    ├── serve_env.py        # 启动 LIBERO/MuJoCo 环境服务
    └── roboagent_eval.py   # 运行一次 Agent + VLA + LIBERO 闭环
```

## 环境安装

LIBERO 使用已经创建的 Python 3.8.13 conda 环境：

```bash
cd /path/to/robo-mem-agent
conda activate roboagent-libero
python -m pip install -r benchmark/libero/requirements.txt

export LIBERO_CONFIG_PATH="$HOME/.libero"
export MUJOCO_GL=egl
```

RoboAgent 客户端使用项目 Python ≥3.11 环境：

```bash
cd /path/to/robo-mem-agent
uv sync --extra libero-client --extra dev
```

## 完整启动

以下命令分别在四个终端运行；不需要 Observer 时可省略终端 3 和对应参数。

终端 1，启动 OpenPI `pi05_libero` policy：

```bash
cd /path/to/robo-mem-agent/openpi
CUDA_VISIBLE_DEVICES=0 uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=pi05_libero \
  --policy.dir="$PWD/checkpoints/pi05_libero"
```

终端 2，启动 Python 3.8 LIBERO/MuJoCo 仿真服务：

```bash
conda activate roboagent-libero
cd /path/to/robo-mem-agent
export LIBERO_CONFIG_PATH="$HOME/.libero"
export MUJOCO_GL=egl
python -m benchmark.libero.scripts.serve_env --host 127.0.0.1 --port 8001
```

终端 3，启动本地 Qwen Observer：

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

终端 4，运行一次真实 Agent 调用：

```bash
cd /path/to/robo-mem-agent
test -f .env || cp .env.example .env
# 编辑 .env：ZHIPU_API_KEY=你的真实密钥

uv run --env-file .env --extra libero-client \
  python -m benchmark.libero.scripts.roboagent_eval \
  --policy-port 8000 --env-port 8001 \
  --suite libero_spatial --task 0 --planner glm \
  --max-vla-chunks-per-call 24 \
  --planner-retries 12 --planner-retry-delay-ms 15000 \
  --observer-base-url http://127.0.0.1:8002/v1 \
  --observer-model robo-observer \
  --observer-timesteps 5 \
  --observer-min-new-observations 10 \
  --observer-record-dir benchmark/libero/logs/observer-task0 \
  --trace-file benchmark/libero/logs/glm-task0.json
```

`glm-4.6v-flash` 支持图像和 Function Call；纯文本的 `glm-4.7-flash`
不用于当前视觉 Agent 闭环。重试参数用于吸收免费模型的瞬时拥塞；真实 `.env`
已被 Git 忽略。命令结束时会在终端格式化展示工具名称、参数、执行状态和结果；
`--trace-file` 会保存同一份完整 JSON。图像只记录数量和 observation 摘要，不保存
base64 payload。

Planner 必须先通过 `create_task_plan` 创建恰好一个运行中步骤，之后
`execute_vla` 只提交当前 running 步骤的 `step_id`，instruction 由 Runtime 从计划读取。单次工具调用最多执行
`--max-vla-chunks-per-call` 个 action chunk；默认 24，达到预算只返回控制权，不会自动把
步骤标成完成。Planner 检查最新画面后可重试相同步骤，或调用 `advance_task_plan` 显式选择
下一步骤。最终计划保存在 trace 的 `details.task_plan`。

环境 adapter 仅在串行 `env.step()` 返回后向只读视觉流发布 observation。Observer 不调用
环境或机器人接口，也不按 action chunk 调度；它按未消费 observation 区间单并发运行，
默认积累 10 个新 observation 后，从中选择最多 5 个 timestep（包含一个上一区间视觉
锚点）。输入按 oldest→newest 排列，并明确区分外部 eye-to-hand 和腕部 eye-in-hand
视角；后续窗口还会携带上一条待验证的视觉 observation。配置夹爪阈值后，sampler 根据
post-step observation 中的 `action[6]` 命令区间和累计 `gripper_qpos` 变化确定实际开合帧。
每次 `execute_vla` 结束后，去重文本、关键帧 robot state 和带标签的双视角图像会作为统一
`observation_context` 返回 Agent；episode trace 仅保存关键帧摘要和图像数量，不保存 base64，
也不改变动作或 Agent 控制流。省略
`--observer-record-dir` 时 trace 仍包含结构化判断，但不会把图像写到磁盘。

唯一保留的成功运行日志是 `benchmark/libero/logs/glm-task0.json`。该目录已被 Git 忽略。
