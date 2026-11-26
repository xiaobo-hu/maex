# HarnessVLA 原版基线迁移

## 官方权重推理环境（独立验证）

官方 LIBERO 配套权重是 `RLinf/RLinf-Pi05-LIBERO-130-fullshot-SFT`。
独立环境位于 `models/harnessvla/official-pi05-venv`，使用 Python 3.11、
Torch 2.7.1+cu128（适配 Blackwell）、RPent-RLinf 0.3.0、Transformers 4.57.6，
OpenPI 固定官方 `rpent` 提交 `a560f4dd8205b8423ecd4c8a0fabb5f54140b8a0`。
本机版本约束、完整依赖锁和验证脚本保存在 `models/harnessvla/official-pi05-validation/`；
该目录为本地安装产物，不进入版本控制。

从项目根目录执行：

```bash
bash models/harnessvla/official-pi05-validation/install.sh
CUDA_VISIBLE_DEVICES=2 PYTHONDONTWRITEBYTECODE=1 \
  models/harnessvla/official-pi05-venv/bin/python \
  models/harnessvla/official-pi05-validation/check_inference.py
```

验证脚本通过官方模型服务及 `Pi05VLAClient` 推理合成双视角观测，检查权重匹配、
5×7 动作形状及有限值，结果保存到 `result.json`，结束后关闭服务。
缺失参数仅在证实与已加载参数为同一对象且值相等时视作共享权重别名。
这验证加载与推理链路，不证明 RoboMemArena 操作成功。
`harness_official` 与首轮脚本现已连接该官方 HTTP 服务；旧 `harness_eval` 入口仍独立保留。
旧权重实验结果不能当作官方权重复现结果，新实验需使用独立记忆目录。

## 官方探索与本地记忆评测入口

`harness_official.py` 使用官方 `RLinf/RPent`，固定版本
`24541d6ff5f6d4398cf0dbb63ed827e7198237cc`，与原有 `HarnessVLA/` 旧分支分开保存。
它直接执行官方 CLI，通过 `RobotSpec` 的运行时组合钩子连接 RoboMemArena；
Planner、Toolkit、reset 预算、跨 session 交接、recipe 导出和
`MemoryManager.merge_memory` 均使用官方实现。没有额外的人工或模型记忆筛选流程。
入口保留官方 `libero` 工具身份，实际 suite 为 `robomemarena`，任务语言来自实际环境；
保留的指南含 LIBERO 经验，因此属于 RoboMemArena 迁移基线，不能声称论文原环境复现。

按照 RoboMemArena 的[官方接入说明](../../.external/RoboMemArena/evaluation_benchmark/docs/evaluate_your_model.md)，
VLM 选择子任务后可以调用微调 VLA。环境的 `task_language` 同时提供与 RoboAgent、HiMe
相同的 `task_primitive_reference`：当前任务描述、有序 checkpoint primitive 标签及数据集备注。
这些内容来自项目固定的官方任务目录副本；它是共同的静态先验，不计作探索学得的记忆。
`harness_prompts.py` 通过官方 PromptBundle 扩展点适配环境身份，移除 LIBERO 专属的
PROVEN LEVERS 任务经验段，保留官方感知、动作与探索工作流。探索提示明确实际 episode
预算，将旧的人工合并说明纠正为自动合并，补充 suite/global YAML 示例及结束前用官方
解析器进行只读检查的命令。模型仍可能不遵循提示，应检查实际草稿与合并结果。
VLA 子指令仍由官方 Agent 自己选择，不增加标签强制映射或脚本策略。
`robomemarena-v3-official-libero-policy` 明确官方权重身份，移除 v2 自加的推进提示。
静态任务参考仅描述任务，不宣称该 LIBERO 权重训练过 RoboMemArena 动作。
已启动会话不热更新，每个任务的提示版本以 `run_config.json` 和实际归档提示为准；
不同提示版本的结果须分别报告。
不向 Agent 挂载其他实验结果或 RoboAgent 演化记忆。

```bash
git clone https://github.com/RLinf/RPent.git benchmark/.external/RPent-official
git -C benchmark/.external/RPent-official checkout 24541d6ff5f6d4398cf0dbb63ed827e7198237cc
```

客户端使用下文轻量 Python 环境，policy 使用上面的独立官方推理环境。
环境服务沿用 `harness_server`；观测双轴翻转及轴角转换匹配 RLinf LIBERO，
policy 输出保持官方 5 个 7 维动作。手动运行时先启动官方 policy HTTP 服务：

```bash
CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
  models/harnessvla/official-pi05-venv/bin/python -m rpent.robots.components.pi05_vla_server \
  --embodiment libero --model-path models/harnessvla/RLinf-Pi05-LIBERO- \
  --transport http --host 127.0.0.1 --port 8020
```

新版官方 SAM3 服务启动方式为：

```bash
source models/harnessvla/runtime.env
CUDA_VISIBLE_DEVICES=2 PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH="$PWD/benchmark/.external/RPent-official" \
"$SAM3_PYTHON" -m rpent.robots.components.sam3_server \
  --transport http --host 127.0.0.1 --port 8012
```

配置 Claude Code 的 `ANTHROPIC_AUTH_TOKEN`、`ANTHROPIC_BASE_URL` 和
`HARNESS_PLANNER_MODEL` 后，从项目根目录执行。推荐通过 `bwrap` 文件隔离启动，
仅挂载运行依赖、本次日志目录和指定记忆库，避免 Claude Code 的 Bash/Read
读到其他方法或旧实验的结果；评测时记忆目录以只读方式挂载，探索时允许写入。
该隔离不修改官方提示或工具。
`agent/` 仅挂载 `roboagent/prompts/robomemarena.py` 和共同任务参考 JSON，
不挂载 RoboAgent 系统/Observer/Reflection 提示、skills 或运行时策略。
隔离内使用空包入口避免导入这些策略；主项目的公开 Python API 保持不变。
挂载变化只对新会话生效，不追溯改变已运行实验的可见范围。
两个位置参数分别为新的实验根目录和持久记忆目录，运行产物写入前者的 `run/`：

```bash
bash benchmark/robomemarena/scripts/run_harness_official.sh \
  benchmark/robomemarena/logs/harness-official-task1-seed0 \
  models/harnessvla/official-libero-robomemarena-memory \
  --task 1 --seed 0 --max-steps 2500 \
  --env-port 8011 --policy-port 8020 \
  --sam3-endpoint http://127.0.0.1:8012 \
  --planner claude_code --model "$HARNESS_PLANNER_MODEL" \
  --explore --explore-sessions 1 --explore-attempts-per-session 5 \
  --max-turns 100 --planner-timeout-s 1200
```

首次探索可以使用空记忆目录。一次 session 最多 5 个物理 episode，包含首次启动，
每次 reset 恢复相同 seed；成功会提前结束。官方提示要求 Agent 在 inbox/wip 写工作笔记，
解决任务后整理 suite/global 草稿；官方 runner 在无运行错误时校验、合并顶层草稿并刷新
索引，只有环境成功才发布 task_only 的 audit/recipe 对。运行错误、超时或无有效草稿
可能不会生成可用记忆；以真实产物为准，不手工补造。5 次是上限，不是保证完成的次数。

评测沿用同一命令与记忆目录，去掉 `--explore` 及两个探索预算参数，换成未用于探索的
seed（如 `--seed 50`）和新的实验根目录。入口固定 `--memory-profile local`，评测只有
一次 episode，不开放 reset、不执行自动合并。不要把探索成功率混入评测成功率。
正式比较还需固定训练/评测 seed、物理步和模型调用预算，并披露 RGB-D 输入差异。

`run_config.json` 记录版本、模型和动作上限，`system_prompt.txt` / `user_prompt.txt`
保存实际适配提示，`*_prompt.official.txt` 保存同参数官方原文；`artifacts/` 保留原版 session 轨迹、Agent audit 和 recipe，
`episodes/episode_NNN/` 在每次 reset 前及最终退出时记录官方评分（不提供给 Planner）。
`completion.json` 记录实际 episode 数、退出码、源码是否未变及记忆是否变化；
`memory_before.json` / `memory_after.json` 用于验证探索写入和评测只读。
评分视频位于 `artifacts/sessions/session_NNN/`，按官方 session 组织。

提示与合并格式的离线回归（需要上述固定官方 checkout 和客户端环境）：

```bash
PYTHONPATH="$PWD/agent:$PWD" PYTHONDONTWRITEBYTECODE=1 \
  benchmark/robomemarena/logs/harness-venv/bin/python -m pytest \
  -o addopts='' -p no:cacheprovider \
  benchmark/robomemarena/tests/test_harness_prompt_contracts.py \
  benchmark/robomemarena/tests/test_harness_baseline.py -q
```

该检查覆盖探索/评测模式、官方操作规则保留、无 YAML 草稿拒收、合法 suite/global
草稿合并与索引生成，不代表模型必然遵循格式，也不代替机器人任务验证。

### GPU2 首轮逐任务探索

脚本自动启动官方 policy HTTP、SAM3 和 RoboMemArena 服务。权重路径可通过
`PI05_CHECKPOINT_PATH` 指定，默认使用本机已下载的 `models/harnessvla/RLinf-Pi05-LIBERO-`。
新记忆目录会写入 `policy_identity.json`；已有无身份标记或身份不同的记忆目录会被拒绝。
单次通路测试可限制为任务 1（独立日志和记忆，每次用新路径）：

```bash
HARNESS_SEED=50 HARNESS_MAX_STEPS=100 HARNESS_MAX_TURNS=24 HARNESS_PLANNER_TIMEOUT=300 \
  bash benchmark/robomemarena/scripts/run_harness_exploration_round.sh \
  benchmark/robomemarena/logs/harness-official-smoke-new \
  models/harnessvla/official-smoke-memory-new 1 1
```

上述短预算仅用于通路检查，不用于任务成功率比较。
默认预算保持原脚本设置；正式探索 seed、次数及预算需单独固定。

2026-09-10 的 GPU2 通路验证已正常退出：Claude Code 经官方 `pi0_pick` 调用该权重，
在真实环境中累计执行 100 步，完成评分和失败记录归档，官方源码未变。
该短测试任务未成功；官方合并器正常返回，但没有 suite/global 草稿可发布，合并数为 0。
因此它验证模型到环境及结束归档的通路，不证明任务成功或实际生成了通用记忆。

先让任务 1–26 各探索一次，共享同一个记忆库，再根据首轮记录决定后续轮次。
这只是用官方 `--explore-sessions 1 --explore-attempts-per-session 1` 控制实验预算，
不增加外部策略选择或记忆筛选；成功后的记忆整理与合并仍由官方 Agent/runner 负责。
失败任务的 wip 会保留在原版 inbox 中。尚未验证的后续轮次不自动启动。

```bash
nohup bash benchmark/robomemarena/scripts/run_harness_exploration_round.sh \
  benchmark/robomemarena/logs/harness-explore-round1 \
  models/harnessvla/official-libero-rounds-memory \
  > benchmark/robomemarena/logs/harness-explore-round1.launch.log 2>&1 < /dev/null &
```

脚本自行在 GPU2 启动三项服务，使用独立端口 8020/8021/8022；不要与另一个相同端口的
round 同时运行。复用已准备的客户端、SAM3 环境和 checkpoint，从独立 `.env_claude`
读取 `LLM_API_KEY`、`LLM_BASE_URL`、`LLM_MODEL_NAME`（每行 `LLM_名称=值`，不加引号）；
不读取其他基线的 `.env`，也不使用 `HARNESS_PLANNER_MODEL` 覆盖该文件的模型选择。
密钥不写入实验配置，`.env_claude` 已加入 Git 忽略。
脚本把 `LLM_API_KEY` 传给 `ANTHROPIC_AUTH_TOKEN`，将 DeepSeek 根地址或 `/v1`
地址转换成 `/anthropic` 兼容入口；已带 `/anthropic` 的地址保持不变。
`LLM_PROVIDER` / `LLM_MAX_TOKENS` / `LLM_DISABLE_THINKING` 不是官方 Claude Code
Planner 的配置接口；Planner 仍沿用官方默认的关闭 thinking 和下述 turns/时间预算。
默认 seed 为 0，每任务 2500 物理步、100 turns、1200 秒 Planner 时间；可分别通过
`HARNESS_SEED`、`HARNESS_MAX_STEPS`、`HARNESS_MAX_TURNS`、`HARNESS_PLANNER_TIMEOUT` 设置。
默认外层 1320 秒
仅为进程退出上限。所有退出码和实际 episode 均写入 `manifest.jsonl`，不重跑或过滤零分任务。
非零退出码（如 Planner 超时或 API 错误）会终止批次并关闭本批服务，避免连续产生无效任务；
任务正常退出但未成功时仍继续。排查后使用新批次目录、同一记忆目录和 `FIRST_TASK LAST_TASK`
从中断任务续跑，原始运行错误记录默认保留，只有明确要求清理时才删除。
`status.log` 显示任务开始/结束，`taskNN.log` 为原版 Agent 日志，`batch_config.json`
记录实验配置。脚本结束时关闭自己启动的服务。最后可追加 `FIRST_TASK LAST_TASK`
做子集检查，例如 `1 1`；每次使用新的批次目录。

## 旧分支的单 episode 入口

以下说明针对保留的 `harness_eval.py` 和本地旧 `HarnessVLA/`，不描述上面的官方探索入口。

本入口直接导入外部 `HarnessVLA/`（RPent）中的 `LiberoToolkit`、`LiberoPrimitives`、
Planner、PromptBundle 和记忆读取工具。原仓库代码、提示词、动作参数、抓取停止启发式与
LIBERO 经验内容均不修改，不引入 RoboAgent 的 Observer、Reflection、恢复策略或 Evolve。

新增代码只承担以下边界：

- `harness_server.py`：在原 RoboMemArena 服务上增加独立 RGB-D/相机标定 RPC；
- `harness_bridge.py`：单环境接口与 OpenPI 请求格式转换；
- `harness_eval.py`：组合原版工具与 Planner，保存运行记录及官方评分。

普通 `serve_env`、`roboagent_eval` 入口不受影响。基线服务使用单独端口，不要与正在进行的
实验共享同一个环境实例。

## 实验定义

这是 **HarnessVLA 原代码在 RoboMemArena 上的迁移评测**，不是原论文成绩复现。
环境改为本项目固定版本及补丁的 RoboMemArena，低层权重改为与本项目相同的冻结
`pi05_robomemarena`；图像使用该 checkpoint 对应的 flipud/INTER_AREA 预处理，
每个 VLA chunk 最多执行 20 个动作；若 checkpoint 返回更短序列，则原样执行，
不重复或补齐动作。当前真实服务返回 10 个动作。所有解析式动作也逐步计入同一 2500 步预算。

保留原版双视角 RGB-D、1024×1024 感知图像、相机内外参和物体名称清单。
名称从观测键提取，物体真实位置/姿态不进入客户端；隐藏阶段、倾倒计数、CSR/TSR 只在
规划结束后读取。该输入条件比普通 RoboAgent 的 RGB 协议丰富，比较时必须披露。

原工具中的 `libero_terminated` 映射为 RoboMemArena 官方最终成功；预算耗尽或额外倾倒
失败映射为 `episode_truncated`，不能把环境 `done` 直接当成功。Planner 的
`finish(status="success")` 只触发现有 benchmark 终态收敛，最终成绩仍取官方 evaluator。
物理控制器的成功启发式保持原样，不由适配层替换成隐藏阶段判定。

原提示词含 LIBERO 场景经验和 seed-0 参考路径，本入口保留这些内容。默认从原版
`RLinf/RPent-memory` 获取资源，或复制 `--resources-dir` 指定的原版
`resources/libero`。资源在每次运行中独立复制，写入不会污染原文件或下一次评测。
正式评测要求 `memory/MEMORY.md` 存在；下载失败不会悄悄变成无记忆实验。
本入口**不构建 RoboMemArena 任务专属记忆**；不能把这组结果标作经过 RoboMemArena
few-shot 经验构建的 HarnessVLA。若后续开展该实验，需要单独定义收集预算和留出 seeds。

## 客户端依赖

从项目根目录执行。使用独立 Python 3.11 环境，直接读取原项目的依赖声明；不安装
`[full]`，因为环境和 VLA 已有独立服务。额外的 scipy 用于原动作工具，imageio-ffmpeg
用于原版录像，其余包用于本项目 RPC/图像预处理。不会修改项目主环境或锁文件。

```bash
uv venv --python 3.11 benchmark/robomemarena/logs/harness-venv
uv pip install --python benchmark/robomemarena/logs/harness-venv/bin/python \
  -e '.[robomemarena-client]' -r HarnessVLA/pyproject.toml \
  'imageio[ffmpeg]' scipy msgpack 'websockets>=13,<14' \
  'opencv-python-headless<5' 'pillow<12'
```

安装本项目是为了满足 `benchmark.robomemarena` 的包导入依赖；基线决策仍使用原 RPent。

## 准备 SAM3 权重与固定记忆资源

DeepSeek 使用远程 API，不需要下载其权重；已有的 `pi05_robomemarena` 也无需重复下载。
SAM3 使用原项目指定的 `sam3==0.1.4`，可将推理依赖安装到独立客户端环境：

```bash
uv pip install --python benchmark/robomemarena/logs/harness-venv/bin/python \
  'sam3==0.1.4' torch torchvision
```

`facebook/sam3` 是需要账号授权的官方模型仓库。先在
[官方模型页](https://huggingface.co/facebook/sam3) 完成访问申请，并在本机登录；不要将
token 写进配置或提交到仓库：

```bash
benchmark/robomemarena/logs/harness-venv/bin/hf auth login
benchmark/robomemarena/logs/harness-venv/bin/python \
  -m benchmark.robomemarena.baselines.prepare_harness_assets
```

准备入口固定 SAM3 revision `3c879f39826c281e95690f02c7821c4de09afae7` 和原版
`RLinf/RPent-memory` revision `8d34642e420e35ea8a883ddabe52471433920177`，下载到已忽略的
`models/harnessvla/`。权重未获授权时仍会准备其他资源，并在
`preparation_status.json` 中记录未完成项，退出码为 1；登录后可重新执行。

官方记忆数据集目前使用 `libero/MEMORY.md`，原代码提示词读取
`libero/memory/MEMORY.md`。准备入口在 `resources-compatible/libero/memory/` 下建立
字节不变的独立副本，不修改原提示词、原下载文件或记忆内容；原始文件校验值保存在
`memory_checksums.json`。后续运行使用：

```bash
export SAM3_CHECKPOINT_PATH="$PWD/models/harnessvla/sam3/sam3.pt"
# 将以下参数加入 harness_eval 命令：
# --resources-dir models/harnessvla/resources-compatible/libero
```

准备入口还下载并校验官方 BPE 词表，补足 `sam3==0.1.4` wheel 缺少的资源。
将词表放到该版本 `model_builder.py` 默认查找的位置，不修改模型或 HarnessVLA 代码：

```bash
SAM3_PYTHON=${SAM3_PYTHON:-benchmark/robomemarena/logs/harness-venv/bin/python}
"$SAM3_PYTHON" - <<'PY'
import importlib.util
import shutil
from pathlib import Path
package = Path(importlib.util.find_spec("sam3").origin).parent
target = package.parent / "assets/bpe_simple_vocab_16e6.txt.gz"
target.parent.mkdir(parents=True, exist_ok=True)
shutil.copyfile("models/harnessvla/sam3/bpe_simple_vocab_16e6.txt.gz", target)
PY
```

本机另外准备了 `logs/harness-sam3-venv`：独立复制已有 GPU 依赖后安装 SAM3，
使用 Python 3.13 与 PyTorch 2.13/CUDA 13；仅用于 SAM3 服务，Planner 仍使用
`logs/harness-venv` 的 Python 3.11。本机路径保存在忽略的
`models/harnessvla/runtime.env`，可 `source models/harnessvla/runtime.env` 加载。
该独立环境已通过原版 SAM3 模块导入、词表编码和 Blackwell CUDA 矩阵运算检查，
结果保存在 `models/harnessvla/runtime_verification.json`。本机随后使用用户下载的
ModelScope `facebook/sam3` 权重，通过了原版模型加载、文本提示和点提示推理检查。

## 启动独立环境服务

使用已经配置好的 Python 3.8 RoboMemArena 环境和上游资产：

```bash
conda activate roboagent-robomemarena
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export NUMBA_CACHE_DIR=/tmp/robomemarena-harness-numba
export MPLCONFIGDIR=/tmp/robomemarena-harness-mpl
python -m benchmark.robomemarena.baselines.harness_server --port 8011
```

相机按需渲染，不推进仿真，也不改变原服务的观测协议、阶段检查或终止窗口。

## 无模型接线验证

在另一个终端、项目根目录运行；`--run-dir` 必须是新目录，且位于原 HarnessVLA 目录之外：

```bash
benchmark/robomemarena/logs/harness-venv/bin/python \
  -m benchmark.robomemarena.baselines.harness_eval \
  --harness-root HarnessVLA \
  --run-dir benchmark/robomemarena/logs/harness-smoke-task1-seed50 \
  --task 1 --seed 50 --env-port 8011 --smoke
```

该命令调用原版 `view_driver_state`、双视角 `back_project`、一次 `set_gripper`。
不请求 Planner、SAM3 或 VLA，不下载记忆，也不输出正式评分汇总。

## 单 episode 正式入口

先按 [RoboMemArena README](../README.md) 启动 `pi05_robomemarena` policy server。
另按原 HarnessVLA 配置启动 SAM3 服务（需要 SAM3 权重及其独立推理依赖）：

```bash
PYTHONPATH="$PWD/HarnessVLA${PYTHONPATH:+:$PYTHONPATH}" \
"${SAM3_PYTHON:-benchmark/robomemarena/logs/harness-venv/bin/python}" \
  -m robots.libero.sam3_server \
  --transport http --host 127.0.0.1 --port 8012
```

Planner 使用原框架的 provider 环境变量，例如 `OPENAI_API_KEY`、`OPENAI_BASE_URL`；
密钥不写进运行配置。下面的模型名由实验配置指定：

```bash
benchmark/robomemarena/logs/harness-venv/bin/python \
  -m benchmark.robomemarena.baselines.harness_eval \
  --harness-root HarnessVLA \
  --run-dir benchmark/robomemarena/logs/harness-task1-seed50-r1 \
  --task 1 --seed 50 --max-steps 2500 \
  --env-port 8011 --policy-port 8020 \
  --sam3-endpoint http://127.0.0.1:8012 \
  --planner api --model "openai-chat:${HARNESS_PLANNER_MODEL}"
```

有原版资源快照时增加 `--resources-dir <原版 resources/libero 目录>`，避免评测时更新资源。
也可使用原框架的 `--planner claude_code` 或 `--planner codex`，并配置相应运行依赖。
原 Planner 默认预算为 100 turns、8192 单次输出 token、1200 秒；对比时应显式固定这些参数。

## 产物与验证

每次运行保存 `source_manifest.json`、`memory_manifest.json`、`environment.json`、
渲染后的原提示词和 `run_config.json`；`source_unchanged.json` 检查原源码内容是否保持一致。
`workspace/` 保存独立资源副本，`artifacts/` 保存原版 states、图像、depth/world map、
执行 recipe 和视频。正式运行另保存 Planner 结果与独立官方评分文件
`evaluation.json`、`episodes.tsv`、`task_summary.tsv`、`summary.json`、`aggregate.json`。

接口测试无需 HarnessVLA 或真实模型：

```bash
.venv/bin/python -m pytest benchmark/robomemarena/tests/test_harness_baseline.py -q
```

正式推理还依赖可用的 Planner、SAM3 和 VLA 服务。接线 smoke 通过不代表这些模型完成了
闭环验证，也不能作为任务成功率证据。

当前 11 项适配测试通过，包含真实服务返回 10 个动作的短序列回归；此前的 2 项原版
Planner 测试与 3 项环境服务回归测试通过。真实 MuJoCo smoke 完成了双视角反投影和
夹爪控制，生成可读取的 H.264 视频，源码内容指纹保持一致。

Claude Code 2.1.259 + `deepseek-v4-flash-vision-exp` 已通过图片读取和工具调用测试。
原版 Planner 已实际调用 SAM3、反投影和运动工具；修复短动作序列适配后，原版
`pi0_pick(max_chunks=1)` 完成一个 VLA chunk。自主试跑按用户要求停止，测试日志已清理；
这些结果仅证明接口可用，不构成完整任务或 26 任务成功率评测。
