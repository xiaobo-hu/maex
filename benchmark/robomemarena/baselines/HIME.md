# HiMe-Same-Backbone 接入实验

`hime_eval.py` 在独立进程中直接组合本地原版 HiMe 的 `ServerTaskManager`、
Planner、Sentry、`MultiTagMemory` 和两轮记忆协议。原 `HiMe/` 源码不修改；
每个 episode 构造全新的记忆实例，不使用 resume，也不进入 RoboAgent runtime。
改动仅包括任务 prompt、模型服务参数、相机/VLA 接口和评测生命周期。

Planner 使用根目录 `.env` 中的 `LLM_API_KEY`、`LLM_BASE_URL`、`LLM_MODEL_NAME`，
输出上限读取 `LLM_MAX_TOKENS`，DeepSeek thinking 参数读取 `LLM_DISABLE_THINKING`。
当前接入目标为 `deepseek-v4-flash-vision-exp`、Qwen3-VL-32B-Instruct 和同一冻结的
`pi05_robomemarena`。Planner/Sentry 发生连接错误或连接超时后，分别等待 2、4 秒，最多
重试两次同一请求。SDK 内置重试关闭，适配层逐次记录 attempt，每次尝试计入实际请求上限。
非流式完整响应返回前不执行记忆操作或新的机器人动作，因此重试不会重复其副作用。
HTTP 状态错误、输出截断和解析问题不通过这条连接重试路径重试。持续连接失败仍使本次
rollout 无效，需要从 reset 重新运行；不从中断动作步恢复。

## 准备与运行

依赖已有的 `HiMe/` checkout、OpenPI checkpoint、Qwen 权重、项目 `.venv` 和
`roboagent-robomemarena` conda 环境。当前机器的 `.venv` 为 Python 3.13，已验证可导入
HiMe；本地 embedding 另使用该环境现有的 torch、transformers 和 huggingface_hub。
不需要 ROS、Widow 机器人客户端或 HiMe HTTP 服务。

默认配置使用本地文本 embedding 进行接入试跑。首次下载：

```bash
.venv/bin/python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
    "sentence-transformers/all-MiniLM-L6-v2",
    local_dir="models/all-MiniLM-L6-v2",
    allow_patterns=["config.json", "model.safetensors", "tokenizer.json",
                    "tokenizer_config.json", "special_tokens_map.json", "vocab.txt",
                    "sentence_bert_config.json", "README.md"],
)
PY
```

本地编码采用 [模型官方说明](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
中的 masked mean pooling 和归一化，最大长度为 256 tokens，在 CPU 上运行。
这是相对原 HiMe embedding 的模型替换；配置中明确标记 `local-MiniLM-adaptation`，
不能将其写成完全复现原 embedding 配置的结果。HiMe 标签检索、文本回退和排序代码保留。

从项目根目录运行一个 300 步短实验：

```bash
bash benchmark/robomemarena/scripts/run_hime_eval.sh 1 50 300
```

三个位置参数分别为 task、seed、环境动作预算。完整 episode 使用 2500 步：

```bash
bash benchmark/robomemarena/scripts/run_hime_eval.sh 1 50 2500
```

脚本默认在 GPU 0 上启动 VLA/环境、GPU 1 上启动 Sentry，使用 8020/8021/8022 端口。
通过进程环境 `POLICY_GPU`、`ENV_GPU`、`OBSERVER_GPU` 和对应 `*_PORT` 可以覆盖。
服务参数位于 `configs/hime_same_backbone.env`。已有服务必须显式复用，环境必须是本实验
独占的实例，不能 reset 正在执行其他实验的服务：

```bash
HIME_REUSE_SERVICES=1 bash benchmark/robomemarena/scripts/run_hime_eval.sh 1 50 300
```

连续运行 Task 1–26 时，可以逐任务冷启动；正常任务失败仍继续，启动或 API 异常则停止，
避免把未执行任务误计为失败。将起始数字改成首个尚未有效执行的 task 即可续跑：

```bash
for task_id in $(seq 1 26); do
  HIME_REUSE_SERVICES=0 bash benchmark/robomemarena/scripts/run_hime_eval.sh "$task_id" 50 2500 || {
    echo "Task $task_id 运行异常，批跑已停止；修复后从该任务续跑。"
    break
  }
done
```

启动脚本会等待本次服务及其子进程退出；端口检测允许复用 TCP `TIME_WAIT` 状态，
但仍拒绝已有监听服务。旧脚本曾在没有监听进程时将 `TIME_WAIT` 误报为端口占用；
只有服务目录、没有 `episode/result.json` 的任务并未完成评测，需要补跑。

如需原版 API embedding，在本地 `.env` 或进程环境配置
`EMBEDDING_OPENAI_API_KEY`、`EMBEDDING_OPENAI_BASE_URL`、`EMBEDDING_MODEL`，并运行：

```bash
HIME_LOCAL_EMBEDDING_MODEL='' bash benchmark/robomemarena/scripts/run_hime_eval.sh 1 50 2500
```

正式运行前会调用一次真实 embedding 验证连接。embedding 失败不会静默退化到空记忆或
零向量。密钥不会写进运行配置或提交文件。

## 任务与输入适配

两份适配 prompt 位于 `hime_prompts/`，由原 `task3.txt` / `task3_obs.txt` 派生。
Planner 保留第一轮 QUERY、第二轮 CREATE/UPDATE/DELETE 与最终计划，以及
`[done] / [current] / [pending]` 格式。任务词汇和子任务语法改为 RoboMemArena primitive，
pick/place 分开。open/close/pour 增加对应视觉完成条件；pour 允许使用当前步骤的时序证据，
不能仅凭最后一帧或前一次倾倒判断完成。Sentry 仍只输出 done/not_done。

初始输入包含官方任务文本及与现有 Ours 相同的 `task_primitive_reference()` 内容。
这些是有序 checkpoint 标签参考，部分包含具体目标位置；本实验并非完全无步骤提示的
任务拆分实验。执行时以模型产生的当前步骤为准，runner 不按参考自动推进。

双视角使用 benchmark 现有 flipud/INTER_AREA 预处理。HiMe 拼图保持左腕部、右外部；
在线不传入真实物体坐标、隐藏阶段、阶段完成事件或倾倒真值。

## 调度、预算与评分

Planner 格式适配 `xml-summary-status-v1` 在接入层将误嵌套或漏闭合的
`summary` 与 `memory_operations`、`plan_list` 恢复为并列字段，并在计划字段内
将 `[当前]`、`[待处理]`、`[已完成]` 映射到原英文状态标签。
动作文本和记忆操作内容不翻译、不补写；其他 XML 错误继续按原协议处理。
不增加模型调用，也不改变原 HiMe 调度与完成判定。`model_calls.jsonl` 保留原始
`response`，发生修复时另记 `normalized_response` 和 `format_revision`；
`run_config.json` 记录 `planner_format_revision`。旧异常记录保留，新运行采用此版本。

- VLA 每次最多执行 20 个动作，返回不足 20 时执行实际返回数，与现有 Ours 的截断语义一致。
  当前服务元数据标明 `action_horizon=10`，实际 chunk 长度另记入 trace。
- 每 5 个 policy calls 上传一批 chunk 结束帧，Sentry 看最近 8 帧；每帧含双相机。
  当前 10 动作 horizon 对应通常每 50 个物理步观察一次，不是每 100 步。
- 原 Sentry done 触发保留；累计帧数严格大于 50 的 Planner fallback 保留。
- Planner 使用原两轮协议，默认最多 80 次实际 API 请求；Sentry 默认最多 125 次。
  所有请求记录 usage 和耗时。这些是独立请求上限，并不声称与 Ours 的总 token 预算匹配。
- 每次运行使用空记忆；不接入 Ours 的过程 Observer、Reflection、恢复工具或跨 episode Skill。
- 环境 done 或物理预算耗尽后不再请求模型。HiMe all-done 只触发 benchmark 原有 settle，
  最终 TSR/CSR 来自 `metrics()`，不能用内部计划完成冒充成功。

调度参数可以通过独立入口显式传入；试跑确定后应冻结，同一批任务共用：

```bash
.venv/bin/python -m benchmark.robomemarena.baselines.hime_eval \
  --task 1 --seed 50 --max-steps 2500 \
  --sentry-interval 5 --sentry-window 8 --fallback-images 50 \
  --local-embedding-model models/all-MiniLM-L6-v2 \
  --run-dir benchmark/robomemarena/logs/hime-manual-task1-seed50
```

## 产物与验证

启动脚本将产物保存到 `logs/hime/task*-seed*-*/episode/`，包括：

- `run_config.json`、`policy_metadata.json`：模型、预算、原 HiMe revision/源码指纹；
- `trace.jsonl`：实际 subtask、动作块长度、计划更新与终止原因；
- `model_calls.jsonl`、`embedding_calls.jsonl`：请求、响应、用量和耗时，不保存图像 base64；
- `_server_data/`、`memory.json`：原 HiMe 图像、轮次日志、计划及记忆快照；
- `videos/`：动作执行期间的 external/wrist 视频；
- `evaluation.json`、`episodes.tsv` 等官方格式成绩，以及 `result.json` 的运行有效性标记。

API、模型解析或接线错误退出码为 1，`result.json.valid_rollout=false`；已执行部分的评分
仅作为诊断保留。正常预算耗尽属于有效失败运行。300 步测试只是接线验证，不能与 2500 步
正式结果混合汇总。单 seed 的一次 Task 1 运行也不代表 26 任务成功率。

```bash
.venv/bin/python -m pytest -q benchmark/robomemarena/tests/test_hime_baseline.py
```

测试覆盖原 HiMe 两轮图文记忆检索、Sentry not_done/fallback、episode 记忆隔离、
模型参数及请求上限、实际短动作块、终态截断与评分信息隔离。

## 已完成的接入验证

2026-09-09：Task 1 / seed 50，在 2500 步预算下完成真实闭环，官方 TSR=1、CSR=100%。
使用 DeepSeek `deepseek-v4-flash-vision-exp`、Qwen3-VL-32B-Instruct、冻结
`pi05_robomemarena` 和本地 MiniLM embedding。共执行 100 次 VLA 调用、1000 个动作步，
终态确认后环境总步数为 1185；Planner 10 次请求、Sentry 20 次请求。两个官方放置阶段
分别在第 509、985 步通过。此结果仅证明该任务的接入闭环可运行。

产物：`logs/hime/task1-seed50-20260909-180820-1087189/episode/`。
`validation.json` 保存源码/适配器/prompt 指纹、模型用量及视频检查。
原 HiMe 源码指纹保持一致；双视角视频均为 H.264、256×256，包含初始帧、每个 VLA 动作步
和 settle 结束帧（settle RPC 的中间空动作帧不包含在视频中）。

以下最小回归集合共 20 项通过，另通过新增 Python 文件的 Ruff 检查与启动脚本语法检查：

```bash
.venv/bin/python -m pytest -q \
  benchmark/robomemarena/tests/test_hime_baseline.py \
  benchmark/robomemarena/tests/test_preprocessing.py \
  benchmark/robomemarena/tests/server/test_evaluation.py \
  benchmark/robomemarena/tests/client/test_artifacts.py
```
