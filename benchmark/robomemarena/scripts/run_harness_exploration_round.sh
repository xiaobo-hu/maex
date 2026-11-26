#!/usr/bin/env bash
# 一轮逐任务探索：每个任务一次 episode，共享官方记忆库，不在轮内重试失败任务。
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: bash $0 BATCH_DIR MEMORY_DIR [FIRST_TASK=1] [LAST_TASK=26]" >&2
  exit 2
fi
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/../../.." && pwd)
batch_dir=$(realpath -m "$1")
memory_dir=$(realpath -m "$2")
first_task=${3:-1}
last_task=${4:-26}
if (( first_task < 1 || last_task > 26 || first_task > last_task )); then
  echo "Task range must be within 1..26" >&2
  exit 2
fi
if [[ -e "$batch_dir" ]]; then
  echo "Use a new BATCH_DIR: $batch_dir" >&2
  exit 2
fi
cd "$PROJECT_ROOT"
source models/harnessvla/runtime.env
source "$SCRIPT_DIR/lib.sh"
# Claude 的独立凭据文件只按赋值读取，避免混入其他基线的 .env 配置。
unset LLM_API_KEY LLM_BASE_URL LLM_MODEL_NAME
while IFS= read -r line || [[ -n "$line" ]]; do
  case "$line" in
    LLM_*=*) export "$line" ;;
  esac
done < "$PROJECT_ROOT/.env_claude"
: "${LLM_API_KEY:?Configure LLM_API_KEY in .env_claude}"
: "${LLM_BASE_URL:?Configure LLM_BASE_URL in .env_claude}"
: "${LLM_MODEL_NAME:?Configure LLM_MODEL_NAME in .env_claude}"
MODEL="$LLM_MODEL_NAME"
PI05_PYTHON="$PROJECT_ROOT/models/harnessvla/official-pi05-venv/bin/python"
PI05_CHECKPOINT_PATH=${PI05_CHECKPOINT_PATH:-"$PROJECT_ROOT/models/harnessvla/RLinf-Pi05-LIBERO-"}
: "${HARNESS_SEED:=0}"
: "${HARNESS_MAX_STEPS:=2500}"
: "${HARNESS_MAX_TURNS:=100}"
: "${HARNESS_PLANNER_TIMEOUT:=1200}"
export HARNESS_SEED HARNESS_MAX_STEPS HARNESS_MAX_TURNS HARNESS_PLANNER_TIMEOUT
export PI05_CHECKPOINT_PATH
[[ -x "$PI05_PYTHON" && -f "$PI05_CHECKPOINT_PATH/model.safetensors" ]]
unset ANTHROPIC_API_KEY
export ANTHROPIC_AUTH_TOKEN="$LLM_API_KEY"
claude_base_url=${LLM_BASE_URL%/}
case "$claude_base_url" in
  */anthropic) export ANTHROPIC_BASE_URL="$claude_base_url" ;;
  *) export ANTHROPIC_BASE_URL="${claude_base_url%/v1}/anthropic" ;;
esac
export CUDA_VISIBLE_DEVICES=2 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$PROJECT_ROOT/benchmark/.external/RPent-official:$PROJECT_ROOT"
export ROBOMEMARENA_ROOT="$PROJECT_ROOT/benchmark/.external/RoboMemArena"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export NUMBA_CACHE_DIR=/tmp/harness-round-numba MPLCONFIGDIR=/tmp/harness-round-mpl
POLICY_GPU=2
POLICY_PORT=8020
ENV_PORT=8021
SAM_PORT=8022
env_python=$(conda run --no-capture-output -n roboagent-robomemarena bash -c 'printf "%s/bin/python\n" "$CONDA_PREFIX"')
mkdir -p "$batch_dir/services" "$memory_dir"

"$HARNESS_PYTHON" - "$batch_dir" "$memory_dir" "$first_task" "$last_task" <<'PY'
import json, os, sys
from pathlib import Path
from benchmark.robomemarena.baselines.harness_prompts import PROMPT_ADAPTATION
batch, memory, first, last = sys.argv[1:]
config = dict(data_role="exploration", tasks=list(range(int(first), int(last)+1)),
              seed=int(os.environ["HARNESS_SEED"]), sessions=1, attempts_per_session=1,
              max_steps=int(os.environ["HARNESS_MAX_STEPS"]),
              max_turns=int(os.environ["HARNESS_MAX_TURNS"]),
              planner_timeout_s=int(os.environ["HARNESS_PLANNER_TIMEOUT"]),
              process_timeout_s=int(os.environ["HARNESS_PLANNER_TIMEOUT"])+120,
              planner="claude_code", model=os.environ["LLM_MODEL_NAME"], gpu=2,
              memory_dir=memory, policy="RLinf/RLinf-Pi05-LIBERO-130-fullshot-SFT", policy_port=8020,
              checkpoint=os.environ["PI05_CHECKPOINT_PATH"], action_chunk_size=5,
              env_port=8021, sam3_port=8022, task_reference="same as RoboAgent/HiMe",
              claude_config_file=".env_claude", prompt_adaptation=PROMPT_ADAPTATION)
marker = Path(memory)/"policy_identity.json"
identity = {"policy": config["policy"], "checkpoint": config["checkpoint"]}
if marker.exists():
    if json.loads(marker.read_text()) != identity:
        raise SystemExit("Memory policy identity differs; use a new memory directory")
elif any(Path(memory).iterdir()):
    raise SystemExit("Existing unlabelled memory cannot be reused with official weights; use a new directory")
else:
    marker.write_text(json.dumps(identity, indent=2)+"\n")
(Path(batch)/"batch_config.json").write_text(json.dumps(config, indent=2)+"\n")
PY

policy_pid=""
env_pid=""
sam_pid=""
cleanup() {
  for pid in "$policy_pid" "$env_pid" "$sam_pid"; do
    if [[ -n "$pid" ]]; then kill "$pid" 2>/dev/null || true; fi
  done
}
trap cleanup EXIT
trap 'exit 130' INT TERM
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 XLA_PYTHON_CLIENT_PREALLOCATE=false \
  "$PI05_PYTHON" -m rpent.robots.components.pi05_vla_server \
  --embodiment libero --model-path "$PI05_CHECKPOINT_PATH" \
  --transport http --host 127.0.0.1 --port "$POLICY_PORT" >"$batch_dir/services/policy.log" 2>&1 &
policy_pid=$!
"$env_python" -m benchmark.robomemarena.baselines.harness_server --host 127.0.0.1 --port "$ENV_PORT" >"$batch_dir/services/env.log" 2>&1 &
env_pid=$!
"$SAM3_PYTHON" -m rpent.robots.components.sam3_server --transport http --host 127.0.0.1 --port "$SAM_PORT" >"$batch_dir/services/sam3.log" 2>&1 &
sam_pid=$!
deadline=$((SECONDS + 600))
until "$HARNESS_PYTHON" -c "from rpent.utils.rpc.http_rpc import HttpRpcClient; HttpRpcClient('http://127.0.0.1:$POLICY_PORT').call('healthz', timeout_s=2)" >/dev/null 2>&1; do
  kill -0 "$policy_pid"
  if (( SECONDS >= deadline )); then echo "Official Pi0.5 startup timed out" >&2; exit 1; fi
  sleep 2
done
wait_for_service "$ENV_PORT" environment websocket
deadline=$((SECONDS + 180))
until "$HARNESS_PYTHON" -c "from rpent.utils.rpc.http_rpc import HttpRpcClient; HttpRpcClient('http://127.0.0.1:$SAM_PORT').call('healthz', timeout_s=2)" >/dev/null 2>&1; do
  if (( SECONDS >= deadline )); then echo "SAM3 startup timed out" >&2; exit 1; fi
  sleep 1
done

for (( task=first_task; task<=last_task; task++ )); do
  kill -0 "$policy_pid" "$env_pid" "$sam_pid"
  task_dir="$batch_dir/task$(printf '%02d' "$task")"
  echo "$(date -Is) task $task START" | tee -a "$batch_dir/status.log"
  task_exit=0
  timeout --signal=INT --kill-after=30s "$((HARNESS_PLANNER_TIMEOUT + 120))s" \
    bash "$SCRIPT_DIR/run_harness_official.sh" "$task_dir" "$memory_dir" \
    --task "$task" --seed "$HARNESS_SEED" --max-steps "$HARNESS_MAX_STEPS" \
    --env-port "$ENV_PORT" --policy-port "$POLICY_PORT" \
    --sam3-endpoint "http://127.0.0.1:$SAM_PORT" \
    --planner claude_code --model "$MODEL" \
    --explore --explore-sessions 1 --explore-attempts-per-session 1 \
    --max-turns "$HARNESS_MAX_TURNS" --planner-timeout-s "$HARNESS_PLANNER_TIMEOUT" \
    >"$batch_dir/task$(printf '%02d' "$task").log" 2>&1 || task_exit=$?
  "$HARNESS_PYTHON" - "$batch_dir" "$task_dir" "$task" "$task_exit" <<'PY'
import json, sys
from pathlib import Path
batch, task_dir, task, code = sys.argv[1:]
run = Path(task_dir)/"run"
completion = run/"completion.json"
record = dict(task_id=int(task), exit_code=int(code), run_dir=str(run))
if completion.exists():
    record["completion"] = json.loads(completion.read_text())
record["episodes"] = [json.loads(p.read_text()) for p in sorted((run/"episodes").glob("*/evaluation.json"))]
with (Path(batch)/"manifest.jsonl").open("a") as output:
    output.write(json.dumps(record)+"\n")
PY
  echo "$(date -Is) task $task END exit=$task_exit" | tee -a "$batch_dir/status.log"
  if (( task_exit != 0 )); then
    echo "$(date -Is) ROUND STOPPED: task $task runtime error; inspect before resuming" | tee -a "$batch_dir/status.log"
    exit "$task_exit"
  fi
done
echo "$(date -Is) ROUND COMPLETE" | tee -a "$batch_dir/status.log"
