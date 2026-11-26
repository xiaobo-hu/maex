#!/usr/bin/env bash
# 常驻三服务逐任务评测 [START, END]:policy/环境/Observer 只启动一次,
# 每个任务仅执行一次 roboagent_eval(env server 支持反复 reset 换任务),
# 单任务失败不中断；传 BATCH_ID 时统一写入独立批次目录。
#
# 用法: bash run_batch_persistent.sh [CONFIG_FILE] [START] [END] [BATCH_ID]
# 传 BATCH_ID 时，全部任务产物写入 logs/batches/<BATCH_ID>/；不传时保持旧布局。

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
CONFIG_FILE="${1:-$PROJECT_ROOT/benchmark/robomemarena/configs/qwen_recovery.env}"
START="${2:-1}"
END="${3:-26}"
BATCH_ID="${4:-}"

if [[ ! -f "$CONFIG_FILE" ]]; then
  echo "Evaluation config not found: $CONFIG_FILE" >&2
  exit 2
fi

# shellcheck disable=SC1090,SC1091
source "$CONFIG_FILE"
source "$SCRIPT_DIR/lib.sh"

: "${TASK_ID:?TASK_ID is required}"
: "${SEED:?SEED is required}"
: "${PLANNER:?PLANNER is required}"
: "${MODEL:?MODEL is required}"

if [[ -n "$BATCH_ID" && ! "$BATCH_ID" =~ ^[[:alnum:]_.-]+$ ]]; then
  echo "BATCH_ID may contain only letters, digits, dot, underscore, and hyphen" >&2
  exit 2
fi
if (( START < 1 || END < START )); then
  echo "Invalid task range: $START..$END" >&2
  exit 2
fi

load_planner_llm_env

if [[ -n "$BATCH_ID" ]]; then
  BATCH_DIR="$PROJECT_ROOT/benchmark/robomemarena/logs/batches/$BATCH_ID"
  RUNS_DIR="$BATCH_DIR/runs"
  SERVICE_DIR="$BATCH_DIR/services/session-$(date +%Y%m%d-%H%M%S)"
  STATUS_FILE="$BATCH_DIR/status.log"
  MANIFEST_FILE="$BATCH_DIR/manifest.jsonl"
  BATCH_CONFIG_FILE="$BATCH_DIR/batch_config.env"
  mkdir -p "$RUNS_DIR" "$SERVICE_DIR"
  if [[ -f "$BATCH_CONFIG_FILE" ]] && ! cmp -s "$CONFIG_FILE" "$BATCH_CONFIG_FILE"; then
    echo "Batch config differs from existing snapshot: $BATCH_CONFIG_FILE" >&2
    exit 2
  fi
  if [[ ! -f "$BATCH_CONFIG_FILE" ]]; then
    cp "$CONFIG_FILE" "$BATCH_CONFIG_FILE"
  fi
else
  BATCH_DIR="$PROJECT_ROOT/benchmark/robomemarena/logs/batch"
  RUNS_DIR="$PROJECT_ROOT/benchmark/robomemarena/logs"
  SERVICE_DIR="$BATCH_DIR/services-$(date +%Y%m%d-%H%M%S)"
  STATUS_FILE="$BATCH_DIR/current.status"
  MANIFEST_FILE=""
  mkdir -p "$SERVICE_DIR"
fi

policy_pid=""
env_pid=""
observer_pid=""

cleanup() {
  trap - EXIT INT TERM
  for pid in "$policy_pid" "$env_pid" "$observer_pid"; do
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
      kill -TERM "$pid" 2>/dev/null || true
    fi
  done
  wait "$policy_pid" "$env_pid" "$observer_pid" 2>/dev/null || true
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

cd "$PROJECT_ROOT"

policy_pid="$(start_policy_service "$SERVICE_DIR")"
env_pid="$(start_environment_service "$SERVICE_DIR")"
if [[ "${OBSERVER_EXTERNAL:-0}" != 1 ]]; then
  observer_pid="$(start_observer_service "$SERVICE_DIR")"
fi

wait_for_service "$POLICY_PORT" policy websocket || exit 1
wait_for_service "$ENV_PORT" environment websocket || exit 1
wait_for_service "$OBSERVER_PORT" observer http || exit 1

ensure_services_alive() {
  # 任务间只做存活检查:端口还在就复用,挂了才重启该服务。
  wait_for_service "$POLICY_PORT" policy websocket || {
    echo "[$(date '+%F %T')] restart policy service" >>"$STATUS_FILE"
    policy_pid="$(start_policy_service "$SERVICE_DIR")"
    wait_for_service "$POLICY_PORT" policy websocket || return 1
  }
  wait_for_service "$ENV_PORT" environment websocket || {
    echo "[$(date '+%F %T')] restart environment service" >>"$STATUS_FILE"
    env_pid="$(start_environment_service "$SERVICE_DIR")"
    wait_for_service "$ENV_PORT" environment websocket || return 1
  }
  wait_for_service "$OBSERVER_PORT" observer http || {
    [[ "${OBSERVER_EXTERNAL:-0}" != 1 ]] || return 1
    echo "[$(date '+%F %T')] restart observer service" >>"$STATUS_FILE"
    observer_pid="$(start_observer_service "$SERVICE_DIR")"
    wait_for_service "$OBSERVER_PORT" observer http || return 1
  }
}

model_tag="${MODEL//[^[:alnum:]._-]/-}"
echo "batch start $(date -Is) config=$CONFIG_FILE tasks=$START..$END services=$SERVICE_DIR" >>"$STATUS_FILE"
for ((task = START; task <= END; task++)); do
  if [[ "${FORCE_RERUN:-0}" != "1" ]] && \
    [[ -n "$MANIFEST_FILE" ]] && [[ -f "$MANIFEST_FILE" ]] && \
    grep -Eq "\"task_id\":${task},.*\"exit_code\":0,.*\"artifact_complete\":true" "$MANIFEST_FILE"; then
    echo "[$(date '+%F %T')] task $task SKIP already complete" >>"$STATUS_FILE"
    continue
  fi
  if ! ensure_services_alive; then
    echo "[$(date '+%F %T')] task $task NOT STARTED: required service unavailable; batch stopped" | tee -a "$STATUS_FILE" >&2
    exit 1
  fi
  cfg="$BATCH_DIR/task${task}.env"
  sed "s/^TASK_ID=.*/TASK_ID=${task}/" "$CONFIG_FILE" >"$cfg"
  if [[ -n "$BATCH_ID" ]]; then
    run_dir="$RUNS_DIR/task$(printf '%02d' "$task")-attempt-$(date +%Y%m%d-%H%M%S)"
  else
    run_dir="$RUNS_DIR/task${task}-seed${SEED}-${model_tag}-$(date +%Y%m%d-%H%M%S)"
  fi
  mkdir -p "$run_dir"
  cp "$cfg" "$run_dir/eval_config.env"
  begin=$(date +%s)
  echo "[$(date '+%F %T')] task $task START cfg=$cfg" | tee -a "$STATUS_FILE"
  # 常驻服务下环境由 roboagent_eval 每次 reset 换任务,无需重启。
  TASK_ID="$task" run_roboagent_eval "$run_dir" 2>&1 | tee "$run_dir/run.log"
  rc=${PIPESTATUS[0]}
  end=$(date +%s)
  duration_seconds=$((end - begin))
  artifact_complete=true
  for artifact in trace.json evaluation.json summary.json run_config.json; do
    if [[ ! -f "$run_dir/$artifact" ]]; then
      artifact_complete=false
      break
    fi
  done
  echo "[$(date '+%F %T')] task $task EXIT=$rc dur=$((duration_seconds / 60))min complete=$artifact_complete" >>"$STATUS_FILE"
  if [[ -n "$MANIFEST_FILE" ]]; then
    run_rel="${run_dir#"$PROJECT_ROOT"/}"
    printf '{"batch_id":"%s","task_id":%d,"seed":%d,"model":"%s","run_dir":"%s","exit_code":%d,"duration_seconds":%d,"artifact_complete":%s}\n' \
      "$BATCH_ID" "$task" "$SEED" "$model_tag" "$run_rel" "$rc" \
      "$duration_seconds" "$artifact_complete" >>"$MANIFEST_FILE"
  fi
done
echo "batch done $(date -Is)" >>"$STATUS_FILE"
