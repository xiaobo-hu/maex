#!/usr/bin/env bash
# 启动本地三服务并将一次 RoboAgent 评测产物收拢到同一时间戳目录。

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
CONFIG_FILE="${1:-$PROJECT_ROOT/benchmark/robomemarena/configs/qwen_recovery.env}"

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

load_planner_llm_env

timestamp="$(date +%Y%m%d-%H%M%S)"
model_tag="${MODEL//[^[:alnum:]._-]/-}"
RUN_DIR="$PROJECT_ROOT/benchmark/robomemarena/logs/task${TASK_ID}-seed${SEED}-${model_tag}-${timestamp}"
SERVICE_DIR="$RUN_DIR/services"
mkdir -p "$SERVICE_DIR"
cp "$CONFIG_FILE" "$RUN_DIR/eval_config.env"

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
trap cleanup EXIT INT TERM

cd "$PROJECT_ROOT"

policy_pid="$(start_policy_service "$SERVICE_DIR")"
env_pid="$(start_environment_service "$SERVICE_DIR")"
observer_pid="$(start_observer_service "$SERVICE_DIR")"

wait_for_service "$POLICY_PORT" policy websocket
wait_for_service "$ENV_PORT" environment websocket
wait_for_service "$OBSERVER_PORT" observer http

set +e
run_roboagent_eval "$RUN_DIR" 2>&1 | tee "$RUN_DIR/run.log"
status=${PIPESTATUS[0]}
set -e

echo "Evaluation artifacts: $RUN_DIR"
exit "$status"
