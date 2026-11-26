#!/usr/bin/env bash
# 使用 RoboMemArena 的现有服务启动函数运行 MAEX。
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
source benchmark/robomemarena/configs/deepseek.env
source benchmark/robomemarena/scripts/lib.sh
load_planner_llm_env

POLICY_PORT="${MAEX_POLICY_PORT:-$POLICY_PORT}"
ENV_PORT="${MAEX_ENV_PORT:-$ENV_PORT}"
OBSERVER_PORT="${MAEX_OBSERVER_PORT:-$OBSERVER_PORT}"

task="${1:-2}"
seed="${2:-$SEED}"
if (( $# >= 2 )); then
  shift 2
elif (( $# == 1 )); then
  shift
fi

output=benchmark/robomemarena/logs/maex
mkdir -p "$output"
run_dir="$(mktemp -d "$output/task${task}-seed${seed}-$(date -u +%Y%m%dT%H%M%SZ)-XXXXXX")"
service_pids=()

cleanup() {
  for pid in "${service_pids[@]}"; do
    kill -TERM "$pid" 2>/dev/null || true
  done
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

if [[ "${MAEX_EXTERNAL_SERVICES:-0}" != 1 ]]; then
  mkdir "$run_dir/services"
  service_pids+=("$(start_policy_service "$run_dir/services")")
  service_pids+=("$(start_environment_service "$run_dir/services")")
  service_pids+=("$(start_observer_service "$run_dir/services")")
fi

wait_for_service "$POLICY_PORT" policy websocket
wait_for_service "$ENV_PORT" environment websocket
wait_for_service "$OBSERVER_PORT" observer http

.venv/bin/python -u -m maex.run_eval \
  --task "$task" --seed "$seed" --run-dir "$run_dir" \
  --policy-port "$POLICY_PORT" --env-port "$ENV_PORT" \
  --max-steps "$MAX_STEPS" --action-chunk-size "$ACTION_CHUNK_SIZE" \
  --attempt-chunks "$MAX_VLA_CHUNKS_PER_CALL" --video-fps "$VIDEO_FPS" \
  --planner "$PLANNER" --model "$MODEL" --planner-retries "$PLANNER_RETRIES" \
  --observer-base-url "http://127.0.0.1:$OBSERVER_PORT/v1" \
  --observer-model "$OBSERVER_SERVED_MODEL_NAME" --observer-timesteps "$OBSERVER_TIMESTEPS" "$@"
