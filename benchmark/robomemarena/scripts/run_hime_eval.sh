#!/usr/bin/env bash
# 启动独立端口的三服务并运行一个 HiMe episode；可显式复用已启动的服务。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$PROJECT_ROOT"
source benchmark/robomemarena/configs/hime_same_backbone.env
source "$SCRIPT_DIR/lib.sh"

TASK_ID=${1:-1}
SEED=${2:-50}
MAX_STEPS=${3:-2500}
JOB_DIR="benchmark/robomemarena/logs/hime/task${TASK_ID}-seed${SEED}-$(date +%Y%m%d-%H%M%S)-$$"
mkdir -p "$JOB_DIR/services"
cp benchmark/robomemarena/configs/hime_same_backbone.env "$JOB_DIR/service_config.env"
service_pids=()
stop_service_tree() {
  local pid="$1" child
  # conda run 还包含 bash/python 子进程，先回收后代，避免残留环境占用端口。
  while read -r child; do
    [[ -n "$child" ]] && stop_service_tree "$child"
  done < <(pgrep -P "$pid" || true)
  kill -TERM "$pid" 2>/dev/null || true
  local deadline=$((SECONDS + 30)) state
  while state=$(ps -o stat= -p "$pid") && [[ "$state" != Z* ]]; do
    if (( SECONDS >= deadline )); then
      kill -KILL "$pid" 2>/dev/null || true
      break
    fi
    sleep 0.2
  done
}
cleanup() {
  trap - EXIT INT TERM
  for pid in "${service_pids[@]}"; do
    stop_service_tree "$pid"
  done
}
trap cleanup EXIT INT TERM

embedding_args=()
if [[ -n "$HIME_LOCAL_EMBEDDING_MODEL" ]]; then
  if [[ ! -f "$HIME_LOCAL_EMBEDDING_MODEL/config.json" ]]; then
    echo "Local embedding model is missing; see baselines/HIME.md for download instructions." >&2
    exit 2
  fi
  embedding_args=(--local-embedding-model "$HIME_LOCAL_EMBEDDING_MODEL")
fi

if [[ "$HIME_REUSE_SERVICES" != 1 ]]; then
  # 不允许意外接管已有实验服务；复用需要显式设置 HIME_REUSE_SERVICES=1。
  .venv/bin/python - "$POLICY_PORT" "$ENV_PORT" "$OBSERVER_PORT" <<'PY'
import socket
import sys
for port in map(int, sys.argv[1:]):
    with socket.socket() as listener:
        # 与服务端一致：允许复用已关闭连接的 TIME_WAIT，但不允许已有监听服务。
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            listener.bind(('127.0.0.1', port))
        except OSError:
            raise SystemExit(f'Port {port} is occupied; select free ports or explicitly reuse services')
PY
  service_pids+=("$(start_policy_service "$JOB_DIR/services")")
  service_pids+=("$(start_environment_service "$JOB_DIR/services")")
  service_pids+=("$(start_observer_service "$JOB_DIR/services")")
fi
wait_for_service "$POLICY_PORT" policy websocket
wait_for_service "$ENV_PORT" environment websocket
wait_for_service "$OBSERVER_PORT" observer http

.venv/bin/python -u -m benchmark.robomemarena.baselines.hime_eval \
  --task "$TASK_ID" --seed "$SEED" --max-steps "$MAX_STEPS" \
  --policy-port "$POLICY_PORT" --env-port "$ENV_PORT" \
  --sentry-base-url "http://127.0.0.1:$OBSERVER_PORT/v1" \
  --sentry-model "$OBSERVER_SERVED_MODEL_NAME" \
  --run-dir "$JOB_DIR/episode" "${embedding_args[@]}" \
  2>&1 | tee "$JOB_DIR/run.log"
echo "HiMe artifacts: $JOB_DIR/episode"
