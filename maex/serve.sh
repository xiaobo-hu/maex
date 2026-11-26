#!/usr/bin/env bash
# 常驻一个策略和 Observer 服务，按需为并行 episode 分配独立的仿真端口。
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
source benchmark/robomemarena/configs/deepseek.env
source benchmark/robomemarena/scripts/lib.sh

POLICY_PORT="${MAEX_POLICY_PORT:-$POLICY_PORT}"
ENV_PORT="${MAEX_ENV_PORT:-$ENV_PORT}"
OBSERVER_PORT="${MAEX_OBSERVER_PORT:-$OBSERVER_PORT}"
environment_ports=("$@")
if (( ${#environment_ports[@]} == 0 )); then
  environment_ports=("$ENV_PORT")
fi
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export NO_PROXY="${NO_PROXY:+$NO_PROXY,}127.0.0.1,localhost"
export no_proxy="${no_proxy:+$no_proxy,}127.0.0.1,localhost"

output=benchmark/robomemarena/logs/maex
mkdir -p "$output"
log_dir="$(mktemp -d "$output/services-XXXXXX")"
service_pids=()

cleanup() {
  for pid in "${service_pids[@]}"; do
    kill -TERM "$pid" 2>/dev/null || true
  done
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

CUDA_VISIBLE_DEVICES="$POLICY_GPU" ROBOMEMARENA_ROOT="$PROJECT_ROOT/benchmark/.external/RoboMemArena" \
  models/openpi/.venv/bin/python baselines/robomemarena/serve_policy.py \
  --port "$POLICY_PORT" --checkpoint models/openpi/checkpoints/pi05_robomemarena \
  >"$log_dir/policy.log" 2>&1 &
service_pids+=("$!")
service_pids+=("$(start_observer_service "$log_dir")")
for port in "${environment_ports[@]}"; do
  mkdir "$log_dir/env-$port"
  service_pids+=("$(ENV_PORT="$port" start_environment_service "$log_dir/env-$port")")
done

echo "MAEX service logs $log_dir"
wait_for_service "$POLICY_PORT" policy websocket
for port in "${environment_ports[@]}"; do
  wait_for_service "$port" environment websocket
done
wait_for_service "$OBSERVER_PORT" observer http
echo "Policy ws://127.0.0.1:$POLICY_PORT"
echo "Observer http://127.0.0.1:$OBSERVER_PORT/v1"
for port in "${environment_ports[@]}"; do
  echo "Environment ws://127.0.0.1:$port"
done

while sleep 5; do
  for pid in "${service_pids[@]}"; do
    if ! kill -0 "$pid" 2>/dev/null || [[ "$(ps -o stat= -p "$pid")" == Z* ]]; then
      echo "MAEX service process $pid stopped. See $log_dir." >&2
      exit 1
    fi
  done
done
