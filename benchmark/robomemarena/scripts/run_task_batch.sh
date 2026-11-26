#!/usr/bin/env bash
# 对 [START, END] 内每个 task_id 调用一次 run_eval.sh(默认 qwen_recovery.env 配置),
# 逐任务串行执行、单任务失败不影响后续,并把逐任务状态与产物目录追加到
# benchmark/robomemarena/logs/batch/current.status。
#
# 用法: bash run_task_batch.sh [CONFIG_FILE] [START] [END]
# 示例: bash run_task_batch.sh                    # qwen 配置, task 1..26
#       bash run_task_batch.sh cfg.env 10 15      # 从 task 10 续跑到 15

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
CONFIG_FILE="${1:-$PROJECT_ROOT/benchmark/robomemarena/configs/qwen_recovery.env}"
START="${2:-1}"
END="${3:-26}"

if [[ ! -f "$CONFIG_FILE" ]]; then
  echo "Evaluation config not found: $CONFIG_FILE" >&2
  exit 2
fi

BATCH_DIR="$PROJECT_ROOT/benchmark/robomemarena/logs/batch"
mkdir -p "$BATCH_DIR"
STATUS_FILE="$BATCH_DIR/current.status"

# 结束上个任务后清理可能残留的评测服务,避免端口被占导致下一个任务 180s 等待超时。
kill_stale_services() {
  for port in 8000 8001 8002; do
    for pid in $(ss -ltnpH "sport = :$port" 2>/dev/null | grep -oP 'pid=\K[0-9]+' | sort -u); do
      cmd=$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null || true)
      case "$cmd" in
        *serve_policy.py*|*serve_env*|*vllm*)
          echo "[preflight] kill stale pid $pid on :$port (cmd: ${cmd:0:80})" >>"$STATUS_FILE"
          kill -TERM "$pid" 2>/dev/null || true
          ;;
      esac
    done
  done
}

wait_ports_free() {
  for _ in $(seq 1 10); do
    local busy=""
    for port in 8000 8001 8002; do
      local pids
      pids=$(ss -ltnpH "sport = :$port" 2>/dev/null | grep -oP 'pid=\K[0-9]+' | sort -u)
      [[ -n "$pids" ]] && busy="$busy $port:$pids"
    done
    [[ -z "$busy" ]] && return 0
    sleep 2
  done
  echo "[preflight] ports still busy:$busy; force kill" >>"$STATUS_FILE"
  for port in 8000 8001 8002; do
    fuser -k -TERM "$port"/tcp 2>/dev/null || true
  done
  sleep 2
}

echo "batch start $(date -Is) config=$CONFIG_FILE tasks=$START..$END" >>"$STATUS_FILE"
for ((task = START; task <= END; task++)); do
  kill_stale_services
  wait_ports_free
  cfg="$BATCH_DIR/task${task}.env"
  sed "s/^TASK_ID=.*/TASK_ID=${task}/" "$CONFIG_FILE" >"$cfg"
  begin=$(date +%s)
  echo "[$(date '+%F %T')] task $task START cfg=$cfg" | tee -a "$STATUS_FILE"
  # run_eval.sh 内部把 roboagent_eval 输出 tee 进该任务 RUN_DIR/run.log,
  # 其自身 stdout 只有 "Evaluation artifacts: <RUN_DIR>" 一行与错误信息。
  bash "$SCRIPT_DIR/run_eval.sh" "$cfg" >>"$STATUS_FILE" 2>&1
  rc=$?
  end=$(date +%s)
  dur=$(((end - begin) / 60))
  echo "[$(date '+%F %T')] task $task EXIT=$rc dur=${dur}min" >>"$STATUS_FILE"
done
kill_stale_services
echo "batch done $(date -Is)" >>"$STATUS_FILE"
