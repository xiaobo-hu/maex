#!/usr/bin/env bash
# 只向官方 Agent 暴露本次实验与指定记忆，防止 Bash/Read 读入其他方法的历史记录。
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: bash $0 RUN_ROOT MEMORY_DIR [official planner/task/explore flags]" >&2
  exit 2
fi
project_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
run_root=$(realpath -m "$1")
memory_root=$(realpath -m "$2")
shift 2
official_root="$project_root/benchmark/.external/RPent-official"
client_root="$project_root/benchmark/robomemarena/logs/harness-venv"
command -v bwrap >/dev/null
mkdir -p "$run_root" "$memory_root"

# 主机系统库只读；项目目录重新按最小运行依赖挂载。服务通过现有网络端口连接。
mounts=(--ro-bind / / --dev /dev --tmpfs "$project_root" --tmpfs /tmp
  --unshare-pid --proc /proc --die-with-parent)
for relative in \
  agent/roboagent/prompts/robomemarena.py \
  agent/roboagent/prompts/fullvlm_v2_26_memory_tasks.json \
  benchmark/__init__.py benchmark/libero/rpc.py \
  benchmark/libero/_vendor \
  benchmark/robomemarena/config.py benchmark/robomemarena/environment.py \
  benchmark/robomemarena/preprocessing.py benchmark/robomemarena/client/__init__.py \
  benchmark/robomemarena/client/artifacts.py benchmark/robomemarena/client/remote_env.py \
  benchmark/robomemarena/baselines/__init__.py \
  benchmark/robomemarena/baselines/harness_bridge.py \
  benchmark/robomemarena/baselines/harness_eval.py \
  benchmark/robomemarena/baselines/harness_prompts.py \
  benchmark/robomemarena/baselines/harness_official.py; do
  mounts+=(--ro-bind "$project_root/$relative" "$project_root/$relative")
done
# 这些包的公开入口会加载 RoboAgent 策略；隔离进程只需要下面已挂载的子模块。
# 使用空包入口，既不暴露策略源码，也不修改其他基线的公共 API。
for relative in agent/roboagent/__init__.py agent/roboagent/prompts/__init__.py \
  benchmark/libero/__init__.py benchmark/robomemarena/__init__.py; do
  mounts+=(--ro-bind /dev/null "$project_root/$relative")
done
memory_mount=--ro-bind
for argument in "$@"; do
  if [[ "$argument" == --explore ]]; then
    memory_mount=--bind
  fi
done
mounts+=(--ro-bind "$official_root" "$official_root"
  --ro-bind "$client_root" "$client_root"
  --bind "$run_root" "$run_root" "$memory_mount" "$memory_root" "$memory_root")
# SDK 的本地运行缓存不与用户已有 Claude Code 会话共享。
if [[ -d "$HOME/.claude" ]]; then
  mounts+=(--tmpfs "$HOME/.claude")
fi
exec bwrap "${mounts[@]}" --chdir "$project_root" -- \
  "$client_root/bin/python" -B -m benchmark.robomemarena.baselines.harness_official \
  "$@" --harness-root "$official_root" --run-dir "$run_root/run" --memory-dir "$memory_root"
