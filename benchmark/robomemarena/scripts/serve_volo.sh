#!/usr/bin/env bash
# 基线服务层：启动原生 VoLo 代理，模型凭据仅通过进程环境传递。
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$PROJECT_ROOT"
set -a
source benchmark/robomemarena/configs/volo.env
set +a

# 只解析配置中的 LLM 字段，不执行密钥文件中的 shell 内容。
exec .venv/bin/python - "$@" <<'PY'
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

root = Path.cwd()
env_file = Path(os.environ.get("VOLO_ENV_FILE", root / ".env"))
if env_file.is_file():
    for line in env_file.read_text().splitlines():
        key, separator, value = line.strip().partition("=")
        if separator and key in {"LLM_API_KEY", "LLM_BASE_URL", "LLM_MODEL_NAME"}:
            os.environ.setdefault(key, value.strip().strip("\"'"))
for target, source in (("VLM_API_KEY", "LLM_API_KEY"),
                       ("VLM_BASE_URL", "LLM_BASE_URL"),
                       ("VLM_MODEL", "LLM_MODEL_NAME")):
    if not os.environ.get(target):
        os.environ[target] = os.environ.get(source, "")
    if not os.environ[target]:
        raise SystemExit(f"Missing {target} or {source} in {env_file}")
upstream = root / "benchmark/.external/VoLoAgent"
executable = upstream / ".venv/bin/vlm-orchestrator"
if not executable.is_file():
    raise SystemExit("Install VoLoAgent first; see benchmark/robomemarena/baselines/VOLO.md")
log_dir = Path(os.environ.get("VOLO_LOG_DIR", "benchmark/robomemarena/logs/volo/proxy"))
log_dir.mkdir(parents=True, exist_ok=True)
grasp_enabled = os.environ.get("VOLO_GRASP_ENABLED", "0") == "1"
args = [str(executable), "--env", "libero", "--host", "127.0.0.1",
        "--port", os.environ.get("VOLO_PORT", "8040"),
        "--vla-host", os.environ.get("VOLO_VLA_HOST", "127.0.0.1"),
        "--vla-port", os.environ.get("VOLO_VLA_PORT", "8000"),
        "--mode", "subgoal", "--failure-monitor", "vlm", "--recovery-mode",
        "replan_grasp" if grasp_enabled else "replan",
        # LIBERO 抓取分支使用 OSC 闭环；不进入 cuRobo 关节轨迹路径。
        "--motion-planner", "linear", "--grasp-seg-mode", "sam3",
        "--check-interval", os.environ.get("VOLO_CHECK_INTERVAL", "40"),
        "--vlm-base-url", os.environ["VLM_BASE_URL"], "--vlm-model", os.environ["VLM_MODEL"],
        "--log-dir", str(log_dir), *sys.argv[1:]]
revision = subprocess.check_output(["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True).strip()
upstream_diff = subprocess.check_output(["git", "-C", str(upstream), "diff", "HEAD", "--"])
(log_dir / "upstream.patch").write_bytes(upstream_diff)
(log_dir / "launch_config.json").write_text(json.dumps({
    "upstream_revision": revision, "argv": args, "external_skill": None,
    "upstream_patch_sha256": hashlib.sha256(upstream_diff).hexdigest(),
    "observer": "native VoLo VLM monitor", "vlm_model": os.environ["VLM_MODEL"],
    "grasp_enabled": grasp_enabled, "place_execution": "VLA",
    "grasp_server_host": os.environ.get("GRASP_SERVER_HOST", "127.0.0.1"),
    "grasp_server_port": int(os.environ.get("GRASP_SERVER_PORT", "8043")),
}, indent=2) + "\n")
os.execv(str(executable), args)
PY
