"""验证 RoboMemArena 批跑入口在必需服务不可用时停止，在重启成功时续跑。"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.mark.parametrize("service", ["policy", "environment", "observer"])
@pytest.mark.parametrize("failure", ["startup", "restart", "recovered"])
def test_batch_requires_healthy_services(tmp_path, service, failure) -> None:
    scripts = tmp_path / "benchmark/robomemarena/scripts"
    scripts.mkdir(parents=True)
    original = Path(__file__).resolve().parents[1] / "scripts/run_batch_persistent.sh"
    script = scripts / original.name
    shutil.copyfile(original, script)
    config = tmp_path / "config.env"
    config.write_text(
        "TASK_ID=1\nSEED=52\nPLANNER=faux\nMODEL=faux\n"
        "POLICY_PORT=8000\nENV_PORT=8001\nOBSERVER_PORT=8002\n"
    )
    (scripts / "lib.sh").write_text('''
load_planner_llm_env() { :; }
start_policy_service() { :; }
start_environment_service() { :; }
start_observer_service() { :; }
wait_for_service() {
  local count=0 file="$BATCH_DIR/$2.checks"
  [[ ! -f "$file" ]] || read -r count < "$file"
  count=$((count + 1))
  echo "$count" > "$file"
  if [[ "$2" == "$TEST_SERVICE" ]]; then
    case "$TEST_FAILURE" in
      startup) return 1 ;;
      restart) (( count < 3 )) || return 1 ;;
      recovered) (( count != 3 )) || return 1 ;;
    esac
  fi
  return 0
}
run_roboagent_eval() {
  echo "$TASK_ID" >> "$BATCH_DIR/executed"
  touch "$1/trace.json" "$1/evaluation.json" "$1/summary.json" "$1/run_config.json"
}
''')
    result = subprocess.run(
        ["bash", str(script), str(config), "1", "2", "test-batch"],
        env={**os.environ, "TEST_SERVICE": service, "TEST_FAILURE": failure},
        capture_output=True, text=True, timeout=5, check=False,
    )
    batch = tmp_path / "benchmark/robomemarena/logs/batches/test-batch"
    executed_file = batch / "executed"
    executed = executed_file.read_text().splitlines() if executed_file.exists() else []
    if failure == "startup":
        assert result.returncode == 1
        assert executed == []
    elif failure == "restart":
        assert result.returncode == 1
        assert executed == ["1"]
        assert "task 2 NOT STARTED" in (batch / "status.log").read_text()
    else:
        assert result.returncode == 0
        assert executed == ["1", "2"]
