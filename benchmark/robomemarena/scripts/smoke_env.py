"""Exercise reset, observe, one dummy step, metrics, and close over RPC."""

from __future__ import annotations

import argparse
import json

from benchmark.robomemarena.config import RoboMemArenaConfig
from benchmark.robomemarena.preprocessing import ROBOMEMARENA_DUMMY_ACTION
from benchmark.robomemarena.client.remote_env import RemoteRoboMemArenaEnv


def main() -> None:
    """Run the environment-only smoke test without loading any policy."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--task-id", type=int, default=1)
    parser.add_argument("--seed", type=int, default=100)
    args = parser.parse_args()

    env = RemoteRoboMemArenaEnv(
        RoboMemArenaConfig(task_id=args.task_id, seed=args.seed),
        host=args.host,
        port=args.port,
    )
    try:
        reset_observation = env.reset()
        observed = env.observe()
        step = env.step(ROBOMEMARENA_DUMMY_ACTION)
        metrics = env.metrics()
        output = {
            "task_description": env.task_description,
            "reset_keys": sorted(reset_observation),
            "observe_keys": sorted(observed),
            "step_done": step.done,
            "step_success": step.success,
            "metrics": metrics,
        }
        print(json.dumps(output, ensure_ascii=False, indent=2))
    finally:
        env.close()


if __name__ == "__main__":
    main()
