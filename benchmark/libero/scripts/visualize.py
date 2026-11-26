"""Launch one LIBERO task in the interactive robosuite/MuJoCo viewer."""

from __future__ import annotations

import argparse
import time
from pathlib import Path


def main() -> None:
    """Open a viewer and execute zero-motion actions until the step limit."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", default="libero_spatial")
    parser.add_argument("--task", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=300)
    args = parser.parse_args()

    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import TASK_MAPPING
    from libero.libero.envs.bddl_utils import get_problem_info
    from robosuite.controllers import load_controller_config

    suites = benchmark.get_benchmark_dict()
    if args.suite not in suites:
        raise ValueError(f"Unknown suite {args.suite!r}; choose from {sorted(suites)}")
    suite = suites[args.suite]()
    if not 0 <= args.task < suite.n_tasks:
        raise ValueError(f"task must be in [0, {suite.n_tasks})")

    task = suite.get_task(args.task)
    bddl_file = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    problem_name = get_problem_info(bddl_file)["problem_name"]
    controller_config = load_controller_config(default_controller="OSC_POSE")
    env_kwargs: dict[str, object] = {
        "bddl_file_name": bddl_file,
        "robots": ["Panda"],
        "controller_configs": controller_config,
        "has_renderer": True,
        "has_offscreen_renderer": False,
        "render_camera": "frontview",
        "use_camera_obs": False,
        "ignore_done": True,
        "control_freq": 20,
    }
    if "TwoArm" in problem_name:
        env_kwargs["env_configuration"] = "single-arm-opposed"

    env = TASK_MAPPING[problem_name](**env_kwargs)
    try:
        env.seed(args.seed)
        env.reset()
        env.render()
        print(f"task={task.language!r}")
        print("Close the MuJoCo window or wait for the step limit to exit.")
        action = [0.0] * 6 + [-1.0]
        for step in range(args.steps):
            _observation, _reward, done, _info = env.step(action)
            env.render()
            if step % 20 == 0:
                print(f"step={step + 1}/{args.steps} done={done}")
            if done:
                break
            time.sleep(1 / 20)
    finally:
        env.close()


if __name__ == "__main__":
    main()
