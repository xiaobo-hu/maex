"""Run the frozen official-style pi05_libero rollout without importing OpenPI client code."""

from __future__ import annotations

import argparse
import collections
import logging
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

from benchmark.libero._vendor.openpi_client import WebsocketPolicyClient
from benchmark.libero.preprocessing import (
    DEFAULT_MAX_STEPS,
    LIBERO_DUMMY_ACTION,
    LIBERO_ENV_RESOLUTION,
    build_policy_input,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--suite", default="libero_spatial")
    parser.add_argument("--task", type=int)
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--replan-steps", type=int, default=5)
    parser.add_argument("--resize-size", type=int, default=224)
    parser.add_argument("--video-out-path", default="/tmp/libero_vla")
    return parser.parse_args()


def main() -> None:
    """Evaluate the selected LIBERO suite with the upstream rollout semantics."""
    args = _parse_args()
    if args.suite not in DEFAULT_MAX_STEPS:
        raise ValueError(f"Unknown LIBERO suite: {args.suite}")
    if args.trials < 1 or args.replan_steps < 1:
        raise ValueError("trials and replan-steps must be at least 1")

    from libero.libero import benchmark

    np.random.seed(args.seed)
    suite = benchmark.get_benchmark_dict()[args.suite]()
    task_ids = range(suite.n_tasks) if args.task is None else [args.task]
    video_dir = Path(args.video_out_path)
    video_dir.mkdir(parents=True, exist_ok=True)
    client = WebsocketPolicyClient(args.host, args.port)
    total_episodes = 0
    total_successes = 0
    try:
        for task_id in task_ids:
            if not 0 <= task_id < suite.n_tasks:
                raise ValueError(f"task must be in [0, {suite.n_tasks})")
            task = suite.get_task(task_id)
            initial_states = suite.get_task_init_states(task_id)
            if args.trials > len(initial_states):
                raise ValueError(f"trials cannot exceed {len(initial_states)} initial states")
            env = _create_env(task, args.seed)
            try:
                for trial in range(args.trials):
                    success, frames = _run_episode(
                        env,
                        initial_states[trial],
                        str(task.language),
                        client,
                        args,
                    )
                    total_episodes += 1
                    total_successes += int(success)
                    suffix = "success" if success else "failure"
                    imageio.mimwrite(
                        video_dir / f"task_{task_id}_trial_{trial}_{suffix}.mp4",
                        frames,
                        fps=10,
                    )
                    logging.info(
                        "task=%s trial=%s success=%s total=%s/%s",
                        task_id,
                        trial,
                        success,
                        total_successes,
                        total_episodes,
                    )
            finally:
                env.close()
    finally:
        client.close()
    logging.info("Total success rate: %.1f%%", 100 * total_successes / total_episodes)


def _create_env(task, seed: int):
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    bddl_file = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(
        bddl_file_name=bddl_file,
        camera_heights=LIBERO_ENV_RESOLUTION,
        camera_widths=LIBERO_ENV_RESOLUTION,
    )
    env.seed(seed)
    return env


def _run_episode(env, initial_state, instruction: str, client, args) -> tuple[bool, list[np.ndarray]]:
    env.reset()
    observation = env.set_init_state(initial_state)
    action_plan = collections.deque()
    frames: list[np.ndarray] = []
    done = False
    for step in range(DEFAULT_MAX_STEPS[args.suite] + args.warmup_steps):
        if step < args.warmup_steps:
            observation, _reward, done, _info = env.step(LIBERO_DUMMY_ACTION)
            if done:
                break
            continue
        payload = build_policy_input(observation, instruction, resize_size=args.resize_size)
        frames.append(np.asarray(payload["observation/image"]))
        if not action_plan:
            actions = np.asarray(client.infer(payload).get("actions"))
            if actions.ndim != 2 or len(actions) < args.replan_steps:
                raise ValueError("OpenPI returned an invalid or too-short action chunk")
            action_plan.extend(actions[: args.replan_steps])
        action = np.asarray(action_plan.popleft())
        if action.shape != (7,) or not np.isfinite(action).all():
            raise ValueError("OpenPI actions must be finite 7-D vectors")
        observation, _reward, done, _info = env.step(action.tolist())
        if done:
            break
    return bool(done), frames


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
