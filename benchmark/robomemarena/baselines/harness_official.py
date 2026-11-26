"""基线组合层：用 RoboMemArena 服务运行官方 RPent 探索、记忆合并和评测入口。"""

from __future__ import annotations

import argparse
from dataclasses import replace
import os
from pathlib import Path
import shutil
import subprocess
import sys

from benchmark.robomemarena.baselines.harness_bridge import OfficialHarnessEnvBridge
from benchmark.robomemarena.baselines.harness_eval import _write_json, file_manifest
from benchmark.robomemarena.baselines.harness_prompts import PROMPT_ADAPTATION, make_prompt_bundle
from benchmark.robomemarena.client.artifacts import write_official_artifacts
from benchmark.robomemarena.client.remote_env import RemoteRoboMemArenaEnv
from benchmark.robomemarena.config import RoboMemArenaConfig
from roboagent.prompts.robomemarena import task_primitive_reference



class RecordedHarnessEnv(OfficialHarnessEnvBridge):
    """在每次物理 reset 前归档评分；评分文件不进入 Planner 的观测。"""

    def __init__(self, remote, *, config, episode_dir: Path, allow_reset: bool):
        super().__init__(remote, allow_reset=allow_reset)
        self.config = config
        self.episode_dir = episode_dir
        self.episode_count = 0

    def reset(self):
        """保存上一 episode，再恢复相同任务、相同 seed 的初始场景。"""
        if self._reset_done and self._allow_reset:
            self.archive_episode()
        result = super().reset()
        self.episode_count += 1
        return result

    def get_task_language(self):
        """提供与 RoboAgent/HiMe 相同的静态 VLA 参考，不参与运行时决策。"""
        instruction = super().get_task_language()
        reference = task_primitive_reference(instruction)
        return instruction + "\n\n" + reference if reference else instruction

    def archive_episode(self) -> None:
        """通过 benchmark 的独立评分通道记录实际发生的 episode。"""
        if not self.episode_count:
            return
        output = self.episode_dir / f"episode_{self.episode_count:03d}"
        write_official_artifacts(
            output, task_id=self.config.task_id, seed=self.config.seed,
            prompt=self.remote.task_description, evaluation=self.remote.metrics(),
            video_dir=output,
        )


def main() -> int:
    """替换 RobotSpec 的环境组合钩子，执行未修改的官方 CLI 主循环。"""
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--harness-root", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--memory-dir", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=2500)
    parser.add_argument("--env-host", default="127.0.0.1")
    parser.add_argument("--env-port", type=int, default=8011)
    parser.add_argument("--policy-host", default="127.0.0.1")
    parser.add_argument("--policy-port", type=int, default=8000)
    options, forwarded = parser.parse_known_args()
    source = options.harness_root.resolve()
    run_dir = options.run_dir.resolve()
    memory_dir = options.memory_dir.resolve()
    if not (source / "rpent/memory/manager.py").is_file():
        parser.error("--harness-root requires official RPent with exploration support")
    if run_dir.is_relative_to(source) or memory_dir.is_relative_to(source):
        parser.error("run and memory directories must be outside the official checkout")
    if memory_dir.is_relative_to(run_dir):
        parser.error("--memory-dir must be persistent and outside --run-dir")

    manifest = file_manifest(source)
    run_dir.mkdir(parents=True, exist_ok=False)
    workspace = run_dir / "workspace"
    # 原提示中的相对 guide 路径在隔离工作目录中保持有效。
    shutil.copytree(source / "robots/libero/guides", workspace / "robots/libero/guides")
    os.environ["RPENT_REPO_ROOT"] = str(workspace)
    os.chdir(workspace)
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(source))

    import robots.libero as libero
    from rpent.cli.main import main as official_main
    from rpent.robots.components.sam3_client import Sam3Client
    from rpent.utils.rpc.http_rpc import HttpRpcClient
    from rpent.robots.components.pi05_vla_client import Pi05VLAClient

    original_spec = libero.get_robot_spec()
    prompts = make_prompt_bundle(original_spec.prompts)
    env, policy, remote = None, None, None
    memory_before = file_manifest(memory_dir)
    _write_json(run_dir / "source_manifest.json", manifest)
    _write_json(run_dir / "memory_before.json", memory_before)

    def parse_config(args):
        # 保存实际提示及官方原文，便于审查环境适配的差异。
        config = original_spec.parse_config(args)
        config.prompt_vars["attempts_per_session"] = args.explore_attempts_per_session
        if not args.sam3_endpoint:
            raise ValueError("--sam3-endpoint is required")
        if args.explore:
            for tier in ("global", "suite", "task_only"):
                (memory_dir / tier).mkdir(parents=True, exist_ok=True)
        _write_json(run_dir / "run_config.json", {
            **vars(args), "baseline": "HarnessVLA-official-RoboMemArena",
            "official_commit": subprocess.check_output(
                ["git", "-C", str(source), "rev-parse", "HEAD"], text=True,
            ).strip(),
            "environment": "RoboMemArena", "policy": "RLinf/RLinf-Pi05-LIBERO-130-fullshot-SFT",
            "action_chunk_limit": 5,
            "policy_preprocessing": "official LIBERO double-axis image flip and axis-angle state",
            "env_endpoint": f"ws://{options.env_host}:{options.env_port}",
            "policy_endpoint": f"http://{options.policy_host}:{options.policy_port}",
            "memory_workflow": "official CLI, Toolkit and MemoryManager; " + PROMPT_ADAPTATION,
            "prompt_adaptation": PROMPT_ADAPTATION,
            "task_reference": "shared task_primitive_reference; same fields as RoboAgent and HiMe",
        })
        variables = {**config.prompt_vars, "output_dir": config.output_dir}
        for kind in ("system", "user"):
            (run_dir / f"{kind}_prompt.txt").write_text(
                prompts.render(kind, variables=variables)
            )
            (run_dir / f"{kind}_prompt.official.txt").write_text(
                original_spec.prompts.render(kind, variables=variables)
            )
        return config

    def init_runtime(args, output_dir, dashboard_events, components):
        nonlocal env, policy, remote
        config = RoboMemArenaConfig(
            task_id=args.task, seed=args.seed, max_steps=args.max_episode_steps,
        )
        rpc = HttpRpcClient(args.sam3_endpoint)
        rpc.call("healthz", timeout_s=10)
        policy_rpc = HttpRpcClient(f"http://{options.policy_host}:{options.policy_port}")
        health = policy_rpc.call("healthz", timeout_s=10)
        policy = Pi05VLAClient(policy_rpc, embodiment="libero")
        _write_json(run_dir / "policy_metadata.json", {
            "health": health, "client": "official Pi05VLAClient", "embodiment": "libero",
            "provenance_note": "Checkpoint path and package versions are recorded by the service launcher; healthz does not attest weights.",
        })
        remote = RemoteRoboMemArenaEnv(config, host=options.env_host, port=options.env_port)
        env = RecordedHarnessEnv(
            remote, config=config, episode_dir=run_dir / "episodes", allow_reset=args.explore,
        )
        return [], {"env": env, "model": policy, "sam3_client": Sam3Client(rpc)}

    # 使用官方扩展契约注入外部服务；不复制或修改 session/reset/merge 的实现。
    libero.get_robot_spec = lambda: replace(
        original_spec, parse_config=parse_config, init_runtime=init_runtime, prompts=prompts,
    )
    original_argv = sys.argv
    sys.argv = [
        "harness_official", "--explore-sessions", "1", *forwarded,
        "--robot", "libero", "--suite", "robomemarena",
        "--max-episode-steps", str(options.max_steps),
        "--memory-profile", "local", "--memory-dir", str(memory_dir),
        "--output-dir", str(run_dir / "artifacts"),
    ]
    exit_code = None
    try:
        exit_code = official_main()
        return exit_code
    finally:
        try:
            if env is not None:
                env.archive_episode()
        finally:
            if remote is not None:
                remote.close()
            sys.argv = original_argv
            libero.get_robot_spec = lambda: original_spec
            memory_after = file_manifest(memory_dir)
            _write_json(run_dir / "memory_after.json", memory_after)
            _write_json(run_dir / "completion.json", {
                "exit_code": exit_code,
                "physical_episodes": env.episode_count if env is not None else 0,
                "source_unchanged": file_manifest(source) == manifest,
                "memory_changed": memory_after != memory_before,
            })


if __name__ == "__main__":
    raise SystemExit(main())
