"""独立基线入口：组合原版 HarnessVLA 与 RoboMemArena，不接入 RoboAgent 决策。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from importlib.metadata import distributions
from pathlib import Path

from benchmark.robomemarena.baselines.harness_bridge import HarnessEnvBridge, HarnessPolicyBridge
from benchmark.robomemarena.client.artifacts import write_official_artifacts
from benchmark.robomemarena.client.remote_env import RemoteRoboMemArenaEnv
from benchmark.robomemarena.config import RoboMemArenaConfig


def file_manifest(root: Path) -> dict[str, str]:
    """记录源码或记忆文件的内容指纹，便于复现实验。"""
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file() and not any(
            part in {".git", ".venv", "__pycache__", ".cache", "logs"}
            for part in path.relative_to(root).parts
        )
    }


def _write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n")


def _smoke(toolkit) -> dict:
    # 只检查原工具的感知与物理接线，不伪造 Planner、SAM 或 VLA 成绩。
    results = []
    for name, arguments in (
        ("view_driver_state", {}),
        ("back_project", {"row": 512, "col": 512}),
        ("back_project", {"row": 512, "col": 512, "camera": "wrist"}),
        ("set_gripper", {"gripper": -1.0, "steps": 1}),
    ):
        result = toolkit.execute_tool(name, arguments)
        if result.result.get("error"):
            raise RuntimeError(f"Smoke failed at {name}: {result.result}")
        results.append({"tool": name, "result": result.result})
    return {"smoke_only": True, "tools": results}


def main() -> int:
    """运行一个独立 episode，并在规划结束后写入官方评分产物。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--harness-root", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--task", type=int, required=True)
    parser.add_argument("--seed", type=int, default=50)
    parser.add_argument("--max-steps", type=int, default=2500)
    parser.add_argument("--env-host", default="127.0.0.1")
    parser.add_argument("--env-port", type=int, default=8011)
    parser.add_argument("--policy-host", default="127.0.0.1")
    parser.add_argument("--policy-port", type=int, default=8000)
    parser.add_argument("--sam3-endpoint", help="Existing original SAM3 HTTP server URL")
    parser.add_argument("--resources-dir", type=Path, help="Original resources/libero directory")
    parser.add_argument("--planner", choices=("api", "claude_code", "codex"), default="api")
    parser.add_argument("--model")
    parser.add_argument("--base-url")
    parser.add_argument("--max-turns", type=int, default=100)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--planner-timeout-s", type=int, default=1200)
    parser.add_argument("--smoke", action="store_true", help="No Planner, VLA or SAM calls")
    args = parser.parse_args()
    config = RoboMemArenaConfig(task_id=args.task, seed=args.seed, max_steps=args.max_steps)
    harness_root, run_dir = args.harness_root.resolve(), args.run_dir.resolve()
    if not (harness_root / "robots/libero/toolkit.py").is_file():
        parser.error("--harness-root must point to the original HarnessVLA/RPent checkout")
    if run_dir.is_relative_to(harness_root):
        parser.error("--run-dir must be outside the original HarnessVLA checkout")
    if not args.smoke and (not args.model or not args.sam3_endpoint):
        parser.error("Evaluation requires --model and --sam3-endpoint")
    resources = args.resources_dir.resolve() if args.resources_dir else None
    if resources is not None and not resources.is_dir():
        parser.error("--resources-dir does not exist")

    source_manifest = file_manifest(harness_root)
    run_dir.mkdir(parents=True, exist_ok=False)
    workspace = run_dir / "workspace"
    workspace.mkdir()
    # 通过原框架已有的路径配置隔离写入；保持原提示词中的相对路径有效。
    shutil.copytree(harness_root / "robots/libero/guides", workspace / "robots/libero/guides")
    if resources is not None:
        shutil.copytree(resources, workspace / "resources/libero")
    os.environ["RPENT_REPO_ROOT"] = str(workspace)
    os.chdir(workspace)
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(harness_root))

    from robots.libero.prompt_bundle import system_prompt, user_prompt
    from robots.libero.toolkit import LiberoToolkit
    from rpent.dashboard.events import NullDashboardEventSink
    from rpent.envs.prompt_bundle import PromptBundle
    from rpent.planner.base import build_planner
    from rpent.utils.logging import init_output_dir
    from rpent.utils.resources import ensure_resources

    init_output_dir(run_dir / "artifacts")
    if not args.smoke and resources is None:
        ensure_resources("libero")
    if not args.smoke and not (workspace / "resources/libero/memory/MEMORY.md").is_file():
        raise RuntimeError(
            "Original HarnessVLA memory is unavailable; provide --resources-dir "
            "or restore access to RLinf/RPent-memory before evaluating the baseline"
        )
    memory_manifest = file_manifest(workspace / "resources")
    _write_json(run_dir / "source_manifest.json", source_manifest)
    _write_json(run_dir / "memory_manifest.json", memory_manifest)
    _write_json(run_dir / "environment.json", {
        "python": sys.version,
        "packages": {dist.metadata["Name"]: dist.version for dist in distributions()},
    })
    _write_json(run_dir / "run_config.json", {
        **vars(args), "baseline": "HarnessVLA-unmodified",
        "policy": "pi05_robomemarena", "action_chunk_size": 20,
        "observation_protocol": "RGB-D + calibration + robot state + object names",
        "memory_protocol": "original LIBERO resources; no RoboMemArena task memory constructed",
    })
    recipe_tag = f"robomemarena_t{args.task}_s{args.seed}"
    variables = {
        "suite": "robomemarena", "task": args.task, "seed": args.seed,
        "recipe_tag": recipe_tag, "output_dir": run_dir / "artifacts",
    }
    prompts = PromptBundle(system=system_prompt, user=user_prompt)
    system = prompts.render("system", variables=variables)
    user = prompts.render("user", variables=variables)
    (run_dir / "system_prompt.txt").write_text(system)
    (run_dir / "user_prompt.txt").write_text(user)

    dashboard = NullDashboardEventSink()
    policy, sam3, planner = None, None, None
    if not args.smoke:
        from rpent.utils.http_rpc import HttpRpcClient
        from rpent.utils.sam3_client import Sam3Client

        from benchmark.libero._vendor.openpi_client.websocket_policy import WebsocketPolicyClient

        sam3_rpc = HttpRpcClient(args.sam3_endpoint)
        sam3_rpc.call("healthz", timeout_s=10.0)
        sam3 = Sam3Client(sam3_rpc)
        planner = build_planner(
            args.planner, output_dir=run_dir / "artifacts", recipe_tag=recipe_tag,
            env_name="libero", base_url=args.base_url, model=args.model,
            max_tokens=args.max_tokens, planner_timeout_s=args.planner_timeout_s,
            dashboard_events=dashboard,
        )
        policy = HarnessPolicyBridge(WebsocketPolicyClient(args.policy_host, args.policy_port))

    remote = RemoteRoboMemArenaEnv(config, host=args.env_host, port=args.env_port)
    env = HarnessEnvBridge(remote)
    toolkit = None
    try:
        toolkit = LiberoToolkit(
            primitives_kwargs={"env": env, "model": policy, "sam3_client": sam3},
            dashboard_events=dashboard, video_path=str(run_dir / "artifacts/episode.mp4"),
        )
        if args.smoke:
            _write_json(run_dir / "smoke.json", _smoke(toolkit))
            return 0
        result = planner.solve(
            system_prompt=system, user_message=user, toolkit=toolkit, max_turns=args.max_turns,
        )
        _write_json(run_dir / "planner_result.json", {
            "finish": result.finish_result, "messages": result.messages,
            "stats": result.stats, "error": result.error,
        })
        toolkit.write_recipe(recipe_tag)
        if (result.finish_result or {}).get("status") == "success":
            env.settle()
        write_official_artifacts(
            run_dir, task_id=args.task, seed=args.seed, prompt=remote.task_description,
            evaluation=remote.metrics(), video_dir=run_dir / "artifacts",
        )
        return 1 if result.error else 0
    finally:
        try:
            if toolkit is not None:
                toolkit.close()
        finally:
            remote.close()
            if policy is not None:
                policy.policy.close()
        _write_json(run_dir / "source_unchanged.json", {
            "unchanged": file_manifest(harness_root) == source_manifest,
        })


if __name__ == "__main__":
    raise SystemExit(main())
