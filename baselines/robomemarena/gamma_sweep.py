"""评测编排层，管理 GAMMA 的服务、回合进度与 RoboMemArena 结果。"""

import argparse
import csv
import fcntl
import hashlib
import io
import json
import os
import signal
import sqlite3
import subprocess
import sys
from collections import defaultdict
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean
from time import monotonic, sleep
from urllib.error import URLError
from urllib.request import ProxyHandler, build_opener

from gamma_eval import lane_targets, run_lane

DIRECTORY = Path(__file__).resolve().parent
PROJECT = DIRECTORY.parents[1]
RUNNER = DIRECTORY / "run.sh"
BENCHMARK_REVISION = "d09f63b671f362985e1e0d038d2014c2d2519488"
BENCHMARK_REPOSITORY = "https://github.com/OpenHelix-Team/RoboMemArena.git"
AGENT_GPUS = (6, 6, 7, 7, 7, 7)
VLA_PORT, QWEN_PORT, FIRST_AGENT_PORT = 8340, 8342, 8350
PROTOCOL = {
    "tasks": list(range(1, 27)), "seeds": list(range(50, 100)), "chunk_steps": 10,
    "warmup_steps": 10, "max_policy_steps": 2500, "post_goal_steps": 200,
    "extra_pour_monitor_steps": 30, "resize": 256, "settle_ticks": 2,
}
EPISODE_FIELDS = (
    "task", "seed", "success", "stage_success_pct", "simulation_steps", "policy_steps",
    "wall_time_s", "policy_calls", "failure_reason", "stages",
)


class ModelServiceError(RuntimeError):
    """模型服务退出或在启动时未就绪。"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, help="只评测这一布局 seed，每任务跑一回合")
    parser.add_argument("--state", type=Path)
    parser.add_argument("--result", type=Path)
    parser.add_argument("--lane", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    state_suffix = f"-seed{args.seed}" if args.seed is not None else ""
    result_suffix = f"_seed{args.seed}" if args.seed is not None else ""
    args.state = args.state or PROJECT / f"baselines/.runtime/gamma-robomemarena{state_suffix}"
    args.result = args.result or PROJECT / f"result/gamma_robomemarena{result_suffix}"
    if args.lane is not None:
        run_lane(args.state, args.lane, len(AGENT_GPUS))
        return

    protocol = PROTOCOL if args.seed is None else {**PROTOCOL, "seeds": [args.seed]}
    args.state.mkdir(parents=True, exist_ok=True)
    args.result.mkdir(parents=True, exist_ok=True)
    with (args.state / "sweep.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        config = prepare_run(args.state, args.result, protocol)
        with closing(sqlite3.connect(args.state / "episodes.sqlite3", timeout=30)) as database:
            database.execute("CREATE TABLE IF NOT EXISTS episodes (task INTEGER, seed INTEGER, data TEXT, PRIMARY KEY (task, seed))")
        expected = len(protocol["tasks"]) * len(protocol["seeds"])

        def handle_stop(_signum, _frame):
            raise InterruptedError("评测进程收到停止信号")

        signal.signal(signal.SIGTERM, handle_stop)
        try:
            failures = 0
            while (completed := write_results(args.state, args.result, config, "运行中")) < expected:
                try:
                    completed = run_sweep(args.state, args.result, config)
                except ModelServiceError as error:
                    saved = write_results(args.state, args.result, config, "模型服务重启中")
                    failures = 1 if saved > completed else failures + 1
                    print(f"Model service failure ({failures}/3): {error}", flush=True)
                    if failures >= 3:
                        raise
                    sleep(10)
                else:
                    break
            if completed != expected:
                raise RuntimeError(f"Recorded {completed} of {expected} episodes")
        except (Exception, KeyboardInterrupt) as error:
            write_results(args.state, args.result, config, f"已暂停，{str(error) or type(error).__name__}")
            raise
        write_results(args.state, args.result, config, "已完成")


def prepare_run(state: Path, result: Path, protocol: dict) -> dict:
    """固定配置并核对续跑时的模型和评测源码版本。"""
    benchmark = state / "RoboMemArena"
    if not benchmark.exists():
        subprocess.run(["git", "clone", "--quiet", "--shared", "--no-checkout", os.environ["ROBOMEMARENA_ROOT"], benchmark], check=True)
        subprocess.run(["git", "-C", benchmark, "update-ref", "refs/heads/installed-rma", "HEAD"], check=True)
        subprocess.run([
            "git", "-C", benchmark, "-c", "protocol.version=2", "fetch", "--quiet", "--depth=2",
            BENCHMARK_REPOSITORY, BENCHMARK_REVISION,
        ], check=True)
        subprocess.run(["git", "-C", benchmark, "checkout", "--quiet", "--detach", BENCHMARK_REVISION], check=True)
    bddls = benchmark / "evaluation_benchmark/bddl"
    task_names = {str(task): next(bddls.glob(f"{task}_*.bddl")).stem.split("_", 1)[1] for task in protocol["tasks"]}
    versions = {}
    for name, root in (
        ("gamma", PROJECT / "baselines/GAMMA"),
        ("robomemarena", benchmark),
    ):
        commit = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
        patch = subprocess.check_output(["git", "-C", str(root), "diff", "HEAD"])
        if name == "robomemarena" and (commit != BENCHMARK_REVISION or patch):
            raise RuntimeError(f"RoboMemArena 评测检出需使用干净的 {BENCHMARK_REVISION}，当前路径 {root}")
        versions[name] = {
            "commit": commit, "patch_sha256": hashlib.sha256(patch).hexdigest(),
        }
    config = {
        "protocol": protocol, "source": versions, "task_names": task_names, "agent_and_env_gpus": AGENT_GPUS,
        "vla_gpu": 6, "qwen_gpus": [4, 5],
        "writer": "models/Qwen3.8-27B", "detector": "models/sam3/model.safetensors",
        "executor": "models/openpi/checkpoints/pi05_robomemarena",
        "vla_labels": "baselines/GAMMA/rma/eval_stack/rma_their_labels.json",
    }
    path = state / "config.json"
    if path.exists():
        previous = json.loads(path.read_text())
        if {key: previous[key] for key in config} != json.loads(json.dumps(config)):
            raise RuntimeError("续跑配置与已有结果不一致")
        config = previous
    else:
        config["started_at_utc"] = datetime.now(UTC).isoformat()
        atomic_write(path, json.dumps(config, indent=2) + "\n")
    atomic_write(result / "run_config.json", json.dumps(config, indent=2) + "\n")
    return config


def run_sweep(state: Path, result: Path, config: dict) -> int:
    """启动独立服务，按种子分片评测并定期更新结果。"""
    benchmark = state / "RoboMemArena"
    libero_config = state / "libero_config"
    libero_config.mkdir(exist_ok=True)
    original_config = (Path(os.environ["LIBERO_CONFIG_PATH"]) / "config.yaml").read_text()
    atomic_write(libero_config / "config.yaml", original_config.replace(os.environ["ROBOMEMARENA_ROOT"], str(benchmark)))
    env = {
        **os.environ, "BASELINE_VLA_PORT": str(VLA_PORT), "BASELINE_QWEN_PORT": str(QWEN_PORT),
        "BASELINE_CHECKPOINT": str(PROJECT / config["executor"]),
        "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1", "PYTHONDONTWRITEBYTECODE": "1",
    }
    for key in ("RMA_SUBGOAL_LOG", "RMA_GAMMA_LOG", "WAM_TRACE"):
        env.pop(key, None)
    services = {}
    workers = {}
    retries = defaultdict(int)
    saved_before_launch = {}
    stopped = set()
    try:
        for component, gpu, port, health in (
            ("vla", "6", VLA_PORT, "/healthz"),
            ("qwen", "4,5", QWEN_PORT, "/health"),
            *(("gamma-agent", str(gpu), FIRST_AGENT_PORT + lane, "/health") for lane, gpu in enumerate(AGENT_GPUS)),
        ):
            services[port] = (
                start_process(["bash", str(RUNNER), component], state / f"service-{port}.log", {
                    **env, "BASELINE_GPU": gpu, "BASELINE_GAMMA_PORT": str(port),
                    "WAM_A1_OFF": "0", "WAM_HARNESS_OFF": "0", "WAM_VERIFY_VISION": "1",
                    "WAM_SAM_OFF": "0", "AUTO_GRACE": "2", "DET_THR": "0.15", "REFUTE_TICKS": "3",
                }),
                health,
            )
        wait_for_models(services)
        print("GAMMA models ready", flush=True)
        while True:
            for port, (process, _) in services.items():
                if process.poll() is not None:
                    raise ModelServiceError(f"model on port {port} exited with {process.returncode}")
            for lane, gpu in enumerate(AGENT_GPUS):
                if lane in stopped:
                    continue
                previous = workers.get(lane)
                if previous is not None and previous.poll() in (None, 0):
                    continue
                with closing(sqlite3.connect(state / "episodes.sqlite3", timeout=30)) as database:
                    completed_targets = set(database.execute("SELECT task, seed FROM episodes"))
                saved = sum(target in completed_targets for target in lane_targets(config["protocol"], lane, len(AGENT_GPUS)))
                if previous is not None and saved == saved_before_launch[lane]:
                    retries[lane] += 1
                    if retries[lane] > 2:
                        stopped.add(lane)
                        print(f"分片 {lane + 1} 多次中断，详见 {state / f'lane-{lane + 1}.log'}", flush=True)
                        continue
                else:
                    retries[lane] = 0
                saved_before_launch[lane] = saved
                workers[lane] = start_process(
                    [sys.executable, str(Path(__file__).resolve()), "--state", str(state), "--lane", str(lane)],
                    state / f"lane-{lane + 1}.log",
                    {
                        **env, "CUDA_VISIBLE_DEVICES": str(gpu), "MUJOCO_EGL_DEVICE_ID": str(gpu),
                        "RMA_AGENT_HOST": "127.0.0.1", "RMA_AGENT_PORT": str(FIRST_AGENT_PORT + lane),
                        "GAMMA_STATELESS_OPENPI": "1", "RMA_BENCH_ROOT": str(benchmark / "evaluation_benchmark"),
                        "LIBERO_CONFIG_PATH": str(libero_config),
                        "RMA_PLAN_LABELS": str(PROJECT / config["vla_labels"]),
                        "PYTHONPATH": str(benchmark / "third_party/openpi_minimal/packages/openpi-client/src"),
                    },
                )
            status = "运行中" if not stopped else f"运行中，{len(stopped)} 个分片暂时中断"
            completed = write_results(state, result, config, status)
            if all(lane in stopped or process.poll() == 0 for lane, process in workers.items()):
                return completed
            sleep(30)
    finally:
        stop_processes(workers.values())
        stop_processes(process for process, _ in services.values())


def wait_for_models(services: dict) -> None:
    """检查每个模型服务的进程与 HTTP 健康端点。"""
    opener = build_opener(ProxyHandler({}))
    pending = set(services)
    deadline = monotonic() + 900
    while pending:
        for port in tuple(pending):
            process, endpoint = services[port]
            if process.poll() is not None:
                raise ModelServiceError(f"model on port {port} exited with {process.returncode} during startup")
            try:
                with opener.open(f"http://127.0.0.1:{port}{endpoint}", timeout=2) as response:
                    if response.status == 200:
                        pending.remove(port)
            except (URLError, TimeoutError):
                continue
        if pending:
            if monotonic() >= deadline:
                raise ModelServiceError(f"model startup timed out on ports {sorted(pending)}")
            sleep(2)


def start_process(command: list[str], log: Path, env: dict) -> subprocess.Popen:
    """启动一个由当前评测回收的进程组。"""
    with log.open("a") as output:
        return subprocess.Popen(command, cwd=PROJECT, env=env, stdout=output, stderr=subprocess.STDOUT, start_new_session=True)


def stop_processes(processes) -> None:
    """回收当前评测启动的服务或分片。"""
    running = [process for process in processes if process.poll() is None]
    for process in running:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            continue
    for process in running:
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()


def write_results(state: Path, result: Path, config: dict, status: str) -> int:
    """从已提交的唯一回合记录生成结果快照。"""
    with closing(sqlite3.connect(state / "episodes.sqlite3", timeout=30)) as database:
        episodes = [json.loads(data) for data, in database.execute("SELECT data FROM episodes ORDER BY task, seed")]
    protocol = config["protocol"]
    seeds = protocol["seeds"]
    single_seed = len(seeds) == 1
    seed_label = str(seeds[0]) if single_seed else f"{seeds[0]}–{seeds[-1]}"
    if not single_seed:
        csv_output = io.StringIO(newline="")
        writer = csv.DictWriter(csv_output, fieldnames=EPISODE_FIELDS)
        writer.writeheader()
        writer.writerows({**episode, "stages": json.dumps(episode["stages"], ensure_ascii=False)} for episode in episodes)
        atomic_write(result / "episodes.csv", csv_output.getvalue())
    by_task = defaultdict(list)
    for episode in episodes:
        by_task[episode["task"]].append(episode)

    planned = len(protocol["tasks"]) * len(seeds)
    elapsed_hours = (datetime.now(UTC) - datetime.fromisoformat(config["started_at_utc"])).total_seconds() / 3600
    report = [
        f"# GAMMA 在 RoboMemArena 上的评测（seed {seed_label}）", "",
        f"**{status}**。已完成 {len(episodes)} / {planned} 个回合，从本次评测启动起经过 {elapsed_hours:.2f} 小时。", "",
        f"覆盖 {len(protocol['tasks'])} 个任务，每任务 {len(seeds)} 个布局，seed {seed_label}。每次预测执行 {protocol['chunk_steps']} 步，策略上限 {protocol['max_policy_steps']} 步。",
        "GAMMA 使用作者的固定子任务序列和机械核验。原论文的 RoboMemArena 专用模型包仍缺，本次使用 Qwen3.8-27B writer、SAM3 与 RoboMemArena 发布版 π₀.₅。 [论文评测协议](https://arxiv.org/html/2610.12793v1#A7) · [权重说明](https://github.com/1999Lyd/GAMMA/blob/main/CHECKPOINTS.md)", "",
        f"评分使用 GAMMA 论文发表时可用的 RoboMemArena 提交 `{BENCHMARK_REVISION[:7]}`。该版倒液评分测试有 4 处断言与实现不一致，实测沿用该提交的评分实现。 [RoboMemArena 提交](https://github.com/OpenHelix-Team/RoboMemArena/commit/{BENCHMARK_REVISION})", "",
    ]
    if single_seed and seeds[0] >= 100:
        report += ["本次使用演示训练布局。每任务的 TSR 由单个回合决定。", ""]
    if "initial_rollouts" in config:
        initial = config["initial_rollouts"]
        report += [
            f"切换到 `{BENCHMARK_REVISION[:7]}` 前有 {len(initial['task_seed_pairs'])} 个回合已完成，实际运行的提交为 `{initial['commit'][:7]}`。版本差异只涉及任务 5 的评分，保留的回合均来自其他任务。", "",
        ]
    metrics = ["仿真步", "耗时 (s)"] if single_seed else ["平均仿真步", "总仿真步", "平均耗时 (s)", "累计耗时 (s)"]
    report += [
        "TSR 是官方任务成功率，CSR 是官方阶段完成率。仿真步数按成功返回的 `env.step` 调用计，含 10 步预热。回合耗时从环境 reset 起算，含推理、排队和仿真。累计耗时是各有效回合之和。", "",
        f"| 任务 | 已完成 / 计划 | 成功 | TSR | CSR | {' | '.join(metrics)} |",
        "|:---|" + "---:|" * (4 + len(metrics)),
    ]
    for task in protocol["tasks"]:
        rows = by_task[task]
        name = f"{task:02d} · `{config['task_names'][str(task)]}`"
        if not rows:
            report.append(f"| {name} | 0 / {len(seeds)} |" + " — |" * (3 + len(metrics)))
            continue
        steps = [row["simulation_steps"] for row in rows]
        times = [row["wall_time_s"] for row in rows]
        successes = sum(row["success"] for row in rows)
        stage_rate = mean(row["stage_success_pct"] for row in rows)
        step_time = f"{sum(steps)} | {sum(times):.1f}" if single_seed else f"{mean(steps):.1f} | {sum(steps)} | {mean(times):.1f} | {sum(times):.1f}"
        report.append(
            f"| {name} | {len(rows)} / {len(seeds)} | {successes} | {100 * successes / len(rows):.2f}% | "
            f"{stage_rate:.2f}% | {step_time} |"
        )
    if len(episodes) == planned:
        report += ["", f"全量 TSR **{100 * mean(row['success'] for row in episodes):.2f}%**，CSR **{mean(row['stage_success_pct'] for row in episodes):.2f}%**。"]
    links = "[运行配置](run_config.json)" if single_seed else "[逐回合数据](episodes.csv) · [运行配置](run_config.json)"
    report += ["", links, ""]
    atomic_write(result / "README.md", "\n".join(report))
    return len(episodes)


def atomic_write(path: Path, content: str) -> None:
    """让读取方始终看到完整的上一次或本次结果。"""
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


if __name__ == "__main__":
    main()
