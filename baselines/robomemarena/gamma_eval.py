"""评测客户端层，通过原 RoboMemArena 评测循环记录 GAMMA 回合的步数与耗时。"""

import json
import logging
import os
import sqlite3
import sys
from contextlib import closing
from pathlib import Path
from time import monotonic


class EpisodeMeter(logging.Handler):
    """统计环境控制步，向外传递原评测循环捕获的运行异常。"""

    def __init__(self):
        super().__init__(logging.ERROR)
        self.steps = 0
        self.seconds = 0.0
        self.error = None

    def emit(self, record):
        if record.exc_info and record.getMessage().startswith("Episode failed:"):
            self.error = record.exc_info[1]

    def wrap(self, evaluate):
        def measured(*, env, **kwargs):
            self.steps = 0
            self.error = None
            original_step = env.step

            def step(action):
                observation = original_step(action)
                self.steps += 1
                return observation

            env.step = step
            started = monotonic()
            try:
                outcome = evaluate(env=env, **kwargs)
            finally:
                self.seconds = monotonic() - started
                del env.step
            if self.error is not None:
                raise RuntimeError("RoboMemArena rollout failed before scoring") from self.error
            # 上游返回两路用于编码的视频帧。逐回合数据保存官方阶段结果与计量值。
            return (*outcome[:-2], [], [])

        return measured


def lane_targets(protocol: dict, lane: int, lane_count: int) -> list[tuple[int, int]]:
    """按固定顺序分配任务与种子，保证分片重启后可续跑。"""
    targets = [(task, seed) for seed in protocol["seeds"] for task in protocol["tasks"]]
    return targets[lane::lane_count]


def run_lane(state: Path, lane: int, lane_count: int) -> None:
    """运行固定分片，使用已落盘的种子记录续跑。"""
    stack = Path(__file__).resolve().parents[1] / "GAMMA/rma/eval_stack"
    sys.path.insert(0, str(stack))
    import run_rma_gamma_eval as gamma

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("websockets").setLevel(logging.WARNING)
    config = json.loads((state / "config.json").read_text())
    protocol = config["protocol"]
    meter = EpisodeMeter()
    logging.getLogger().addHandler(meter)
    gamma.tasks26._patch_env_resolution()
    gamma.ec.run_episode_with_stages = meter.wrap(gamma.ec.run_episode_with_stages)
    gamma.tasks26.run_episode_with_stateful_stages = meter.wrap(gamma.tasks26.run_episode_with_stateful_stages)

    # 角色与标签沿用 GAMMA 的现有适配。每个分片连接自己的 GAMMA agent。
    provider = gamma.GammaProvider()
    adapter = gamma.mme_rma_adapter.build_adapter(
        host="127.0.0.1", port=int(os.environ["BASELINE_VLA_PORT"]), action_horizon=10,
        prompt_from_subgoal=True,
    )
    provider.bind(adapter)
    try:
        with closing(sqlite3.connect(state / "episodes.sqlite3", timeout=30)) as database:
            completed = set(database.execute("SELECT task, seed FROM episodes"))
            for task, seed in lane_targets(protocol, lane, lane_count):
                if (task, seed) in completed:
                    continue
                logging.info("Task %d seed %d started", task, seed)
                provider.task_id = task
                provider.plan = gamma.PLANS[task]
                provider.prompt = gamma.ec.TASK_PROMPTS[f"task{task}"]
                provider.episode = seed - 1
                result = gamma.r._run_task(
                    task_id=task, adapter=adapter, num_trials_per_task=1,
                    resize_size=protocol["resize"], replan_steps=protocol["chunk_steps"],
                    num_steps_wait=protocol["warmup_steps"], max_steps=protocol["max_policy_steps"],
                    post_goal_steps=protocol["post_goal_steps"], fail_on_extra_pour=True,
                    extra_pour_monitor_steps=protocol["extra_pour_monitor_steps"],
                    video_dir=state / "eval_work", seed=seed,
                )
                episode, = result["episodes"]
                if episode["seed"] != seed or meter.steps <= protocol["warmup_steps"]:
                    raise RuntimeError(f"Invalid episode record for task {task}, seed {seed}")
                record = {
                    "task": task,
                    "seed": seed,
                    "success": int(episode["TSR"] == 100.0),
                    "stage_success_pct": episode["CSR"],
                    "simulation_steps": meter.steps,
                    "policy_steps": meter.steps - protocol["warmup_steps"],
                    "wall_time_s": round(meter.seconds, 6),
                    "policy_calls": provider.calls,
                    "failure_reason": episode.get("failure_reason") or "",
                    "stages": episode["stage_done"],
                }
                with database:
                    database.execute("INSERT INTO episodes VALUES (?, ?, ?)", (task, seed, json.dumps(record)))
                logging.info(
                    "Task %d seed %d saved, success=%d steps=%d elapsed=%.1fs",
                    task, seed, record["success"], meter.steps, meter.seconds,
                )
    finally:
        adapter.close()
