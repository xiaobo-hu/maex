from __future__ import annotations

import json
import stat
from argparse import Namespace
from hashlib import sha256

from benchmark.robomemarena.scripts.roboagent_eval import (
    _create_planner,
    _create_planner_stream_options,
    _output_result,
    _read_plan_advance_approval,
    _snapshot_task_execution_skill,
    _write_run_config,
)

from roboagent import TaskExecutionSkill, TaskStep


def test_glm_planner_uses_zhipu_vision_endpoint() -> None:
    model, _stream_fn = _create_planner("glm", None, "place the objects")

    assert model.id == "glm-4.6v-flash"
    assert model.provider == "zhipu"
    assert model.base_url == "https://open.bigmodel.cn/api/paas/v4"
    assert model.input == ["text", "image"]


def test_deepseek_planner_uses_official_vision_endpoint(monkeypatch) -> None:
    monkeypatch.setenv("LLM_MODEL_NAME", "deepseek-v4-flash-vision-exp")

    model, _stream_fn = _create_planner("deepseek", None, "place the objects")

    assert model.id == "deepseek-v4-flash-vision-exp"
    assert model.provider == "deepseek"
    assert model.base_url == "https://api.deepseek.com/v1"
    assert model.input == ["text", "image"]


def test_qwen_planner_uses_dashscope_compatible_endpoint(monkeypatch) -> None:
    monkeypatch.setenv("LLM_MODEL_NAME", "qwen3.8-max")
    monkeypatch.setenv("LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")

    model, _stream_fn = _create_planner("qwen", None, "place the objects")

    assert model.id == "qwen3.8-max"
    assert model.provider == "qwen"
    assert model.base_url == "https://dashscope.aliyuncs.com/compatible-mode/v1"
    assert model.input == ["text", "image"]


def test_qwen_planner_maps_disabled_thinking_to_dashscope(monkeypatch) -> None:
    monkeypatch.setenv("LLM_DISABLE_THINKING", "true")

    options = _create_planner_stream_options("qwen", max_retries=3, max_retry_delay_ms=5_000)

    assert options is not None
    assert options.extra_body == {"enable_thinking": False}


def test_faux_planner_is_available_for_local_closed_loop_smoke() -> None:
    model, stream_fn = _create_planner("faux", None, "place the objects")

    assert model.provider == "faux"
    assert callable(stream_fn)


def test_output_result_persists_episode_and_official_evaluation(tmp_path) -> None:
    trace_file = tmp_path / "task1.json"
    output = {
        "episode": {"status": "completed", "details": {"observer_trace": []}},
        "evaluation": {"CSR": 50.0, "TSR": 0},
    }

    _output_result(output, trace_file)

    assert json.loads(trace_file.read_text()) == output


def test_terminal_plan_approval_rejects_with_feedback(monkeypatch) -> None:
    answers = iter(("n", "The drawer is still closed."))
    monkeypatch.setattr("builtins.input", lambda _prompt: next(answers))

    approved, feedback = _read_plan_advance_approval(
        TaskStep(id="open", instruction="open drawer", status="running"),
        TaskStep(id="pick", instruction="pick object", status="pending"),
        "The drawer may be open.",
    )

    assert approved is False
    assert feedback == "The drawer is still closed."


def test_run_directory_snapshots_task_execution_skill(tmp_path) -> None:
    source = tmp_path / "candidate"
    source.mkdir()
    content = "Use direct evidence before retrying."
    digest = sha256(content.encode()).hexdigest()
    (source / "SKILL.md").write_text(content + "\n", encoding="utf-8")
    (source / "PURPOSE.md").write_text("One atomic change.\n", encoding="utf-8")
    (source / "manifest.json").write_text("{}\n", encoding="utf-8")
    skill = TaskExecutionSkill(
        skill_id="embodied-task-execution",
        version="v1",
        content=content,
        content_sha256=digest,
        source_path=str(source / "SKILL.md"),
    )
    run_dir = tmp_path / "run"

    _write_run_config(run_dir, Namespace(task=1), "task", "model", skill)
    _snapshot_task_execution_skill(run_dir, skill)

    config = json.loads((run_dir / "run_config.json").read_text())
    assert config["task_execution_skill_sha256"] == digest
    assert config["task_execution_skill_version"] == "v1"
    snapshot = run_dir / "task_execution_skill/SKILL.md"
    assert snapshot.read_text().strip() == content
    assert snapshot.stat().st_mode & stat.S_IWUSR == 0
