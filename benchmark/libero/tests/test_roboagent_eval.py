from __future__ import annotations

import json

from benchmark.libero.scripts.roboagent_eval import _create_planner, _output_result

from roboagent import EpisodeResult


def test_glm_planner_uses_zhipu_vision_endpoint() -> None:
    model, _stream_fn = _create_planner("glm", None, "pick the bowl")

    assert model.id == "glm-4.6v-flash"
    assert model.provider == "zhipu"
    assert model.base_url == "https://open.bigmodel.cn/api/paas/v4"
    assert model.input == ["text", "image"]


def test_output_result_displays_and_saves_tool_trace(tmp_path, capsys) -> None:
    trace_file = tmp_path / "logs" / "episode.json"
    result = EpisodeResult(
        status="success",
        summary="done",
        turns=2,
        tool_calls=1,
        details={"tool_trace": [{"name": "finish", "status": "completed"}]},
    )

    _output_result(result, trace_file)

    saved = json.loads(trace_file.read_text(encoding="utf-8"))
    assert saved["details"]["tool_trace"][0]["name"] == "finish"
    assert '"tool_trace"' in capsys.readouterr().out
