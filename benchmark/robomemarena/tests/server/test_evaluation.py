from __future__ import annotations

from dataclasses import dataclass

from benchmark.robomemarena.server.evaluation import OfficialStageEvaluator


@dataclass
class Spec:
    name: str
    threshold: int

    def check_fn(self, env, state, stage_start: int) -> bool:
        return state["step_idx"] >= self.threshold


class FakeOfficial:
    def __init__(
        self,
        names: list[str],
        *,
        counting: bool = False,
        third_at: int | None = None,
        optional_final_stage: str | None = None,
    ):
        self.names = names
        self.counting = counting
        self.third_at = third_at
        self.optional_final_stage = optional_final_stage

    def _task_specs(self, task_id: int) -> list[Spec]:
        return [
            Spec(name, 999 if name == self.optional_final_stage else index + 1)
            for index, name in enumerate(self.names)
        ]

    def _build_initial_state(self, env) -> dict[str, int]:
        return {"step_idx": 0}

    def _update_state(self, observation, state: dict[str, int]) -> None:
        state["step_idx"] += 1

    def _is_counting_pour_task(self, task_id: int) -> bool:
        return self.counting

    def _extra_pour_check(self, task_id: int):
        if not self.counting:
            return None
        return lambda env, state, start: self.third_at == state["step_idx"]

    def _stage_success_from_stage_done(self, task_id: int, done: dict[str, bool]) -> bool:
        required = [
            complete
            for name, complete in done.items()
            if name != self.optional_final_stage
        ]
        return bool(required) and all(required)

    def _stage_score_pct(self, task_id: int, done: dict[str, bool]) -> float:
        required = [
            complete
            for name, complete in done.items()
            if name != self.optional_final_stage
        ]
        return 100.0 * sum(required) / len(required)


def test_non_counting_task_continues_official_post_goal_delay() -> None:
    evaluator = OfficialStageEvaluator(
        1,
        object(),
        max_steps=20,
        post_goal_steps=2,
        official_module=FakeOfficial(["01_Place"]),
    )
    evaluator.update({})
    assert evaluator.done is False
    assert evaluator.metrics()["TSR"] == 1
    assert evaluator.metrics()["stage_success"] is True
    assert evaluator.metrics()["stage_events"] == [
        {"stage": "01_Place", "environment_step": 1}
    ]
    evaluator.update({})
    assert evaluator.done is False
    evaluator.update({})
    assert evaluator.done is True
    assert evaluator.success is True


def test_counting_task_monitors_30_step_semantics() -> None:
    evaluator = OfficialStageEvaluator(
        6,
        object(),
        max_steps=20,
        extra_pour_monitor_steps=3,
        official_module=FakeOfficial(["01_Pour_One", "02_Pour_Two"], counting=True),
    )
    for _ in range(4):
        evaluator.update({})
        assert evaluator.done is False
    evaluator.update({})
    assert evaluator.done is True
    assert evaluator.success is True
    assert evaluator.metrics()["extra_monitor_end_step"] == 4


def test_counting_task_third_pour_fails_immediately() -> None:
    evaluator = OfficialStageEvaluator(
        6,
        object(),
        max_steps=20,
        extra_pour_monitor_steps=3,
        official_module=FakeOfficial(
            ["01_Pour_One", "02_Pour_Two"], counting=True, third_at=3
        ),
    )
    evaluator.update({})
    evaluator.update({})
    evaluator.update({})
    assert evaluator.done is True
    assert evaluator.success is False
    assert evaluator.metrics()["failure_reason"] == "extra_pour"


def test_counting_task_ends_after_required_stages_without_optional_close() -> None:
    optional_close = "04_Close_Microwave"
    evaluator = OfficialStageEvaluator(
        22,
        object(),
        max_steps=20,
        extra_pour_monitor_steps=1,
        official_module=FakeOfficial(
            [
                "01_Pour_One",
                "02_Pour_Two",
                "03_Place_Cookies_Microwave",
                optional_close,
            ],
            counting=True,
            optional_final_stage=optional_close,
        ),
    )
    evaluator.update({})
    evaluator.update({})
    evaluator.update({})

    assert evaluator.done is True
    assert evaluator.success is True
    assert evaluator.metrics()["stage_done"][optional_close] is False


def test_max_step_terminates_incomplete_episode() -> None:
    evaluator = OfficialStageEvaluator(
        1,
        object(),
        max_steps=2,
        official_module=FakeOfficial(["01", "02", "03"]),
    )
    evaluator.update({})
    evaluator.update({})
    assert evaluator.done is True
    assert evaluator.success is False
    assert evaluator.metrics()["failure_reason"] == "incomplete_stage"
