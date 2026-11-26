"""Incremental wrapper around the fixed upstream official stage checker."""

from __future__ import annotations

from typing import Any

from .upstream import load_official_modules


class OfficialStageEvaluator:
    """Apply upstream stage checks after each action using official stop semantics."""

    def __init__(
        self,
        task_id: int,
        env: Any,
        *,
        max_steps: int = 2500,
        post_goal_steps: int = 200,
        extra_pour_monitor_steps: int = 30,
        official_module: Any = None,
    ) -> None:
        if official_module is None:
            _common, official_module = load_official_modules()
        self._official = official_module
        self._task_id = task_id
        self._env = env
        self._max_steps = max_steps
        self._post_goal_steps = post_goal_steps
        self._extra_pour_monitor_steps = extra_pour_monitor_steps
        self._specs = list(official_module._task_specs(task_id))
        self._stage_done = {spec.name: False for spec in self._specs}
        self._stage_events: list[dict[str, object]] = []
        self._stage_idx = 0
        self._state = official_module._build_initial_state(env)
        self._current_stage_start = int(self._state["step_idx"])
        self._counting = bool(official_module._is_counting_pour_task(task_id))
        self._extra_pour_check = official_module._extra_pour_check(task_id)
        self._step = 0
        self._goal_reached_step = None
        self._extra_monitor_start_state_idx = None
        self._extra_monitor_deadline_step = None
        self._extra_pour_detected = False
        self._pour_1_step = None
        self._pour_2_step = None
        self._done = False

    @property
    def done(self) -> bool:
        """Whether official or simulator termination has been reached."""
        return self._done

    @property
    def success(self) -> bool:
        """Return final TSR only after termination."""
        return self._done and self._stage_success()

    @property
    def remaining_stage_checks(self) -> int:
        """Return ordered stage checks that may still need one update each."""
        return len(self._specs) - self._stage_idx

    def update(self, observation: Any, *, simulator_done: bool = False) -> None:
        """Consume one post-action simulator state and update official metrics."""
        if self._done:
            raise RuntimeError("RoboMemArena episode is already done")
        t = self._step
        self._official._update_state(observation, self._state)

        if self._stage_idx < len(self._specs):
            spec = self._specs[self._stage_idx]
            if spec.check_fn(self._env, self._state, self._current_stage_start):
                self._stage_done[spec.name] = True
                self._stage_events.append(
                    {"stage": spec.name, "environment_step": t + 1}
                )
                if spec.name.endswith("_Pour_One"):
                    self._pour_1_step = t
                elif spec.name.endswith("_Pour_Two"):
                    self._pour_2_step = t
                    if self._counting:
                        self._extra_monitor_start_state_idx = int(self._state["step_idx"])
                        self._extra_monitor_deadline_step = t + self._extra_pour_monitor_steps
                self._stage_idx += 1
                self._current_stage_start = int(self._state["step_idx"])

        if self._should_check_third_pour(t):
            assert self._extra_pour_check is not None
            if self._extra_pour_check(
                self._env,
                self._state,
                self._extra_monitor_start_state_idx,
            ):
                self._extra_pour_detected = True

        required_complete = self._required_complete()
        if not self._counting and self._goal_reached_step is None and required_complete:
            self._goal_reached_step = t

        self._step += 1
        post_goal_complete = (
            self._goal_reached_step is not None
            and t - self._goal_reached_step >= self._post_goal_steps
        )
        counting_complete = required_complete and self._extra_monitor_complete(t)
        self._done = bool(
            simulator_done
            or self._step >= self._max_steps
            or (not self._counting and post_goal_complete)
            or (self._counting and (self._extra_pour_detected or counting_complete))
        )

    def metrics(self) -> dict[str, object]:
        """Return evaluation-only CSR, TSR, and official diagnostics."""
        current_t = max(0, self._step - 1)
        stage_success = self._stage_success()
        failure_reason = None
        if self._done:
            if self._extra_pour_detected:
                failure_reason = "extra_pour"
            elif not stage_success:
                failure_reason = "incomplete_stage"
            elif self._counting and not self._extra_monitor_complete(current_t):
                failure_reason = "monitor_incomplete"
        return {
            "task_id": self._task_id,
            "step": self._step,
            "CSR": float(self._official._stage_score_pct(self._task_id, self._stage_done)),
            "TSR": int(stage_success),
            "stage_score_pct": float(
                self._official._stage_score_pct(self._task_id, self._stage_done)
            ),
            "stage_success": bool(stage_success),
            "stage_done": dict(self._stage_done),
            "stage_events": [dict(event) for event in self._stage_events],
            "required_stages_complete": bool(self._required_complete()),
            "extra_pour_detected": bool(self._extra_pour_detected),
            "pour_1_step": self._pour_1_step,
            "pour_2_step": self._pour_2_step,
            "extra_monitor_end_step": (
                self._extra_monitor_deadline_step
                if self._extra_monitor_complete(current_t)
                else None
            ),
            "failure_reason": failure_reason,
            "done": self._done,
        }

    def _required_complete(self) -> bool:
        return bool(
            self._official._stage_success_from_stage_done(self._task_id, self._stage_done)
        )

    def _extra_monitor_complete(self, t: int) -> bool:
        return bool(
            not self._counting
            or (
                self._extra_monitor_deadline_step is not None
                and t >= self._extra_monitor_deadline_step
            )
        )

    def _stage_success(self) -> bool:
        current_t = max(0, self._step - 1)
        return bool(
            self._required_complete()
            and (
                not self._counting
                or (
                    self._extra_monitor_complete(current_t)
                    and not self._extra_pour_detected
                )
            )
        )

    def _should_check_third_pour(self, t: int) -> bool:
        return bool(
            self._counting
            and self._extra_pour_check is not None
            and self._extra_monitor_start_state_idx is not None
            and self._extra_monitor_deadline_step is not None
            and self._pour_2_step is not None
            and self._pour_2_step < t <= self._extra_monitor_deadline_step
        )
