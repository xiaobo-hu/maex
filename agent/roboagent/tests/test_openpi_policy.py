from __future__ import annotations

import asyncio

import pytest

from roboagent import ActionChunk, Observation, OpenPiPolicyAdapter


class FakePolicy:
    def infer(self, payload):
        assert payload["prompt"] == "pick"
        return {"actions": [[1, 2], [3, 4]], "server_timing": {"infer_ms": 1}}


def test_openpi_adapter_converts_actions() -> None:
    async def run() -> None:
        adapter = OpenPiPolicyAdapter(
            FakePolicy(),
            lambda instruction, observation: {"prompt": instruction, "state": observation.robot_state},
        )
        result = await adapter.infer("pick", Observation(robot_state={"eef": [0.0]}))
        assert isinstance(result, ActionChunk)
        assert result.actions == [[1.0, 2.0], [3.0, 4.0]]
        assert result.metadata == {"server_timing": {"infer_ms": 1}}

    asyncio.run(run())


def test_openpi_adapter_limits_action_horizon() -> None:
    async def run() -> None:
        adapter = OpenPiPolicyAdapter(
            FakePolicy(),
            lambda instruction, observation: {"prompt": instruction},
            max_actions=1,
        )
        result = await adapter.infer("pick", Observation())
        assert result.actions == [[1.0, 2.0]]

    asyncio.run(run())


def test_openpi_adapter_rejects_bad_actions() -> None:
    class BadPolicy:
        def infer(self, payload):
            del payload
            return {"actions": "bad"}

    async def run() -> None:
        adapter = OpenPiPolicyAdapter(BadPolicy(), lambda instruction, observation: {})
        with pytest.raises(ValueError, match="2-D 'actions'"):
            await adapter.infer("pick", Observation())

    asyncio.run(run())
