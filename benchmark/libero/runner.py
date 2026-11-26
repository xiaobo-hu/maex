"""Run one LIBERO task through the RoboAgent loop."""

from __future__ import annotations

from typing import Protocol

from roboagent import EpisodeResult

from .adapter import LiberoRobotPort
from .tasks import LiberoTask


class RoboAgentLike(Protocol):
    """Runner 所需的最小 RoboAgent 生命周期。"""

    async def run(self, task: str) -> EpisodeResult: ...


class LiberoEpisodeRunner:
    """负责 reset、运行 Agent 和关闭环境，不管理模型或任务规划。"""

    def __init__(self, agent: RoboAgentLike, robot: LiberoRobotPort) -> None:
        self._agent = agent
        self._robot = robot

    async def run(self, task: LiberoTask) -> EpisodeResult:
        """执行一个 LIBERO episode。"""
        await self._robot.reset(seed=task.spec.seed)
        return await self._agent.run(task.instruction)
