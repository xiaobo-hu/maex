"""RoboMemArena benchmark integration with an isolated Python 3.8 service."""

import sys

# 环境服务必须可在 Python 3.8 中启动，因此这里只在客户端进程暴露
# 依赖 RoboAgent（Python 3.11）的公共类型。
if sys.version_info >= (3, 11):  # noqa: UP036
    from .client.adapter import RoboMemArenaRobotPort
    from .config import RoboMemArenaConfig
    from .environment import RoboMemArenaEnv, RoboMemArenaStep
    from .client.remote_env import RemoteRoboMemArenaEnv

    __all__ = [
        "RemoteRoboMemArenaEnv",
        "RoboMemArenaConfig",
        "RoboMemArenaEnv",
        "RoboMemArenaRobotPort",
        "RoboMemArenaStep",
    ]
else:
    __all__ = []
