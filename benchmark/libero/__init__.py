"""LIBERO benchmark integration with a Python 3.8 simulator boundary."""

import sys

# LIBERO 官方环境固定在 Python 3.8；该进程只导入服务端模块，不能提前
# 加载要求 Python 3.11 的 RoboAgent 领域包。
if sys.version_info >= (3, 11):  # noqa: UP036
    from .adapter import LiberoRobotPort
    from .config import LiberoEvalConfig, LiberoTaskSpec
    from .environment import LiberoEnv, LiberoStep
    from .runner import LiberoEpisodeRunner
    from .tasks import LiberoTask, make_task

    __all__ = [
        "LiberoEnv",
        "LiberoEpisodeRunner",
        "LiberoEvalConfig",
        "LiberoRobotPort",
        "LiberoStep",
        "LiberoTaskSpec",
        "LiberoTask",
        "make_task",
    ]
else:
    __all__ = []
