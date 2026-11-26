"""视觉与机器人执行哨兵的公开协议、类型和服务适配器。"""

from ..memory import GripperEventRecord, VisualKeyframe, VisualMemory
from .execution_diagnostics import ExecutionDiagnosticsRunner
from .openai_adapter import OpenAiObserverAdapter
from .ports import ObservationSink, ObservationSource, ObserverPort
from .sampler import (
    ObserverSamplingOptions,
    detect_gripper_keyframes,
    sample_observer_window,
)
from .stream import create_observation_stream
from .types import ObservationEvent, ObserverAssessment, ObserverWindow
from .visual_observer import VisualObserver

# 兼容现有 benchmark 组合代码；新代码使用语义明确的 VisualObserver。
ObserverRunner = VisualObserver

__all__ = [
    "ObservationEvent",
    "ObservationSink",
    "ObservationSource",
    "ObserverAssessment",
    "ObserverPort",
    "ObserverRunner",
    "ObserverSamplingOptions",
    "ObserverWindow",
    "GripperEventRecord",
    "VisualKeyframe",
    "ExecutionDiagnosticsRunner",
    "OpenAiObserverAdapter",
    "create_observation_stream",
    "detect_gripper_keyframes",
    "sample_observer_window",
    "VisualMemory",
    "VisualObserver",
]
