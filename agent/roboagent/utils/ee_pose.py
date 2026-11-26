"""把受限末端语义动作转换为当前 OSC_POSE 控制器使用的增量。"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, TypeAlias

from ..types import Observation

Vector3: TypeAlias = tuple[float, float, float]
Quaternion: TypeAlias = tuple[float, float, float, float]
EePoseAdjustmentAction = Literal[
    "world_z_up",
    "camera_z_retract",
    "camera_x_left",
    "camera_x_right",
    "camera_rx_pos",
    "camera_rx_neg",
    "camera_ry_pos",
    "camera_ry_neg",
]

TRANSLATION_ACTIONS = frozenset(
    {"world_z_up", "camera_z_retract", "camera_x_left", "camera_x_right"}
)
ROTATION_ACTIONS = frozenset(
    {"camera_rx_pos", "camera_rx_neg", "camera_ry_pos", "camera_ry_neg"}
)

# Panda eye_in_hand 相机应用 robosuite optical-frame 修正后，相对 right_hand：
# +X_cam = +Y_ee，+Y_cam = -X_ee，+Z_cam = +Z_ee。
_CAMERA_X_IN_EE: Vector3 = (0.0, 1.0, 0.0)
_CAMERA_Y_IN_EE: Vector3 = (-1.0, 0.0, 0.0)
_CAMERA_Z_IN_EE: Vector3 = (0.0, 0.0, 1.0)


@dataclass(frozen=True, slots=True)
class EePoseAdjustmentConfig:
    """受限局部恢复动作的安全幅度和 OSC 控制参数。"""

    max_translation_m: float = 0.08
    max_rotation_deg: float = 30.0
    translation_tolerance_m: float = 0.003
    rotation_tolerance_deg: float = 1.5
    max_control_steps: int = 12
    osc_translation_scale_m: float = 0.05
    osc_rotation_scale_rad: float = 0.5
    workspace_min_m: Vector3 | None = None
    workspace_max_m: Vector3 | None = None

    def __post_init__(self) -> None:
        values = (
            self.max_translation_m,
            self.max_rotation_deg,
            self.translation_tolerance_m,
            self.rotation_tolerance_deg,
            self.osc_translation_scale_m,
            self.osc_rotation_scale_rad,
        )
        if any(not math.isfinite(value) or value <= 0 for value in values):
            raise ValueError("EE pose adjustment limits and tolerances must be positive")
        if self.max_control_steps < 1:
            raise ValueError("max_control_steps must be at least 1")
        if (self.workspace_min_m is None) != (self.workspace_max_m is None):
            raise ValueError("workspace bounds must be configured together")
        if (
            self.workspace_min_m is not None
            and self.workspace_max_m is not None
            and any(
                low >= high
                for low, high in zip(
                    self.workspace_min_m,
                    self.workspace_max_m,
                    strict=True,
                )
            )
        ):
            raise ValueError("Each workspace minimum must be below its maximum")


@dataclass(frozen=True, slots=True)
class EePose:
    """世界坐标系中的末端位置和轴角。"""

    position: Vector3
    axis_angle: Vector3

    def summary(self) -> dict[str, list[float]]:
        """返回 Agent 可读且 JSON-safe 的 pose。"""
        return {
            "position_m": list(self.position),
            "axis_angle_rad": list(self.axis_angle),
        }


def pose_from_observation(observation: Observation) -> EePose:
    """从公开 robot state 构造 EEF pose，不读取 benchmark 内部状态。"""
    return EePose(
        position=_vector3(observation.robot_state.get("eef_position"), "eef_position"),
        axis_angle=_vector3(
            observation.robot_state.get("eef_axis_angle"),
            "eef_axis_angle",
        ),
    )


def build_adjustment_target(
    pose: EePose,
    action: EePoseAdjustmentAction,
    magnitude: float,
    config: EePoseAdjustmentConfig,
) -> EePose:
    """校验单轴语义动作并生成固定的世界系目标 pose。"""
    if not math.isfinite(magnitude) or magnitude <= 0:
        raise ValueError("magnitude must be a positive finite number")
    orientation = _axis_angle_to_quaternion(pose.axis_angle)
    camera_x = _rotate(orientation, _CAMERA_X_IN_EE)
    camera_y = _rotate(orientation, _CAMERA_Y_IN_EE)
    camera_z = _rotate(orientation, _CAMERA_Z_IN_EE)
    if action in TRANSLATION_ACTIONS:
        if magnitude > config.max_translation_m:
            raise ValueError(
                f"translation magnitude exceeds {config.max_translation_m} meters"
            )
        directions = {
            "world_z_up": (0.0, 0.0, 1.0),
            "camera_z_retract": _scale(camera_z, -1.0),
            "camera_x_left": _scale(camera_x, -1.0),
            "camera_x_right": camera_x,
        }
        return EePose(
            position=_add(pose.position, _scale(directions[action], magnitude)),
            axis_angle=pose.axis_angle,
        )
    if action in ROTATION_ACTIONS:
        if magnitude > config.max_rotation_deg:
            raise ValueError(f"rotation magnitude exceeds {config.max_rotation_deg} degrees")
        axis = camera_x if action.startswith("camera_rx") else camera_y
        sign = 1.0 if action.endswith("_pos") else -1.0
        delta = _axis_angle_to_quaternion(
            _scale(axis, math.radians(magnitude) * sign)
        )
        target = _multiply_quaternions(delta, orientation)
        return EePose(
            position=pose.position,
            axis_angle=_quaternion_to_axis_angle(target),
        )
    raise ValueError(f"Unsupported EE pose adjustment action: {action}")


def build_osc_action(
    current: EePose,
    target: EePose,
    gripper_command: float,
    config: EePoseAdjustmentConfig,
) -> list[float]:
    """生成一个归一化 7D OSC action，同时补偿非目标自由度漂移。"""
    translation = _clip(
        _subtract(target.position, current.position),
        config.osc_translation_scale_m,
    )
    rotation = _clip(_rotation_delta(current, target), config.osc_rotation_scale_rad)
    action = [
        *(value / config.osc_translation_scale_m for value in translation),
        *(value / config.osc_rotation_scale_rad for value in rotation),
        gripper_command,
    ]
    return [max(-1.0, min(1.0, float(value))) for value in action]


def target_is_reached(
    current: EePose,
    target: EePose,
    config: EePoseAdjustmentConfig,
) -> bool:
    """同时检查目标位置和朝向，避免非目标自由度静默漂移。"""
    return (
        _norm(_subtract(target.position, current.position))
        <= config.translation_tolerance_m
        and rotation_distance_deg(current, target) <= config.rotation_tolerance_deg
    )


def target_is_in_workspace(target: EePose, config: EePoseAdjustmentConfig) -> bool:
    """在配置绝对工作区时检查目标位置。"""
    return workspace_violation_distance(target, config) == 0.0


def workspace_motion_is_allowed(
    current: EePose,
    target: EePose,
    config: EePoseAdjustmentConfig,
) -> bool:
    """只允许安全区内目标，或让已越界 EEF 明确靠近安全区的目标。"""
    current_distance = workspace_violation_distance(current, config)
    target_distance = workspace_violation_distance(target, config)
    if current_distance == 0.0:
        return target_distance == 0.0
    return target_distance < current_distance


def workspace_violation_distance(
    pose: EePose,
    config: EePoseAdjustmentConfig,
) -> float:
    """返回 EEF 到配置 AABB 的欧氏距离；未配置或位于内部时为零。"""
    if config.workspace_min_m is None or config.workspace_max_m is None:
        return 0.0
    violations = tuple(
        low - value if value < low else value - high if value > high else 0.0
        for value, low, high in zip(
            pose.position,
            config.workspace_min_m,
            config.workspace_max_m,
            strict=True,
        )
    )
    return _norm(violations)


def primary_error(
    current: EePose,
    target: EePose,
    action: EePoseAdjustmentAction,
) -> float:
    """返回语义动作主要自由度的剩余误差。"""
    if action in TRANSLATION_ACTIONS:
        return _norm(_subtract(target.position, current.position))
    return rotation_distance_deg(current, target)


def rotation_distance_deg(left: EePose, right: EePose) -> float:
    """返回两个姿态之间的最短角距离，单位为度。"""
    return math.degrees(_norm(_rotation_delta(left, right)))


def applied_rotation_deg(before: EePose, after: EePose) -> Vector3:
    """返回实际世界系轴角变化，单位为度。"""
    return _scale(_rotation_delta(before, after), 180.0 / math.pi)


def vector_subtract(left: Vector3, right: Vector3) -> Vector3:
    """返回两个三维向量之差。"""
    return _subtract(left, right)


def _rotation_delta(current: EePose, target: EePose) -> Vector3:
    current_q = _axis_angle_to_quaternion(current.axis_angle)
    target_q = _axis_angle_to_quaternion(target.axis_angle)
    delta = _multiply_quaternions(target_q, _conjugate(current_q))
    return _quaternion_to_axis_angle(delta)


def _axis_angle_to_quaternion(value: Vector3) -> Quaternion:
    angle = _norm(value)
    if angle < 1e-12:
        return (1.0, 0.0, 0.0, 0.0)
    scale = math.sin(angle / 2.0) / angle
    return (
        math.cos(angle / 2.0),
        value[0] * scale,
        value[1] * scale,
        value[2] * scale,
    )


def _quaternion_to_axis_angle(value: Quaternion) -> Vector3:
    norm = math.sqrt(sum(component * component for component in value))
    quaternion = tuple(component / norm for component in value)
    if quaternion[0] < 0:
        quaternion = tuple(-component for component in quaternion)
    vector_norm = math.sqrt(sum(component * component for component in quaternion[1:]))
    if vector_norm < 1e-12:
        return (0.0, 0.0, 0.0)
    angle = 2.0 * math.atan2(vector_norm, quaternion[0])
    return tuple(angle * component / vector_norm for component in quaternion[1:])  # type: ignore[return-value]


def _multiply_quaternions(left: Quaternion, right: Quaternion) -> Quaternion:
    lw, lx, ly, lz = left
    rw, rx, ry, rz = right
    return (
        lw * rw - lx * rx - ly * ry - lz * rz,
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
    )


def _conjugate(value: Quaternion) -> Quaternion:
    return (value[0], -value[1], -value[2], -value[3])


def _rotate(rotation: Quaternion, vector: Vector3) -> Vector3:
    rotated = _multiply_quaternions(
        _multiply_quaternions(rotation, (0.0, *vector)),
        _conjugate(rotation),
    )
    return rotated[1], rotated[2], rotated[3]


def _clip(vector: Vector3, maximum: float) -> Vector3:
    norm = _norm(vector)
    return vector if norm <= maximum else _scale(vector, maximum / norm)


def _add(left: Vector3, right: Vector3) -> Vector3:
    return tuple(a + b for a, b in zip(left, right, strict=True))  # type: ignore[return-value]


def _subtract(left: Vector3, right: Vector3) -> Vector3:
    return tuple(a - b for a, b in zip(left, right, strict=True))  # type: ignore[return-value]


def _scale(vector: Vector3, scale: float) -> Vector3:
    return tuple(value * scale for value in vector)  # type: ignore[return-value]


def _norm(vector: Vector3) -> float:
    return math.sqrt(sum(value * value for value in vector))


def _vector3(value: object, name: str) -> Vector3:
    if not isinstance(value, list | tuple) or len(value) != 3:
        raise ValueError(f"robot_state.{name} must contain three finite values")
    try:
        result = tuple(float(item) for item in value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"robot_state.{name} must contain three finite values") from error
    if not all(math.isfinite(item) for item in result):
        raise ValueError(f"robot_state.{name} must contain three finite values")
    return result  # type: ignore[return-value]
