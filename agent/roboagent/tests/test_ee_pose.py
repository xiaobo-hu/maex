"""验证受限 EEF 语义动作到 OSC 增量的确定性转换。"""

import math

import pytest

from roboagent.utils.ee_pose import (
    EePose,
    EePoseAdjustmentConfig,
    build_adjustment_target,
    build_osc_action,
    target_is_in_workspace,
    workspace_motion_is_allowed,
    workspace_violation_distance,
)


@pytest.mark.parametrize(
    ("action", "expected"),
    [
        ("world_z_up", (0.0, 0.0, 0.02)),
        ("camera_z_retract", (0.0, 0.0, -0.02)),
        ("camera_x_left", (0.0, -0.02, 0.0)),
        ("camera_x_right", (0.0, 0.02, 0.0)),
    ],
)
def test_translation_actions_use_declared_world_and_camera_axes(action, expected) -> None:
    pose = EePose((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))

    target = build_adjustment_target(pose, action, 0.02, EePoseAdjustmentConfig())

    assert target.position == pytest.approx(expected)
    assert target.axis_angle == pose.axis_angle


@pytest.mark.parametrize(
    ("action", "expected"),
    [
        ("camera_rx_pos", (0.0, math.radians(10.0), 0.0)),
        ("camera_rx_neg", (0.0, -math.radians(10.0), 0.0)),
        ("camera_ry_pos", (-math.radians(10.0), 0.0, 0.0)),
        ("camera_ry_neg", (math.radians(10.0), 0.0, 0.0)),
    ],
)
def test_rotation_actions_use_current_camera_axes(action, expected) -> None:
    pose = EePose((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))

    target = build_adjustment_target(pose, action, 10.0, EePoseAdjustmentConfig())

    assert target.position == pose.position
    assert target.axis_angle == pytest.approx(expected)


def test_camera_axis_follows_eef_orientation_near_pi() -> None:
    pose = EePose((0.0, 0.0, 0.0), (math.pi, 0.0, 0.0))

    target = build_adjustment_target(
        pose,
        "camera_x_right",
        0.02,
        EePoseAdjustmentConfig(),
    )

    assert target.position == pytest.approx((0.0, -0.02, 0.0), abs=1e-8)


def test_osc_action_clips_one_step_and_preserves_gripper_command() -> None:
    config = EePoseAdjustmentConfig()
    current = EePose((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
    target = EePose((0.0, 0.0, 0.08), (0.0, 0.0, math.radians(30.0)))

    action = build_osc_action(current, target, 0.7, config)

    assert action == pytest.approx([0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.7])


def test_limits_and_optional_workspace_reject_unsafe_targets() -> None:
    pose = EePose((0.0, 0.0, 0.5), (0.0, 0.0, 0.0))
    config = EePoseAdjustmentConfig(
        workspace_min_m=(-0.2, -0.2, 0.2),
        workspace_max_m=(0.2, 0.2, 0.55),
    )

    with pytest.raises(ValueError, match="translation magnitude"):
        build_adjustment_target(pose, "world_z_up", 0.09, config)
    target = build_adjustment_target(pose, "world_z_up", 0.06, config)

    assert not target_is_in_workspace(target, config)


def test_workspace_allows_only_inward_motion_after_eef_is_outside() -> None:
    config = EePoseAdjustmentConfig(
        workspace_min_m=(-0.2, -0.2, 0.2),
        workspace_max_m=(0.2, 0.2, 0.8),
    )
    outside = EePose((0.0, -0.3, 0.5), (0.0, 0.0, 0.0))
    inward = EePose((0.0, -0.25, 0.5), (0.0, 0.0, 0.0))
    outward = EePose((0.0, -0.35, 0.5), (0.0, 0.0, 0.0))

    assert workspace_violation_distance(outside, config) == pytest.approx(0.1)
    assert workspace_motion_is_allowed(outside, inward, config)
    assert not workspace_motion_is_allowed(outside, outward, config)
