from __future__ import annotations

import base64
from io import BytesIO

import cv2
import numpy as np
from benchmark.robomemarena.client.video_recording import RoboMemArenaVideoRecorder
from PIL import Image

from roboagent import ImageFrame, Observation


def test_video_recorder_streams_both_views_to_official_names(tmp_path) -> None:
    recorder = RoboMemArenaVideoRecorder(tmp_path, task_id=3, fps=10)
    observation = Observation(
        rgb=_frame((255, 0, 0)),
        wrist_rgb=_frame((0, 255, 0)),
    )

    recorder.publish(observation)
    recorder.publish(observation)
    paths = recorder.close(success=True)

    assert paths == {
        "external": str(tmp_path / "task3_success_ep0.mp4"),
        "wrist": str(tmp_path / "task3_success_ep0_wrist.mp4"),
    }
    for path in paths.values():
        capture = cv2.VideoCapture(path)
        assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == 2
        # libx264 输出的容器 fourcc 可能是 avc1/H264/h264,都是 H.264。
        assert int(capture.get(cv2.CAP_PROP_FOURCC)) in {
            cv2.VideoWriter_fourcc(*"avc1"),
            cv2.VideoWriter_fourcc(*"H264"),
            cv2.VideoWriter_fourcc(*"h264"),
        }
        capture.release()


def _frame(color: tuple[int, int, int]) -> ImageFrame:
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    image[:, :] = color
    output = BytesIO()
    Image.fromarray(image).save(output, format="PNG")
    return ImageFrame(data=base64.b64encode(output.getvalue()).decode("ascii"))
