"""将 RoboMemArena 双视角 observation 流式写入评测视频。"""

from __future__ import annotations

import base64
import shutil
import subprocess
from io import BytesIO
from pathlib import Path
from typing import cast

import cv2
import numpy as np
from PIL import Image

from roboagent import ImageFrame, Observation


class RoboMemArenaVideoRecorder:
    """逐帧写入 external 与 wrist MP4，避免缓存完整 episode。"""

    def __init__(self, video_dir: Path, *, task_id: int, fps: int = 10) -> None:
        if fps < 1:
            raise ValueError("video fps must be at least 1")
        self._video_dir = video_dir
        self._task_id = task_id
        self._fps = fps
        self._writers: dict[str, cv2.VideoWriter] = {}
        self._temporary_paths: dict[str, Path] = {}
        self._closed = False

    def publish(self, observation: Observation) -> None:
        """Append the available external and wrist images for one physical state."""
        if self._closed:
            raise RuntimeError("video recorder is already closed")
        for view, frame in (("external", observation.rgb), ("wrist", observation.wrist_rgb)):
            if frame is None:
                continue
            image = _decode_rgb(frame)
            writer = self._writers.get(view)
            if writer is None:
                writer = self._open_writer(view, image)
                self._writers[view] = writer
            writer.write(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    def close(self, *, success: bool) -> dict[str, str]:
        """Finalize both streams and give them official-style episode names."""
        if self._closed:
            return {}
        self._closed = True
        for writer in self._writers.values():
            writer.release()

        outcome = "success" if success else "failure"
        paths: dict[str, str] = {}
        for view, temporary_path in self._temporary_paths.items():
            suffix = "" if view == "external" else "_wrist"
            final_path = self._video_dir / f"task{self._task_id}_{outcome}_ep0{suffix}.mp4"
            _finalize_video(temporary_path, final_path)
            paths[view] = str(final_path)
        return paths

    def _open_writer(self, view: str, image: np.ndarray) -> cv2.VideoWriter:
        self._video_dir.mkdir(parents=True, exist_ok=True)
        path = self._video_dir / f".task{self._task_id}_ep0_{view}.recording.mp4"
        height, width = image.shape[:2]
        writer = cv2.VideoWriter(
            str(path),
            cv2.VideoWriter_fourcc(*"mp4v"),  # type: ignore[attr-defined]
            float(self._fps),
            (width, height),
        )
        if not writer.isOpened():
            writer.release()
            raise RuntimeError(f"Failed to open MP4 writer for {view}: {path}")
        self._temporary_paths[view] = path
        return writer


def _finalize_video(temporary_path: Path, final_path: Path) -> None:
    """把录制文件转成 H.264(faststart),便于直接播放;ffmpeg 不可用或失败时退回 mp4v 原文件。"""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is not None:
        result = subprocess.run(
            [
                ffmpeg,
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(temporary_path),
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "23",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(final_path),
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0 and final_path.is_file():
            temporary_path.unlink(missing_ok=True)
            return
    temporary_path.replace(final_path)


def _decode_rgb(frame: ImageFrame) -> np.ndarray:
    with Image.open(BytesIO(base64.b64decode(frame.data))) as image:
        return cast(np.ndarray, np.asarray(image.convert("RGB")).copy())
