"""把物理工具的原始图像转换为供状态反思和决策使用的视觉证据。"""

from __future__ import annotations

import base64
import json
from collections.abc import Sequence
from importlib import import_module

from ai import ImageContent, TextContent

from .types import ImageFrame


def sanitize_reflection_content(
    content: Sequence[TextContent | ImageContent],
) -> list[TextContent | ImageContent]:
    """仅从 Reflection 可见文本中删除计划和执行边界字段。"""
    return [
        _reflection_header(item) if isinstance(item, TextContent) else item
        for item in content
    ]


def build_reflection_visual_evidence(
    content: Sequence[TextContent | ImageContent],
) -> list[TextContent | ImageContent]:
    """保留结构化诊断，并压缩物理工具双视角起点、终点和差分高亮。"""
    reflection_content = sanitize_reflection_content(content)

    labels = _physical_observation_labels(content)
    if labels is None:
        return reflection_content
    start_label, end_label = labels
    start_views = _images_after(content, start_label)
    end_views = _images_after(content, end_label)
    if not start_views or not end_views:
        return reflection_content

    adapted: list[TextContent | ImageContent] = []
    if reflection_content and isinstance(reflection_content[0], TextContent):
        adapted.append(reflection_content[0])
    adapted.extend(
        [
            TextContent(
                text=(
                    "Visual Evidence Adapter output. Compare each labeled camera view "
                    "between START and END of the latest physical tool execution. Each "
                    "highlight marks only pixel-level change and does not imply task "
                    "success."
                )
            ),
        ]
    )
    view_pairs = [
        ("external", start_views[0], end_views[0]),
        *(
            [("wrist", start_views[1], end_views[1])]
            if len(start_views) > 1 and len(end_views) > 1
            else []
        ),
    ]
    for name, start, end in view_pairs:
        adapted.extend(
            [
                TextContent(
                    text=f"START {name} image (before the latest physical tool execution)."
                ),
                start,
            ]
        )
        highlight = _change_highlight(start, end)
        if highlight is not None:
            adapted.extend(
                [
                    TextContent(
                        text=(
                            f"CHANGE HIGHLIGHT over the END {name} image. Red overlays "
                            "and boxes mark regions with pixel-level change between "
                            "the labeled START and END images from this camera."
                        )
                    ),
                    highlight,
                ]
            )
        adapted.extend(
            [
                TextContent(
                    text=f"END {name} image (after the latest physical tool execution)."
                ),
                end,
            ]
        )
    adapted.extend(
        item
        for item in reflection_content
        if isinstance(item, TextContent)
        and item.text.startswith("Observer advisory")
    )
    return adapted


def build_decision_visual_evidence(
    original: Sequence[TextContent | ImageContent],
    reflection_evidence: Sequence[TextContent | ImageContent],
) -> list[TextContent | ImageContent]:
    """复用同一组差分图，恢复完整结果头并把 Observer 文本放在最后。"""
    content = [
        item
        for item in reflection_evidence
        if not (
            isinstance(item, TextContent)
            and item.text.startswith("Observer advisory")
        )
    ]
    if original and content and isinstance(original[0], TextContent):
        content[0] = original[0]
    content.extend(
        item
        for item in original
        if isinstance(item, TextContent)
        and item.text.startswith("Observer advisory")
    )
    return content


def _physical_observation_labels(
    content: Sequence[TextContent | ImageContent],
) -> tuple[str, str] | None:
    pairs = (
        ("VLA call start observation", "VLA call end/latest observation"),
        ("RESTAGE START observation", "RESTAGE END/latest observation"),
        (
            "RETURN-TO-HOME START observation",
            "RETURN-TO-HOME END/latest observation",
        ),
    )
    texts = [item.text for item in content if isinstance(item, TextContent)]
    return next(
        (
            pair
            for pair in pairs
            if any(text.startswith(pair[0]) for text in texts)
            and any(text.startswith(pair[1]) for text in texts)
        ),
        None,
    )


def _reflection_header(item: TextContent) -> TextContent:
    """从 VLA 结果中仅保留反思所需的物理事实，避免执行边界锚定判断。"""
    try:
        payload = json.loads(item.text)
    except (json.JSONDecodeError, TypeError):
        return item
    if not isinstance(payload, dict):
        return item

    current_step = payload.get("current_step")
    execution = payload.get("execution_diagnostics")
    if not isinstance(current_step, dict) or not isinstance(execution, dict):
        sanitized = _without_reflection_anchors(payload, preserve_status=True)
        if sanitized == payload:
            return item
        return TextContent(text=json.dumps(sanitized, ensure_ascii=False, default=str))

    header = {
        "current_step": {
            key: current_step[key]
            for key in ("id", "instruction")
            if key in current_step
        },
        "gripper_command_convention": (
            "gripper_command > 0 means close; gripper_command < 0 means open"
        ),
        "execution_diagnostics": {
            key: execution.get(key) for key in ("start", "gripper_events", "end")
        },
    }
    return TextContent(text=json.dumps(header, ensure_ascii=False, default=str))


def _without_reflection_anchors(
    value: object,
    *,
    preserve_status: bool = False,
) -> object:
    if isinstance(value, dict):
        return {
            key: _without_reflection_anchors(item)
            for key, item in value.items()
            if key not in {"stop_reason", "chunks_used", "plan_progress"}
            and (key != "status" or preserve_status)
        }
    if isinstance(value, list):
        return [_without_reflection_anchors(item) for item in value]
    return value


def _images_after(
    content: Sequence[TextContent | ImageContent],
    label_prefix: str,
) -> list[ImageContent]:
    for index, item in enumerate(content):
        if not isinstance(item, TextContent) or not item.text.startswith(label_prefix):
            continue
        images: list[ImageContent] = []
        for candidate in content[index + 1 :]:
            if isinstance(candidate, TextContent):
                break
            if isinstance(candidate, ImageContent):
                images.append(candidate)
        return images
    return []


def build_change_highlight(
    start_frame: ImageFrame,
    end_frame: ImageFrame,
) -> ImageFrame | None:
    """在 END 图上高亮 START→END 的像素级变化区域,返回标注后的 ImageFrame。"""
    try:
        cv2 = import_module("cv2")
        np = import_module("numpy")
    except ImportError:
        # OpenCV 属于机器人 benchmark 可选依赖；调用方仍可使用 START/END 原图。
        return None

    try:
        start_image = cv2.imdecode(
            np.frombuffer(base64.b64decode(start_frame.data), dtype=np.uint8),
            cv2.IMREAD_COLOR,
        )
        end_image = cv2.imdecode(
            np.frombuffer(base64.b64decode(end_frame.data), dtype=np.uint8),
            cv2.IMREAD_COLOR,
        )
    except (ValueError, TypeError):
        return None
    if start_image is None or end_image is None or start_image.shape != end_image.shape:
        return None

    difference = cv2.absdiff(start_image, end_image)
    gray = cv2.cvtColor(difference, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 25, 255, cv2.THRESH_BINARY)
    kernel = np.ones((5, 5), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    highlighted = end_image.copy()
    red = np.zeros_like(highlighted)
    red[:, :] = (0, 0, 255)
    overlay = cv2.addWeighted(highlighted, 0.55, red, 0.45, 0.0)
    highlighted[mask > 0] = overlay[mask > 0]

    minimum_area = max(64.0, highlighted.shape[0] * highlighted.shape[1] * 0.001)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if cv2.contourArea(contour) < minimum_area:
            continue
        x, y, width, height = cv2.boundingRect(contour)
        cv2.rectangle(highlighted, (x, y), (x + width, y + height), (0, 0, 255), 2)

    encoded, png = cv2.imencode(".png", highlighted)
    if not encoded:
        return None
    return ImageFrame(
        data=base64.b64encode(png.tobytes()).decode("ascii"),
        mime_type="image/png",
    )


def _change_highlight(start: ImageContent, end: ImageContent) -> ImageContent | None:
    """ImageContent 适配层,复用 build_change_highlight 的差分核心。"""
    frame = build_change_highlight(
        ImageFrame(data=start.data, mime_type=start.mime_type),
        ImageFrame(data=end.data, mime_type=end.mime_type),
    )
    if frame is None:
        return None
    return ImageContent(data=frame.data, mime_type=frame.mime_type)


__all__ = [
    "build_change_highlight",
    "build_decision_visual_evidence",
    "build_reflection_visual_evidence",
    "sanitize_reflection_content",
]
