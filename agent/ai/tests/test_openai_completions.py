from __future__ import annotations

from ai import Context, ImageContent, TextContent, ToolResultMessage
from ai.api.openai_completions import _convert_messages


def test_tool_result_images_become_followup_user_observation() -> None:
    messages = _convert_messages(
        Context(
            messages=[
                ToolResultMessage(
                    toolCallId="tool-1",
                    toolName="observe",
                    content=[
                        TextContent(text='{"status":"observed"}'),
                        ImageContent(data="aW1hZ2U=", mimeType="image/png"),
                    ],
                )
            ]
        )
    )

    assert messages[0] == {
        "role": "tool",
        "content": '{"status":"observed"}',
        "tool_call_id": "tool-1",
    }
    assert messages[1]["role"] == "user"
    assert messages[1]["content"][1]["image_url"]["url"] == "data:image/png;base64,aW1hZ2U="
