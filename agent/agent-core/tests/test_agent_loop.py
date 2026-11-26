from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import pytest

from agent_core import (
    AgentContext,
    AgentLoopConfig,
    AgentToolResult,
    ToolExecutionEndEvent,
    run_agent_loop,
)
from ai import (
    Message,
    TextContent,
    ToolCall,
    ToolResultMessage,
    UserMessage,
    create_models,
)
from ai.providers.faux import faux_assistant_message, faux_provider, faux_tool_call


@dataclass
class EchoTool:
    name: str = "echo"
    label: str = "Echo"
    description: str = "Echo a required text argument"
    parameters: dict[str, object] = field(
        default_factory=lambda: {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
            "additionalProperties": False,
        }
    )
    prepare_arguments = None
    calls: int = 0

    async def execute(self, tool_call_id, params, on_update=None):  # type: ignore[no-untyped-def]
        del tool_call_id
        self.calls += 1
        if on_update is not None:
            on_update(AgentToolResult(content=[TextContent(text="working")]))
        return AgentToolResult(content=[TextContent(text=str(params["text"]))])


class ExplodingTool(EchoTool):
    async def execute(self, tool_call_id, params, on_update=None):  # type: ignore[no-untyped-def]
        del tool_call_id, params, on_update
        self.calls += 1
        raise RuntimeError("tool exploded")


def setup_loop(*responses, tools=None):  # type: ignore[no-untyped-def]
    models = create_models()
    faux = faux_provider()
    model = faux.get_model()
    assert model is not None
    faux.set_responses(responses)
    models.set_provider(faux.provider)
    config = AgentLoopConfig(model=model)
    return faux, config, AgentContext(tools=list(tools or [])), models.stream_simple


@pytest.mark.asyncio
async def test_plain_text_agent_flow_has_complete_lifecycle() -> None:
    _, config, context, stream_fn = setup_loop(faux_assistant_message("hello"))
    events = []

    messages = await run_agent_loop(
        [UserMessage(content="hi")], context, config, events.append, stream_fn
    )

    assert [message.role for message in messages] == ["user", "assistant"]
    assert events[0].type == "agent_start"
    assert events[-1].type == "agent_end"
    assert [event.type for event in events].count("message_update") > 0


@pytest.mark.asyncio
async def test_user_tool_result_final_reply_closes_the_loop() -> None:
    tool = EchoTool()
    faux, config, context, stream_fn = setup_loop(
        faux_assistant_message(
            faux_tool_call("echo", {"text": "hello"}, tool_call_id="call-1"),
            stop_reason="toolUse",
        ),
        faux_assistant_message("finished"),
        tools=[tool],
    )
    events = []

    messages = await run_agent_loop(
        [UserMessage(content="echo this")], context, config, events.append, stream_fn
    )

    assert [message.role for message in messages] == [
        "user",
        "assistant",
        "toolResult",
        "assistant",
    ]
    tool_result = messages[2]
    assert isinstance(tool_result, ToolResultMessage)
    assert not tool_result.is_error
    assert tool_result.content == [TextContent(text="hello")]
    assert tool.calls == 1
    assert faux.state.call_count == 2
    assert [event.type for event in events].count("tool_execution_update") == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("tool_call", "tools", "expected"),
    [
        (faux_tool_call("missing", {}, tool_call_id="missing-1"), [], "not found"),
        (
            faux_tool_call("echo", {}, tool_call_id="invalid-1"),
            [EchoTool()],
            "Validation failed",
        ),
        (
            faux_tool_call("echo", {"text": "boom"}, tool_call_id="boom-1"),
            [ExplodingTool()],
            "tool exploded",
        ),
    ],
)
async def test_tool_failures_become_error_results(
    tool_call: ToolCall, tools: list[EchoTool], expected: str
) -> None:
    _, config, context, stream_fn = setup_loop(
        faux_assistant_message(tool_call, stop_reason="toolUse"),
        faux_assistant_message("recovered"),
        tools=tools,
    )
    events = []

    messages = await run_agent_loop(
        [UserMessage(content="go")], context, config, events.append, stream_fn
    )

    result = next(message for message in messages if isinstance(message, ToolResultMessage))
    assert result.is_error
    assert expected in result.content[0].text
    end = next(event for event in events if isinstance(event, ToolExecutionEndEvent))
    assert end.is_error


@dataclass(frozen=True)
class StatusMessage:
    role: Literal["status"] = "status"
    text: str = ""


@pytest.mark.asyncio
async def test_application_message_is_owned_outside_core_and_converted_at_boundary() -> None:
    captured: list[list[Message]] = []

    async def response(context, options, state, model):  # type: ignore[no-untyped-def]
        del options, state, model
        captured.append(list(context.messages))
        return faux_assistant_message("done")

    _, config, context, stream_fn = setup_loop(response)

    def convert(messages: list[Message | StatusMessage]) -> list[Message]:
        return [
            UserMessage(content=message.text) if isinstance(message, StatusMessage) else message
            for message in messages
        ]

    config.convert_to_llm = convert
    status = StatusMessage(text="working")
    messages = await run_agent_loop([status], context, config, lambda event: None, stream_fn)

    assert messages[0] is status
    assert isinstance(captured[0][0], UserMessage)
    assert captured[0][0].content == "working"
