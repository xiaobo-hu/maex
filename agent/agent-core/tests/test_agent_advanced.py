from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field

import pytest

from agent_core import (
    AfterToolCallResult,
    Agent,
    AgentContext,
    AgentLoopConfig,
    AgentLoopTurnUpdate,
    AgentOptions,
    AgentState,
    AgentToolResult,
    BeforeToolCallResult,
    ToolExecutionEndEvent,
    run_agent_loop,
    set_default_stream_fn,
)
from ai import Context, Message, SimpleStreamOptions, TextContent, ToolResultMessage, UserMessage
from ai.providers.faux import faux_assistant_message, faux_provider, faux_tool_call


def create_agent(*responses, **option_updates):  # type: ignore[no-untyped-def]
    models = option_updates.pop("models", None)
    models = models or __import__("ai").create_models()
    faux = faux_provider(chunk_size=2)
    model = faux.get_model()
    assert model is not None
    faux.set_responses(responses)
    models.set_provider(faux.provider)
    options = AgentOptions(
        initial_state=AgentState(model=model),
        stream_fn=models.stream_simple,
        **option_updates,
    )
    return Agent(options), faux


@pytest.mark.asyncio
async def test_transform_dynamic_api_key_and_cancel_event_reach_provider() -> None:
    captured: dict[str, object] = {}

    async def response(context, options, state, model):  # type: ignore[no-untyped-def]
        del state, model
        captured["messages"] = list(context.messages)
        captured["options"] = options
        return faux_assistant_message("done")

    async def transform(messages, cancel_event):  # type: ignore[no-untyped-def]
        assert cancel_event is not None
        return [*messages, UserMessage(content="injected")]

    converted: list[list[Message]] = []

    def convert(messages):  # type: ignore[no-untyped-def]
        converted.append(list(messages))
        return messages

    agent, _ = create_agent(
        response,
        transform_context=transform,
        convert_to_llm=convert,
        get_api_key=lambda provider: f"key-for-{provider}",
        stream_options=SimpleStreamOptions(session_id="session-1"),
    )
    await agent.prompt("hello")

    assert len(converted[0]) == 2
    assert len(captured["messages"]) == 2
    options = captured["options"]
    assert options.api_key == "key-for-faux"
    assert options.session_id == "session-1"
    assert options.cancel_event is not None


@pytest.mark.asyncio
async def test_agent_uses_configured_default_stream_function() -> None:
    models = __import__("ai").create_models()
    faux = faux_provider()
    model = faux.get_model()
    assert model is not None
    faux.set_responses([faux_assistant_message("default")])
    models.set_provider(faux.provider)
    set_default_stream_fn(models.stream_simple)
    try:
        agent = Agent(AgentOptions(initial_state=AgentState(model=model)))
        await agent.prompt("hello")
        assert agent.state.messages[-1].content[0].text == "default"
    finally:
        set_default_stream_fn(None)


@pytest.mark.asyncio
async def test_steering_and_follow_up_queues_use_one_at_a_time() -> None:
    contexts: list[list[str]] = []

    async def response(context, options, state, model):  # type: ignore[no-untyped-def]
        del options, model
        contexts.append([message.role for message in context.messages])
        return faux_assistant_message(f"answer-{state.call_count}")

    agent, faux = create_agent(response, response, response)
    agent.steer(UserMessage(content="steer"))
    agent.follow_up(UserMessage(content="follow-1"))
    agent.follow_up(UserMessage(content="follow-2"))
    assert agent.has_queued_messages()

    await agent.prompt("prompt")

    assert faux.state.call_count == 3, (agent.state.error_message, agent.state.messages)
    assert [message.role for message in agent.state.messages] == [
        "user",
        "user",
        "assistant",
        "user",
        "assistant",
        "user",
        "assistant",
    ]
    assert contexts[0] == ["user", "user"]
    assert not agent.has_queued_messages()


@dataclass
class TimedTool:
    name: str
    delay: float
    terminate: bool = False
    execution_mode: str | None = None
    label: str = "Timed"
    description: str = "timed"
    parameters: dict[str, object] = field(default_factory=lambda: {"type": "object"})
    prepare_arguments = None

    async def execute(self, tool_call_id, params, cancel_event=None, on_update=None):  # type: ignore[no-untyped-def]
        del tool_call_id, params, on_update
        await asyncio.sleep(self.delay)
        return AgentToolResult(
            content=[TextContent(text=self.name)],
            terminate=self.terminate,
        )


def setup_loop(*responses, tools):  # type: ignore[no-untyped-def]
    models = __import__("ai").create_models()
    faux = faux_provider()
    model = faux.get_model()
    assert model is not None
    faux.set_responses(responses)
    models.set_provider(faux.provider)
    return model, AgentContext(tools=tools), models.stream_simple


@pytest.mark.asyncio
async def test_parallel_tool_end_completion_order_but_results_source_order() -> None:
    slow = TimedTool("slow", 0.03)
    fast = TimedTool("fast", 0)
    model, context, stream_fn = setup_loop(
        faux_assistant_message(
            [
                faux_tool_call("slow", {}, tool_call_id="slow-1"),
                faux_tool_call("fast", {}, tool_call_id="fast-1"),
            ],
            stop_reason="toolUse",
        ),
        faux_assistant_message("done"),
        tools=[slow, fast],
    )
    events = []
    messages = await run_agent_loop(
        [UserMessage(content="go")],
        context,
        AgentLoopConfig(model=model, tool_execution="parallel"),
        events.append,
        stream_fn,
    )

    ends = [event.tool_name for event in events if isinstance(event, ToolExecutionEndEvent)]
    results = [message.tool_name for message in messages if isinstance(message, ToolResultMessage)]
    assert ends == ["fast", "slow"]
    assert results == ["slow", "fast"]


@pytest.mark.asyncio
async def test_sequential_override_hooks_and_batch_termination() -> None:
    order: list[str] = []

    @dataclass
    class RecordingTool(TimedTool):
        async def execute(self, tool_call_id, params, cancel_event=None, on_update=None):  # type: ignore[no-untyped-def]
            del tool_call_id, params, cancel_event, on_update
            order.append(f"start:{self.name}")
            await asyncio.sleep(self.delay)
            order.append(f"end:{self.name}")
            return AgentToolResult(content=[TextContent(text=self.name)])

    first = RecordingTool("first", 0.01, execution_mode="sequential")
    second = RecordingTool("second", 0)
    model, context, stream_fn = setup_loop(
        faux_assistant_message(
            [
                faux_tool_call("first", {}, tool_call_id="first-1"),
                faux_tool_call("second", {}, tool_call_id="second-1"),
            ],
            stop_reason="toolUse",
        ),
        tools=[first, second],
    )

    async def after(call_context, cancel_event):  # type: ignore[no-untyped-def]
        del cancel_event
        return AfterToolCallResult(
            content=[TextContent(text=f"final:{call_context.tool_call.name}")],
            terminate=True,
        )

    messages = await run_agent_loop(
        [UserMessage(content="go")],
        context,
        AgentLoopConfig(model=model, tool_execution="parallel", after_tool_call=after),
        lambda event: None,
        stream_fn,
    )

    assert order == ["start:first", "end:first", "start:second", "end:second"]
    results = [message for message in messages if isinstance(message, ToolResultMessage)]
    assert [result.content[0].text for result in results] == ["final:first", "final:second"]


@pytest.mark.asyncio
async def test_before_hook_can_block_and_mutate_validated_args() -> None:
    received: list[dict[str, object]] = []

    @dataclass
    class ArgTool(TimedTool):
        parameters: dict[str, object] = field(
            default_factory=lambda: {
                "type": "object",
                "properties": {"value": {"type": "string"}},
                "required": ["value"],
            }
        )

        async def execute(self, tool_call_id, params, cancel_event=None, on_update=None):  # type: ignore[no-untyped-def]
            del tool_call_id, cancel_event, on_update
            received.append(dict(params))
            return AgentToolResult(content=[TextContent(text=str(params["value"]))])

    tool = ArgTool("arg", 0)
    model, context, stream_fn = setup_loop(
        faux_assistant_message(
            [
                faux_tool_call("arg", {"value": "one"}, tool_call_id="one"),
                faux_tool_call("arg", {"value": "two"}, tool_call_id="two"),
            ],
            stop_reason="toolUse",
        ),
        faux_assistant_message("done"),
        tools=[tool],
    )

    async def before(call_context, cancel_event):  # type: ignore[no-untyped-def]
        del cancel_event
        if call_context.tool_call.id == "one":
            return BeforeToolCallResult(block=True, reason="blocked")
        call_context.args["value"] = "mutated"
        return None

    messages = await run_agent_loop(
        [UserMessage(content="go")],
        context,
        AgentLoopConfig(model=model, before_tool_call=before),
        lambda event: None,
        stream_fn,
    )
    results = [message for message in messages if isinstance(message, ToolResultMessage)]
    assert results[0].is_error and results[0].content[0].text == "blocked"
    assert received == [{"value": "mutated"}]


@pytest.mark.asyncio
async def test_tool_pipeline_runs_prepare_validate_before_execute_after_in_order() -> None:
    order: list[str] = []

    @dataclass
    class PipelineTool(TimedTool):
        parameters: dict[str, object] = field(
            default_factory=lambda: {
                "type": "object",
                "properties": {"items": {"type": "array", "items": {"type": "integer"}}},
                "required": ["items"],
            }
        )

        def prepare_arguments(self, arguments: dict[str, object]) -> dict[str, object]:
            order.append("prepare")
            return {"items": json.loads(str(arguments["items"]))}

        async def execute(self, tool_call_id, params, cancel_event=None, on_update=None):  # type: ignore[no-untyped-def]
            del tool_call_id, cancel_event, on_update
            order.append("execute")
            assert params == {"items": [1, 2]}
            return AgentToolResult(content=[TextContent(text="executed")])

    tool = PipelineTool("pipeline", 0)
    original_arguments = {"items": '["1", 2]'}
    model, context, stream_fn = setup_loop(
        faux_assistant_message(
            faux_tool_call("pipeline", original_arguments, tool_call_id="pipeline-1"),
            stop_reason="toolUse",
        ),
        faux_assistant_message("done"),
        tools=[tool],
    )

    async def before(call_context, cancel_event):  # type: ignore[no-untyped-def]
        del cancel_event
        order.append("before")
        assert call_context.args == {"items": [1, 2]}
        assert call_context.tool_call.arguments == original_arguments
        return None

    async def after(call_context, cancel_event):  # type: ignore[no-untyped-def]
        del cancel_event
        order.append("after")
        assert call_context.args == {"items": [1, 2]}
        return None

    await run_agent_loop(
        [UserMessage(content="go")],
        context,
        AgentLoopConfig(model=model, before_tool_call=before, after_tool_call=after),
        lambda event: None,
        stream_fn,
    )

    assert order == ["prepare", "before", "execute", "after"]
    assert original_arguments == {"items": '["1", 2]'}


@pytest.mark.asyncio
async def test_execute_failure_still_runs_after_hook() -> None:
    seen_error: list[bool] = []

    @dataclass
    class FailingTool(TimedTool):
        async def execute(self, tool_call_id, params, cancel_event=None, on_update=None):  # type: ignore[no-untyped-def]
            del tool_call_id, params, cancel_event, on_update
            raise RuntimeError("execution failed")

    tool = FailingTool("failing", 0)
    model, context, stream_fn = setup_loop(
        faux_assistant_message(
            faux_tool_call("failing", {}, tool_call_id="failing-1"),
            stop_reason="toolUse",
        ),
        faux_assistant_message("done"),
        tools=[tool],
    )

    async def after(call_context, cancel_event):  # type: ignore[no-untyped-def]
        del cancel_event
        seen_error.append(call_context.is_error)
        return AfterToolCallResult(content=[TextContent(text="normalized")])

    messages = await run_agent_loop(
        [UserMessage(content="go")],
        context,
        AgentLoopConfig(model=model, after_tool_call=after),
        lambda event: None,
        stream_fn,
    )

    result = next(message for message in messages if isinstance(message, ToolResultMessage))
    assert seen_error == [True]
    assert result.is_error
    assert result.content == [TextContent(text="normalized")]


@pytest.mark.asyncio
async def test_preflight_failure_skips_execute_and_after_hook() -> None:
    executed = False
    finalized = False

    @dataclass
    class InvalidTool(TimedTool):
        def prepare_arguments(self, arguments: dict[str, object]) -> dict[str, object]:
            del arguments
            raise ValueError("cannot prepare arguments")

        async def execute(self, tool_call_id, params, cancel_event=None, on_update=None):  # type: ignore[no-untyped-def]
            nonlocal executed
            del tool_call_id, params, cancel_event, on_update
            executed = True
            return AgentToolResult()

    tool = InvalidTool("invalid", 0)
    model, context, stream_fn = setup_loop(
        faux_assistant_message(
            faux_tool_call("invalid", {}, tool_call_id="invalid-1"),
            stop_reason="toolUse",
        ),
        faux_assistant_message("done"),
        tools=[tool],
    )

    async def after(call_context, cancel_event):  # type: ignore[no-untyped-def]
        nonlocal finalized
        del call_context, cancel_event
        finalized = True
        return None

    messages = await run_agent_loop(
        [UserMessage(content="go")],
        context,
        AgentLoopConfig(model=model, after_tool_call=after),
        lambda event: None,
        stream_fn,
    )

    result = next(message for message in messages if isinstance(message, ToolResultMessage))
    assert result.is_error
    assert result.content == [TextContent(text="cannot prepare arguments")]
    assert not executed
    assert not finalized


@pytest.mark.asyncio
async def test_after_hook_failure_replaces_tool_result() -> None:
    tool = TimedTool("tool", 0)
    model, context, stream_fn = setup_loop(
        faux_assistant_message(
            faux_tool_call("tool", {}, tool_call_id="tool-1"),
            stop_reason="toolUse",
        ),
        faux_assistant_message("done"),
        tools=[tool],
    )

    async def after(call_context, cancel_event):  # type: ignore[no-untyped-def]
        del call_context, cancel_event
        raise RuntimeError("finalization failed")

    messages = await run_agent_loop(
        [UserMessage(content="go")],
        context,
        AgentLoopConfig(model=model, after_tool_call=after),
        lambda event: None,
        stream_fn,
    )

    result = next(message for message in messages if isinstance(message, ToolResultMessage))
    assert result.is_error
    assert result.content == [TextContent(text="finalization failed")]


@pytest.mark.asyncio
async def test_tool_updates_are_closed_after_execute_returns() -> None:
    late_update = None

    @dataclass
    class UpdatingTool(TimedTool):
        async def execute(self, tool_call_id, params, cancel_event=None, on_update=None):  # type: ignore[no-untyped-def]
            nonlocal late_update
            del tool_call_id, params, cancel_event
            assert on_update is not None
            on_update(AgentToolResult(content=[TextContent(text="working")]))
            late_update = on_update
            return AgentToolResult(content=[TextContent(text="done")])

    tool = UpdatingTool("updating", 0)
    model, context, stream_fn = setup_loop(
        faux_assistant_message(
            faux_tool_call("updating", {}, tool_call_id="updating-1"),
            stop_reason="toolUse",
        ),
        faux_assistant_message("done"),
        tools=[tool],
    )
    events = []

    await run_agent_loop(
        [UserMessage(content="go")],
        context,
        AgentLoopConfig(model=model),
        events.append,
        stream_fn,
    )
    assert late_update is not None
    late_update(AgentToolResult(content=[TextContent(text="late")]))
    await asyncio.sleep(0)

    updates = [event for event in events if event.type == "tool_execution_update"]
    assert len(updates) == 1
    assert updates[0].partial_result.content == [TextContent(text="working")]


@pytest.mark.asyncio
async def test_prepare_next_turn_updates_context_and_thinking_for_next_request() -> None:
    captured: list[tuple[list[str], str | None]] = []

    async def response(context, options, state, model):  # type: ignore[no-untyped-def]
        del model
        captured.append(([message.role for message in context.messages], options.reasoning))
        if state.call_count == 1:
            return faux_assistant_message(
                faux_tool_call("step", {}, tool_call_id="step-1"),
                stop_reason="toolUse",
            )
        return faux_assistant_message("done")

    tool = TimedTool("step", 0)

    async def prepare(turn):  # type: ignore[no-untyped-def]
        updated = AgentContext(
            system_prompt=turn.context.system_prompt,
            messages=[*turn.context.messages, UserMessage(content="prepared")],
            tools=list(turn.context.tools),
        )
        return AgentLoopTurnUpdate(context=updated, thinking_level="high")

    models = __import__("ai").create_models()
    faux = faux_provider()
    model = faux.get_model()
    assert model is not None
    faux.set_responses([response, response])
    models.set_provider(faux.provider)
    await run_agent_loop(
        [UserMessage(content="go")],
        AgentContext(tools=[tool]),
        AgentLoopConfig(model=model, prepare_next_turn=prepare),
        lambda event: None,
        models.stream_simple,
    )

    assert captured[0][1] is None
    assert captured[1][0][-1] == "user"
    assert captured[1][1] == "high"


@pytest.mark.asyncio
async def test_should_stop_after_turn_prevents_another_model_call() -> None:
    tool = TimedTool("stop", 0)
    model, context, stream_fn = setup_loop(
        faux_assistant_message(
            faux_tool_call("stop", {}, tool_call_id="stop-1"),
            stop_reason="toolUse",
        ),
        tools=[tool],
    )
    seen = 0

    async def should_stop(turn):  # type: ignore[no-untyped-def]
        nonlocal seen
        seen += 1
        return bool(turn.tool_results)

    messages = await run_agent_loop(
        [UserMessage(content="go")],
        context,
        AgentLoopConfig(model=model, should_stop_after_turn=should_stop),
        lambda event: None,
        stream_fn,
    )
    assert seen == 1
    assert [message.role for message in messages] == ["user", "assistant", "toolResult"]


@pytest.mark.asyncio
async def test_abort_produces_aborted_assistant_and_listener_signal() -> None:
    started = asyncio.Event()

    async def delayed(context: Context, options, state, model):  # type: ignore[no-untyped-def]
        del context, state, model
        started.set()
        await options.cancel_event.wait()
        return faux_assistant_message("late")

    agent, _ = create_agent(delayed)
    signals: list[asyncio.Event] = []

    def listener(event, signal):  # type: ignore[no-untyped-def]
        del event
        signals.append(signal)

    agent.subscribe(listener)
    task = asyncio.create_task(agent.prompt("stop"))
    await started.wait()
    assert agent.signal is not None
    agent.abort()
    await task

    final = agent.state.messages[-1]
    assert final.role == "assistant"
    assert final.stop_reason == "aborted"
    assert signals and signals[-1].is_set()
    assert not agent.state.is_streaming


@pytest.mark.asyncio
async def test_run_failure_emits_complete_lifecycle() -> None:
    async def transform(messages, cancel_event):  # type: ignore[no-untyped-def]
        del messages, cancel_event
        raise RuntimeError("transform failed")

    agent, _ = create_agent(transform_context=transform)
    event_types: list[str] = []
    agent.subscribe(lambda event: event_types.append(event.type))  # type: ignore[arg-type]

    await agent.prompt("go")

    assert event_types[-4:] == ["message_start", "message_end", "turn_end", "agent_end"]
    assert agent.state.error_message == "transform failed"
    assert agent.state.messages[-1].stop_reason == "error"
