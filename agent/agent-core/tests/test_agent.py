from __future__ import annotations

import asyncio

import pytest

from agent_core import Agent, AgentOptions, AgentState
from ai import Context, TextContent, ToolResultMessage, create_models
from ai.providers.faux import faux_assistant_message, faux_provider


def create_agent(*responses):  # type: ignore[no-untyped-def]
    models = create_models()
    faux = faux_provider()
    model = faux.get_model()
    assert model is not None
    faux.set_responses(responses)
    models.set_provider(faux.provider)
    state = AgentState(model=model)
    return Agent(AgentOptions(initial_state=state, stream_fn=models.stream_simple)), faux


@pytest.mark.asyncio
async def test_agent_reduces_transcript_and_waits_for_async_subscribers() -> None:
    agent, _ = create_agent(faux_assistant_message("answer"))
    delivered: list[str] = []

    async def listener(event):  # type: ignore[no-untyped-def]
        await asyncio.sleep(0)
        delivered.append(event.type)

    unsubscribe = agent.subscribe(listener)
    await agent.prompt("question")

    assert [message.role for message in agent.state.messages] == ["user", "assistant"]
    assert agent.state.messages[0].content == [TextContent(text="question")]
    assert delivered[-1] == "agent_end"
    assert not agent.state.is_streaming
    await agent.wait_for_idle()

    unsubscribe()
    agent.reset()
    assert agent.state.messages == []


@pytest.mark.asyncio
async def test_agent_rejects_overlapping_prompts() -> None:
    started = asyncio.Event()
    release = asyncio.Event()

    async def delayed(
        context: Context,
        options,
        state,
        model,  # type: ignore[no-untyped-def]
    ):
        del context, options, state, model
        started.set()
        await release.wait()
        return faux_assistant_message("done")

    agent, _ = create_agent(delayed)
    first = asyncio.create_task(agent.prompt("first"))
    await started.wait()

    with pytest.raises(RuntimeError, match="already processing"):
        await agent.prompt("second")

    release.set()
    await first


@pytest.mark.asyncio
async def test_continue_requires_non_assistant_tail_and_uses_existing_transcript() -> None:
    dangling = ToolResultMessage(
        tool_call_id="call-1",
        tool_name="manual",
        content=[TextContent(text="ready")],
    )
    agent, _ = create_agent(faux_assistant_message("continued"))
    agent.state.messages.append(dangling)

    await agent.continue_()

    assert [message.role for message in agent.state.messages] == [
        "toolResult",
        "assistant",
    ]
    with pytest.raises(RuntimeError, match="assistant"):
        await agent.continue_()
