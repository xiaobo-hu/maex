"""使用现有 LLM Provider 生成意图，使用本地 VLM 核查只读证据。"""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Callable
from typing import TypeVar

from openai import AsyncOpenAI
from pydantic import BaseModel

from ai import (
    AssistantMessageEventStream,
    Context,
    Model,
    SimpleStreamOptions,
    TextContent,
    Tool,
    ToolCall,
    UserMessage,
)
from roboagent.prompts.robomemarena import task_primitive_reference
from roboagent.types import Observation, summarize_robot_state

from .schema import (
    ActiveTask,
    Assessment,
    EvidenceWindow,
    Fact,
    Judgment,
    PickupRepair,
    Plan,
    PlanRequest,
    Predicate,
    RecoveryDecision,
    TreePatch,
)

Output = TypeVar("Output", bound=BaseModel)
Usage = dict[str, int]

PLAN_PROMPT = """You plan sequential robot manipulation goals for MAEX.
Return an ordered plan via the supplied tool. Every step must have one visually testable
effect. The allowed predicates are held (pick up), placed (release at a named destination),
opened, closed, and poured (one tilt toward the destination followed by return upright).
Use the checkpoint-era label verbatim if it matches the task and names the needed location.
Preserve useful pick and place
primitives as separate steps. Use the actual task's objects, order and destinations whenever
the reference uses obsolete names. An instruction for placing must name the destination.
Use the same short object name for all steps for that object. Include opening or closing
when required for access or requested by the task. Checkpoint labels can include cleanup
motions. Use the same name for opening, placement and closing of a container. Name the door explicitly in the
action phrase when working with a hinged appliance, such as 'open the microwave door'.
For 'the location where X was placed'
or 'the same container', inherit the actual earlier placement's destination by setting
destination_from to that earlier step's zero-based index. Leave destination empty and
write the instruction with $destination, such as 'place popcorn into $destination'.
Otherwise, use a concrete destination with its name in the instruction. Never add
another pour. Number pours of the same object and
destination from 1. An inspection task opens and closes each requested drawer in order.
For a drawer selected by its contents, set drawer_selection to the condition and the ordered
list of drawer names, and write $drawer in each late-bound goal role and its instruction.
The old labels for reopening, filling and closing one named drawer are examples. Bind those
three goals to $drawer even if a label names the middle drawer. Inspect each drawer in
the requested order, then reopen $drawer, pick up the requested object, place it into
$drawer, and close $drawer. Use the actual object name in the placement instruction.
Leave drawer_selection null for all other tasks. Existing visible state can inform grounding.
The descriptions and images are inputs only. Plan goals, not facts about future effects."""

EXPANSION_PROMPT = """Complete the local outcome branches for MAEX using the supplied tool.
Only address the listed step indices. Keep their identity, destination, order and action
unchanged. For each step describe a visual landmark or ambiguity that may help diagnose
stalled execution. Choose retry or retreat for a confirmed
deviation. Retreat is only a candidate and requires fresh evidence of a clear corridor.
When the result is uncertain, the executive inspects existing views. Do not propose new
actions, unsupported tools, coordinates, goals or claims about future observations."""

VERIFY_PROMPT = """You are a read-only perception module for a robot. Evaluate each listed
step from the ordered real images, with actual gripper feedback provided as context.
Return checks in the prescribed structure. Only cite exact image IDs in the request.
Return exactly one judgment per listed step, with all supported facets in that judgment.
Use verification_focus, if present, as a question to inspect. Base judgments on the images and measured feedback.
The outcome is always the truth value of the registered predicate for the named goal.
Make your current assessment from the last observation and cite its exact frame IDs
for any contradiction. Earlier images only establish actually observed events or changes.

For every true judgment, cite identity, the visually matched target. For held, give
effect when the target is visibly off its support and held between the fingers. Use
the latest clear grip image taken after this execution started. Confirm the named
target through its shape, appearance and movement across the views. Readable text
is one cue for identity. Match the object actually between the fingers.
When a verified appearance reference is provided, compare the object between the
fingers with that same previously verified target. A matching appearance can
establish identity with the label obscured. Cite current image IDs for all support.
For placed, true requires identity, release and destination. Release means the correct
object has visually separated from the gripper. Destination means the same object is
visibly in/on the exact named receptacle or region after separation. Cite images of
release and destination in that order. Use the latest clear image if it shows both.
If the fingers keep the same occupied width throughout the window, look closely for
the object still in the jaws. The microwave door and lip are outside its cavity.
For a recheck_step with a placed predicate, release was verified earlier. Check the
same item's current location against the verified appearance reference when available.
Occlusion is unknown. Report a contradiction only when you can identify the same item
visibly outside the destination in the latest images.
For a later placement being assessed during pickup, give prior_grasp if an earlier image
shows the same object held. For opened, use the fixed front camera for effect. Compare
the first and last front views for motion of the panel relative to the appliance body,
then cite aperture when the panel is swung aside or the drawer extends far enough to
access the interior. A dark front panel, a hand at the handle, or a narrow gap is
insufficient. For closed, show the last view with the panel against its frame.
When inspect_drawer is set, match the named top, middle or bottom panel with its
vertical level in the fixed front camera. A different panel opening does not satisfy
the goal. Inspect the interior of that drawer in either view. Cite occupied when an
object is visible inside. Cite empty only when the floor is visible and clear. Cite the
same timestep for aperture and the content claim. Leave contents unknown if occluded.
For a poured goal, mark effect at a visibly tipped
container aimed at the correct destination and upright at a later image of the same
container back upright. Check one complete cycle in this execution window. If asked for
two numbered pours, use disjoint image intervals and check the earlier cycle first.
If pickup is also being checked, cite prior_grasp for a visible grasp before the pour.
In this simulator liquid may be invisible. Evaluate the directed tilt and return cycle
of the correct container using its visible pose, even when there is no visible stream.

For the named door handoff, evaluate hand_clear independently of the object outcome.
Include hand_clear only if the latest front view shows the arm and gripper outside
the opening and closing path. For a drawer, look at its sliding path from the open front to the
cabinet. An empty gripper above and clear of that path can be clear even when the
placed item is hidden. Use retreat_clear only when the last view shows an unobstructed
short upward movement and no risk of pulling a held object or handle.
For false, cite contradiction in the last observation showing a clear departure such as
a dropped/wrong object or a visibly failed result. A motion still in progress or a blocked
view is unknown. At an attempt boundary, a clearly closed door or an object still on
its original support is evidence for false. Give no positive conclusion based on commands, expected effects, or
one ambiguous image. Write one short sentence per judgment and use one image reference
per supported facet when possible."""

FIXTURE_CHECK_PROMPT = """Match the named panel to its frame in the fixed front camera.
For a drawer, locate its exact vertical level and check whether that panel translates
outward with the drawer box. For a hinged door, check whether the named panel swings
away from the doorway of that same appliance or cabinet. A visibly accessible
interior behind that exact panel supports an opened result, including when the panel
was moved earlier. For closed, look for the same panel seated against its own frame.
Use the current front view to cite the effect and the current views to cite aperture.
If only a neighboring panel, handle or robot motion is visible, keep the result
unknown unless the named panel is visibly in the opposite state."""

RECOVERY_PROMPT = """You are choosing a recovery for the current RoboArena manipulation intent.
Use the two latest camera views and measured robot state to diagnose why the active
instruction is stalled. Earlier Observer descriptions may be wrong. Compare them to
the actual images and gripper feedback. If earlier images are present, use them to
distinguish a stalled approach from visible progress, and cite only the latest images.

Use gripper_width_m as the physical gap. A width near zero means the fingers are
fully closed and empty. About 0.08 m means the fingers are open. A positive
gripper_command closes the fingers. Never close a fully closed gripper again.

Choose retry to resume the same VLA instruction, open_gripper when the jaws are empty
and their closure blocks a reapproach, or close_gripper when they are visibly aligned
around the named target. While operating a drawer, close_gripper is also available
when the latest front and wrist images both show the open jaws aligned around the
handle of the specified drawer, at its correct height. Cite both latest images.
To open a drawer, choose pull_drawer only if both latest views show a stable grip
on the named drawer's handle and a clear straight, horizontal backward path along
the drawer track. This pulls 5 cm while keeping the grip. Cite both latest views.
During placement,
choose release_gripper only when the two latest views together show that the held
target is inside the intended receptacle and resting on its support surface. Cite
both views for this stationary release. A target above the destination, at the
threshold, or blocked from view still needs placement or a new approach.
Use restage_vla to retract a short distance from a bad local approach when both
latest views show the path behind the wrist is clear. Cite both views. Use
return_to_home to reset the arm pose while preserving
its gripper command when the current approach remains unsuitable.
Recent attempts show the measured start and end widths and displacement. An
executed recovery is complete, with its effect still evaluated separately. If the
same move repeatedly ends with no verified effect, select a different grounded
approach supported by the current images.

Keep the current intent and target. This decision cannot mark a goal verified or
advance the plan. Cite a latest image ID for any action other than retry. State the
direct visual or robot-state reason for the action."""

PICKUP_REPAIR_PROMPT = """The gripper has lost a previously verified grasp. Review the
latest images to decide if the same object needs to be picked up again for the current
goal. Set needs_regrasp=true only when the target is clearly visible outside the
specified destination and is detached from the gripper. Cite the latest image IDs
showing that relation. Set needs_regrasp=false if it is in the destination or its
position cannot be established. After release, the gripper can block the view of the
destination. Set clear_retraction=true if both latest views show a clear 5 cm path
straight back from the gripper, without dragging the object or contacting a fixture.
Cite both current image IDs for retraction. Use the actual images ahead of advisory
descriptions."""


def use_checkpoint_pour_labels(plan: Plan, reference: str) -> Plan:
    """只在对象、目的地及次数匹配时使用 checkpoint 的倾倒指令。"""
    labels = tuple(re.finditer(
        r"^\d+\. (?P<label>pour (?P<target>.+?) (?:into|over) (?P<destination>.+?) (?P<number>[12])(?:st|nd))$",
        reference, flags=re.MULTILINE,
    ))
    goals = []
    for goal in plan.goals:
        if goal.predicate == Predicate.POURED:
            matches = [
                label["label"] for label in labels
                if int(label["number"]) == goal.occurrence
                and label["target"].casefold() == goal.target.casefold()
                and (label["destination"].casefold() == goal.destination.casefold()
                     or label["destination"].casefold().endswith(f" {goal.destination.casefold()}"))
            ]
            if len(matches) == 1:
                goal = goal.model_copy(update={"instruction": matches[0]})
        goals.append(goal)
    return plan.model_copy(update={"goals": tuple(goals)})


class Planner:
    """为全局目标和局部前瞻使用同一个冻结 VLM。"""

    def __init__(
        self,
        model: Model,
        stream: Callable[[Model, Context, SimpleStreamOptions], AssistantMessageEventStream],
        options: SimpleStreamOptions,
        timeout: float,
    ) -> None:
        self.model = model
        self.stream = stream
        self.options = options
        self.timeout = timeout

    async def plan(self, task: str, observation: Observation, feedback: str = "") -> tuple[Plan, Usage]:
        """提交全局约束，初始化有序目标。"""
        reference = task_primitive_reference(task)
        context = f"Task\n{task}"
        if reference:
            context = f"{context}\n\n{reference}"
        if feedback:
            context = f"{context}\n\nRevise the previous proposal to fix these validation errors:\n{feedback}"
        plan, usage = await self.ask("create_intention_tree", Plan, PLAN_PROMPT, context, observation)
        if (reference and len(plan.goals) > 1 and plan.goals[-1].predicate == Predicate.CLOSED
                and not re.search(r"\b(?:close[ds]?|closing|shut(?:ting)?)\b", task, re.IGNORECASE)):
            plan = plan.model_copy(update={"goals": plan.goals[:-1]})
        return use_checkpoint_pour_labels(plan, reference or ""), usage

    async def expand(
        self, task: str, request: PlanRequest, facts: tuple[Fact, ...], observation: Observation,
    ) -> tuple[TreePatch, Usage]:
        """按请求快照补全两步内的后继与证据查询。"""
        snapshot = {
            "task": task,
            "anchor": request.anchor,
            "steps": [
                {"step": step, **goal.model_dump(mode="json")} for step, goal, _ in request.dependencies
            ],
            "confirmed_events": [
                {"step": fact.step, "predicate": fact.goal.predicate, "target": fact.goal.target,
                 "destination": fact.goal.destination} for fact in facts
            ],
        }
        return await self.ask(
            "expand_outcomes", TreePatch, EXPANSION_PROMPT, json.dumps(snapshot), observation,
        )

    async def recover(
        self, task: str, active: ActiveTask, step: int, observation: Observation,
        judgments: tuple[Judgment, ...], hint: str, lost_pickup: int | None,
        recovery_history: list[dict[str, object]], *, before: Observation | None = None,
    ) -> tuple[RecoveryDecision, Usage]:
        """根据最近实测状态选择一次受限的同意图恢复。"""
        context = {
            "task": task, "intent": active.goal.model_dump(mode="json"),
            "attempt": active.attempt, "step": step, "lost_pickup": lost_pickup,
            "recent_observer": [item.model_dump(mode="json") for item in judgments],
            "procedure_hint": hint, "robot_state": summarize_robot_state(observation.robot_state),
            "recent_attempts": recovery_history,
            "image_ids_in_order": [
                f"{view}_{step}" for view, frame in (
                    ("front", observation.rgb), ("wrist", observation.wrist_rgb),
                ) if frame is not None
            ],
        }
        if lost_pickup is not None:
            prompt = PICKUP_REPAIR_PROMPT
            if before is not None:
                context["verified_pickup_images_first"] = len(before.llm_content())
                prompt += (
                    "\nThe earlier images show the verified pickup. Match the same object's appearance "
                    "before using its current location as evidence for regrasp. Cite only the latest images."
                )
            repair, usage = await self.ask(
                "review_lost_pickup", PickupRepair, prompt,
                json.dumps(context), observation, before=before,
            )
            action = "retry"
            if repair.needs_regrasp:
                action = "regrasp"
            elif (
                active.goal.predicate == Predicate.PLACED and active.mode != "restage_vla"
                and repair.clear_retraction and set(repair.refs) == {f"front_{step}", f"wrist_{step}"}
            ):
                action = "restage_vla"
            return RecoveryDecision(
                action=action,
                refs=repair.refs, reason=repair.reason,
            ), usage
        context["earlier_images_first"] = len(before.llm_content()) if before is not None else 0
        return await self.ask("recover_intention", RecoveryDecision, RECOVERY_PROMPT,
                              json.dumps(context), observation, before=before)

    async def ask(
        self, name: str, output: type[Output], prompt: str, context: str, observation: Observation,
        *, before: Observation | None = None,
    ) -> tuple[Output, Usage]:
        """解析单个结构化结果，取消时通知底层 Provider。"""
        cancel = asyncio.Event()
        images = observation.llm_content()
        if before is not None:
            images = [*before.llm_content(), *images]
        stream = self.stream(
            self.model,
            Context(
                system_prompt=prompt,
                messages=[UserMessage(content=[TextContent(text=context), *images])],
                tools=[Tool(name=name, description=output.__doc__ or name,
                            parameters=output.model_json_schema())],
            ),
            self.options.model_copy(update={"cancel_event": cancel, "max_tokens": 1800}),
        )
        try:
            async with asyncio.timeout(self.timeout):
                result = await asyncio.shield(stream.result())
        finally:
            cancel.set()
        calls = [item for item in result.content if isinstance(item, ToolCall)]
        if result.stop_reason != "toolUse" or len(calls) != 1 or calls[0].name != name:
            raise RuntimeError(f"Planner expected {name}, received {result.stop_reason}")
        return output.model_validate(calls[0].arguments), {
            "input": result.usage.input, "output": result.usage.output,
        }


class Observer:
    """识别多视角时间窗口中的事件，返回候选事实和证据引用。"""

    def __init__(self, client: AsyncOpenAI, model: str) -> None:
        self.client = client
        self.model = model

    async def verify(self, window: EvidenceWindow) -> tuple[Assessment, Usage]:
        """使用 vLLM 的 JSON Schema 返回只读判断。"""
        goals = [{"step": step, **goal.model_dump(
            mode="json", include={"predicate", "target", "destination", "occurrence"},
        )} for step, goal in window.goals]
        context = {"goals": goals, "door_handoff": window.clearance, "latest_step": window.end,
                   "execution_started_at_step": window.active.start_step,
                   "at_attempt_boundary": window.at_boundary, "inspect_drawer": window.inspect_drawer,
                   "recheck_step": window.recheck_step}
        if window.focus:
            context["verification_focus"] = window.focus
        content = [{"type": "text", "text": json.dumps(context, ensure_ascii=False)}]
        if any(goal.predicate in {Predicate.OPENED, Predicate.CLOSED} for _, goal in window.goals):
            content.append({"type": "text", "text": FIXTURE_CHECK_PROMPT})
        fixture = window.fixture_reference
        if fixture is not None:
            image = fixture.observation.rgb
            assert image is not None
            content.extend((
                {"type": "text", "text": (
                    f"Fixed front view before this fixture goal at step {fixture.sequence}. "
                    "Use it for panel correspondence. Cite the named current images below."
                )},
                {"type": "image_url", "image_url": {"url": f"data:{image.mime_type};base64,{image.data}"}},
            ))
        reference = window.identity_reference
        if reference is not None:
            reference_text = (
                "Verified appearance reference showing the earlier placement. "
                "Use it to recognize the same item and receptacle in the current frames."
                if window.recheck_step is not None else
                "Verified appearance reference showing the target during an earlier pickup. "
                "Use it to distinguish the same object from similar items. Cite current frames for the result."
            )
            content.append({"type": "text", "text": reference_text})
            for frame in (reference.observation.rgb, reference.observation.wrist_rgb):
                if frame is not None:
                    content.append({"type": "image_url", "image_url": {
                        "url": f"data:{frame.mime_type};base64,{frame.data}",
                    }})
        for frame in window.frames:
            state = summarize_robot_state(frame.observation.robot_state)
            timing = "LATEST" if frame.sequence == window.end else "EARLIER"
            content.append({
                "type": "text",
                "text": (f"{timing} step {frame.sequence}. Actual gripper gap "
                         f"{state.get('gripper_width_m')} m (near zero is shut, 0.08 m is open). "
                         f"Gripper command {state.get('gripper_command_state')}."),
            })
            for view, image in (("front", frame.observation.rgb), ("wrist", frame.observation.wrist_rgb)):
                if image is not None:
                    content.extend((
                        {"type": "text", "text": f"{view}_{frame.sequence}"},
                        {"type": "image_url", "image_url": {
                            "url": f"data:{image.mime_type};base64,{image.data}",
                        }},
                    ))
        result = await self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": VERIFY_PROMPT}, {"role": "user", "content": content}],
            temperature=0,
            max_tokens=700,
            response_format={"type": "json_schema", "json_schema": {
                "name": "maex_assessment", "schema": Assessment.model_json_schema(), "strict": True,
            }},
        )
        choice = result.choices[0]
        if choice.finish_reason != "stop" or choice.message.content is None:
            raise RuntimeError(f"Observer returned {choice.finish_reason}")
        usage = result.usage
        return Assessment.model_validate_json(choice.message.content), {
            "input": usage.prompt_tokens if usage else 0,
            "output": usage.completion_tokens if usage else 0,
        }
