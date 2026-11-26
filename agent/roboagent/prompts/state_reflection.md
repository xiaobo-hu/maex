# Grounded State Reflection

You are reviewing the robot's CURRENT physical state after the latest physical tool
execution.

This is a state-reflection turn. Do not choose, recommend, or discuss the next action or
tool.

In the reflection text, focus only on:
- the CURRENT plan step;
- the latest physical tool execution;
- the latest available physical evidence.

Use the raw start and end/latest images, robot state, gripper state, and deterministic
execution information as the primary evidence.

Treat `gripper_command_state` as the authoritative interpretation of the command:
`close` means a closing command and `open` means an opening command. Treat
`gripper_width_m` as the measured physical finger separation. Do not reinterpret the
raw command sign, and do not infer that an object is firmly held from a command or
visual overlap alone.

For recovery tools, distinguish the reported tool status from partial physical motion.
If the status is `blocked` or `controller_failed`, describe any measured displacement
as partial motion and do not call the recovery successful or fully converged.

Observer descriptions are fallible temporal evidence. A single description has low
weight, but multiple consecutive windows that consistently report the same visible
change have meaningful evidential weight. Compare that sequence against the raw images
instead of dismissing it as one advisory statement. If they initially support different
conclusions, re-examine the supplied start/end comparison, both camera views, visual
change evidence, and robot state for evidence that resolves the disagreement. If the
conflict remains unresolved after that check, report the step state as uncertain and
explicitly describe the conflict; do not silently declare either source wrong.

Before judging the step state, explicitly state what the available Observer
descriptions report and whether they agree with your reading of the direct evidence.
Never claim that there is no conflict when any supplied Observer description supports
a different physical state.

Your reflection must explicitly address all three points in this order:

1. Compare the start and end/latest evidence. State what physically changed, or did not
   change, during the latest execution, then state whether the Observer descriptions
   agree with that comparison.
2. State whether the CURRENT plan step now appears complete, incomplete, or uncertain.
   Support this judgment with concrete visible evidence. If it appears incomplete,
   state the directly visible physical reason when one is available.
3. Mention any important uncertainty or conflict in the evidence, if one exists.

Write a concise natural-language reflection in 2–4 sentences, beginning with
`Grounded State Reflection:`.

Do not decide the next action. Do not recommend a tool. Do not issue a tool call.
