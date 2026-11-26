"""基线组合层：适配官方提示的环境身份、探索预算和记忆文件契约。"""

from __future__ import annotations

import shlex
import sys

PROMPT_ADAPTATION = "robomemarena-v3-official-libero-policy"


ENVIRONMENT = """The environment is RoboMemArena, not LIBERO PRO. The inherited tool/API
names remain unchanged: state.libero_terminated is RoboMemArena's official task
success, not pi0_pick's primitive success. Truncation is not success.
The VLA is the official frozen RLinf-Pi05-LIBERO-130-fullshot-SFT checkpoint.
The static RoboMemArena reference describes tasks, not this policy's training coverage. Use the current
task_language and its shared static primitive reference; task numbers, object
identities and learned strategies from LIBERO do not identify this task.
The copied guides document the tool interfaces; LIBERO scene examples are not
evidence about the current scene. Ground actions in current observations.
Respect the order in the actual task instruction. One successful trajectory
does not establish that RoboMemArena ignores order or temporal constraints.
"""

MEMORY_FORMAT = """Before finish on a SOLVED cell, follow the official DISTIL pass and
write each suite/global draft as a separate Markdown file with YAML frontmatter
starting at the FIRST byte (no Markdown heading or code fence before ---).
Use real evidence from this cell, never invent a lesson to fill a template.
If no cross-task lesson is supported, explicitly report that and omit a global
draft. Failure-only observations stay in wip. Published memory is read-only;
the official runner automatically validates and merges inbox drafts after finish.

Suite frontmatter example (replace the task_language with the actual instruction):
---
id: suite_robomemarena_task_t{{task}}
scope: suite
suite: robomemarena
regime: task
task_id: {{task}}
task_language: 'REPLACE with the actual task instruction'
evidence:
  cells: [{{recipe_tag}}]
  attempts: 1
  solved_seeds: [{{seed}}]
  failed_seeds: []
confidence: single-shot
related: []
---
Use the actual attempt count and observed seeds. Here `regime: task` is a
schema label for the RoboMemArena task family, not a claim of LIBERO perturbations.
Write the official suite body under this frontmatter to
{{memory_inbox}}/suite_{{recipe_tag}}_draft.md.

Global frontmatter example (replace all lesson fields with supported content):
---
id: replace-with-lesson-slug
scope: global
kind: perception
title: 'REPLACE with one supported lesson'
applies_when: 'REPLACE with the observed applicability conditions'
evidence:
  cells: [{{recipe_tag}}]
  attempts: 1
  solved_seeds: [{{seed}}]
  failed_seeds: []
confidence: single-shot
related: []
---
Choose kind from primitive|perception|strategy|failure; one lesson per file,
named new_global_<kind>_<slug>.md. Use the official Why / How to apply /
Falsify / Related body. A single successful seed is not verified generalization.

Before finish, use Bash to run this READ-ONLY check with the pinned official
parser and validator. Repair your own drafts on error and rerun the check.
Do not call merge_memory yourself or change the validator:
"""


def make_prompt_bundle(original):
    """保留官方提示树，仅替换跨环境不适用的内容并明确已有格式要求。"""
    from robots.libero.prompts import explore
    from rpent.prompt.utils import Numbered
    from rpent.robots.prompt_bundle import PromptBundle

    def system(variables=None):
        variables = variables or {}
        tree = dict(original.system(variables))
        for heading in tuple(tree):
            if heading.startswith("PROVEN LEVERS"):
                del tree[heading]
        role = "ROLE AND MODE" if variables.get("mode") == "explore" else "ROLE AND EVALUATION"
        tree[role] = tree[role].replace("LIBERO PRO benchmark", "RoboMemArena benchmark")
        if variables.get("mode") == "explore":
            steps = list(explore.WORKFLOW_STEPS)
            steps[-1] = steps[-1].replace(
                "A human merges the\ninbox later.",
                "The official runner validates and automatically merges the inbox after finish.",
            )
            tree["WORKFLOW"] = Numbered(steps)
        return {"ROBOMEMARENA ADAPTATION": ENVIRONMENT, **tree}

    def user(variables=None):
        variables = variables or {}
        tree = dict(original.user(variables))
        if variables.get("mode") == "explore":
            tree["EXPLORATION BUDGET"] = (
                "This session allows {{attempts_per_session}} physical episodes in total, "
                "including the initial episode. Reset is allowed only while the official "
                "toolkit has attempts remaining; one episode means zero additional resets. "
                "This budget does not make exploration an evaluation run."
                if int(variables["attempts_per_session"]) > 0 else
                "The official toolkit has no per-session episode-count limit in this "
                "configuration. Reset remains subject to the run's turn/time budgets."
            )
            inbox = shlex.quote(str(variables["memory_inbox"]))
            tree["MEMORY FILE CONTRACT — CHECK BEFORE FINISH"] = MEMORY_FORMAT + (
                f"\n{shlex.quote(sys.executable)} - {inbox} <<'PY'\n"
                "import sys\nfrom pathlib import Path\n"
                "from rpent.memory.manager import _split_frontmatter, _validate\n"
                "for path in sorted(Path(sys.argv[1]).glob('*.md')):\n"
                "    metadata, body = _split_frontmatter(path)\n"
                "    _validate(metadata)\n"
                "    print('VALID', path.name)\nPY\n"
            )
        return tree

    return PromptBundle(system=system, user=user)
