"""验证 RoboMemArena 提示适配与固定官方记忆合并契约。"""

from pathlib import Path

import pytest


@pytest.fixture
def official(monkeypatch):
    source = Path(__file__).resolve().parents[2] / ".external/RPent-official"
    if not source.is_dir():
        pytest.skip("requires the pinned official RPent checkout")
    monkeypatch.syspath_prepend(str(source))
    from robots.libero.robot_spec import get_robot_spec
    return get_robot_spec().prompts


@pytest.mark.parametrize("attempts", [0, 1, 5])
def test_explore_prompt_removes_foreign_tasks_and_preserves_workflow(official, attempts):
    from benchmark.robomemarena.baselines.harness_prompts import make_prompt_bundle
    variables = dict(mode="explore", suite="robomemarena", task=2, seed=0,
                     recipe_tag="robomemarena_t2_s0", memory_inbox="/tmp/inbox",
                     memory_dir="/tmp/memory", output_dir="/tmp/output",
                     session_number=1, session_max=1, attempts_per_session=attempts)
    adapted = make_prompt_bundle(official)
    system = adapted.render("system", variables=variables)
    user = adapted.render("user", variables=variables)
    assert "You cannot reset" not in system
    assert "t4 SINGLE-ATTEMPT LEVERS" not in system
    assert "A human merges" not in system
    assert "LIBERO PRO benchmark" not in system
    assert "THIS PASS RUNS ONLY WHEN THE CELL IS SOLVED" in system
    assert (f"allows {attempts} physical episodes" if attempts else
            "no per-session episode-count limit") in user
    assert "_split_frontmatter, _validate" in user
    assert "RLinf-Pi05-LIBERO-130-fullshot-SFT" in system
    assert "pi05_robomemarena" not in system
    assert "EXECUTE AFTER NECESSARY PERCEPTION" not in user
    # 感知和动作规则保持原文，适配不能悄悄换掉官方操作策略。
    before, after = official.system(variables), adapted.system(variables)
    for heading in ("RUNTIME", "RULES (NON-NEGOTIABLE)", "KEY HYPERPARAMETERS"):
        assert before[heading] == after[heading]


def test_eval_retains_single_attempt_and_no_distillation(official):
    from benchmark.robomemarena.baselines.harness_prompts import make_prompt_bundle
    bundle = make_prompt_bundle(official)
    variables = dict(mode="eval", memory_profile="local", suite="robomemarena",
                     task=2, seed=50, recipe_tag="robomemarena_t2_s50",
                     reference_tag="robomemarena_t2_s0", memory_dir="/tmp/memory",
                     output_dir="/tmp/output")
    assert "SINGLE-ATTEMPT" in bundle.render("system", variables=variables)
    assert "MEMORY FILE CONTRACT" not in bundle.render("user", variables=variables)
    assert "EXECUTE AFTER NECESSARY PERCEPTION" not in bundle.render("user", variables=variables)


def test_official_merge_rejects_missing_header_then_publishes_valid_templates(official, tmp_path):
    from benchmark.robomemarena.baselines.harness_prompts import MEMORY_FORMAT
    from rpent.memory.manager import MemoryManager
    from rpent.prompt.utils import format_prompt
    cell = "robomemarena_t2_s0"
    root = tmp_path / "memory"
    inbox = root / "_internal/inbox" / cell
    inbox.mkdir(parents=True)
    manager = MemoryManager(root)
    bad = inbox / "global_draft.md"
    bad.write_text("# Missing header\nDiagnostic fixture, not learned experience.\n")
    assert manager.merge_memory(cell_tag=cell, run_state_dir=tmp_path, solved=False)["skipped"]
    bad.unlink()
    rendered = format_prompt(MEMORY_FORMAT, variables=dict(
        task=2, seed=0, recipe_tag=cell, memory_inbox=str(inbox)))
    blocks = rendered.split("\n---\n")
    for scope, metadata in zip(("suite", "global"), (blocks[1], blocks[3])):
        (inbox / f"{scope}_draft.md").write_text(
            "---\n" + metadata + "\n---\nDiagnostic fixture only.\n"
        )
    result = manager.merge_memory(cell_tag=cell, run_state_dir=tmp_path, solved=True)
    assert (result["suite"], result["global"], result["skipped"]) == (1, 1, [])
    assert manager.validate() == []
    assert "Global" in (root / "MEMORY.md").read_text()
    assert "Suite" in (root / "MEMORY.md").read_text()

