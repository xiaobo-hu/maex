"""加载 RoboMemArena π0.5  primitive 参考。"""

from __future__ import annotations

import json
from importlib.resources import files
from typing import cast

_TASK_IDS_BY_PROMPT = {
    "Pick and place cookies into the basket, then pick and place tomato sauce into the same basket.": 1,
    "Pick and place butter into the basket, then pick and place popcorn into the same basket.": 2,
    "Pick and place cream into the basket, then pick and place chocolate into the same basket.": 3,
    "Open and close all drawers in order to check. Put butter into the drawer that already contains an object.": 4,
    "Open and close all drawers in order to check. Put butter into the empty drawer.": 5,
    "Pour tomato sauce over cookies twice and place the sauce bottle into the bowl drainer.": 6,
    "Pour tomato sauce over the frypan twice and place the sauce bottle into the bowl drainer.": 7,
    "Pick and place chocolate into the frypan, pour tomato sauce over it twice, then place the sauce bottle into the bowl drainer.": 8,
    "Put butter into the frypan, pour tomato sauce over it twice, then place the sauce bottle into the bowl drainer.": 9,
    "Pour wine into the mug twice.": 10,
    "Put cookies into the top drawer and put butter into another drawer.": 11,
    "Put cookies into the middle drawer and then put chocolate into the same drawer.": 12,
    "Put cookies into the middle drawer and then put butter into the same drawer.": 13,
    "Put cookies into the top drawer and put chocolate into another drawer.": 14,
    "Pick and place butter into the frypan, then pour milk over it twice.": 15,
    "Pick milk from the table, pour it into the mug twice, then place the milk container into the bowl drainer.": 16,
    "Put butter into the middle drawer and then put chocolate into the same drawer.": 17,
    "Pick and place chocolate and butter from cabinet1 to cabinet2, respectively.": 18,
    "Pick and place tomato sauce, milk, and orange juice from cabinet1 to cabinet2.": 19,
    "Put cookies into the microwave and then put chocolate into the location where the cookies were placed.": 20,
    "Put butter into the microwave and then put chocolate into the location where the butter was placed.": 21,
    "Pour tomato sauce over cookies twice, then put the cookies into the microwave.": 22,
    "Put cream into the microwave and then put popcorn into the location where the cream was placed.": 23,
    "Put cookies into the microwave and then put popcorn into the location where the cookies were placed.": 24,
    "Pick and place butter and cream from plate1 to plate2, respectively.": 25,
    "Pick and place chocolate and cream from plate1 to plate2, respectively.": 26,
}


def task_primitive_reference(task: str) -> str | None:
    """返回与官方高层任务文本精确匹配的有序 VLA primitive 参考。"""
    task_id = _TASK_IDS_BY_PROMPT.get(task.strip())
    if task_id is None:
        return None

    resource = files(__package__).joinpath("fullvlm_v2_26_memory_tasks.json")
    catalog = cast(dict[str, object], json.loads(resource.read_text(encoding="utf-8")))
    tasks = cast(list[dict[str, object]], catalog["tasks"])
    entry = next(item for item in tasks if item["task_id"] == task_id)
    primitives = cast(list[dict[str, str]], entry["primitive_order"])

    lines = [
        f"RoboMemArena π0.5 primitive reference (task {task_id}, advisory):",
        f"Reference task: {entry['brief_description']}",
        "Ordered checkpoint-era VLA labels:",
    ]
    lines.extend(
        f"{index}. {primitive['label']}"
        for index, primitive in enumerate(primitives, start=1)
    )
    note = entry.get("notes")
    if isinstance(note, str) and note:
        lines.append(f"Dataset note: {note}")
    return "\n".join(lines)


__all__ = ["task_primitive_reference"]
