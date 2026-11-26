"""验证 HiMe 接入层的格式兼容，不修改动作内容或推断任务完成。"""
import xml.etree.ElementTree as ET

import pytest

from benchmark.robomemarena.baselines.hime_clients import normalize_planner_format


@pytest.mark.parametrize('ending', ['', '</summary>'])
def test_summary_wrapping_is_flattened_without_losing_memory(ending):
    raw = '<summary>Observation'
    raw += '<memory_operations><operation><type>QUERY</type><query>drawer</query></operation></memory_operations>'
    raw += '<plan_list>[current] open drawer</plan_list>' + ending
    root = ET.fromstring('<root>' + normalize_planner_format(raw) + '</root>')
    assert root.find('summary').text == 'Observation'
    assert root.find('memory_operations/operation/query').text == 'drawer'
    assert root.find('plan_list').text == '[current] open drawer'


def test_status_translation_is_limited_to_plan_and_preserves_action():
    raw = '<summary>[当前] evidence</summary><memory_operations/><plan_list>[当前] 拿起番茄酱\n[待处理] pour twice\n[已完成] inspect</plan_list>'
    fixed = normalize_planner_format(raw)
    assert '<summary>[当前] evidence</summary>' in fixed
    assert '[current] 拿起番茄酱\n[pending] pour twice\n[done] inspect' in fixed


@pytest.mark.parametrize('raw', [
    '<summary>ok</summary><memory_operations></memory_operations><plan_list>[current] pick</plan_list>',
    '<summary>ok</summary><memory_operations></memory_operations><plan_list></plan_list>',
    '<summary>x<memory_operations><operation></memory_operations><plan_list>[当前] pick</plan_list>',
    'plain response',
])
def test_valid_or_unrepairable_output_unchanged(raw):
    assert normalize_planner_format(raw) == raw
