"""Guard rails for the operating docs: the zero-spend rule and the T14 data-factory brief
are load-bearing while there are no cameras and no budget. Fail loudly if an edit drops them."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_agents_md_has_zero_spend_rule_12():
    text = (ROOT / "AGENTS.md").read_text()
    rule = re.search(r"^12\. \*\*Zero-spend\.\*\*(.+?)(?=^\d+\. |^## )", text, re.M | re.S)
    assert rule, "AGENTS.md must keep rule 12 (**Zero-spend.**)"
    body = " ".join(rule.group(1).split())
    for phrase in ("No paid datasets", "paid APIs", "cloud GPUs", "written OK"):
        assert phrase in body, f"rule 12 lost {phrase!r}"


def test_t14_data_factory_brief_exists_and_is_indexed():
    brief = ROOT / "tasks" / "T14-data-factory.md"
    assert brief.is_file(), "tasks/T14-data-factory.md is missing"
    text = brief.read_text()
    assert text.startswith("# T14"), "T14 brief lost its title"
    assert "Zero spend" in text, "T14 brief lost its zero-spend rule"
    assert "## Acceptance" in text
    index = (ROOT / "tasks" / "README.md").read_text()
    assert "(T14-data-factory.md)" in index, "T14 missing from tasks/README.md"
