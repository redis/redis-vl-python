"""Tests for the pre-PR review skill's helper scripts, which agents run against real PRs."""

import importlib.util
from pathlib import Path

SCRIPT = (
    Path(__file__).resolve().parents[2]
    / ".agents/skills/pre-pr-review/scripts/update_brief.py"
)
_spec = importlib.util.spec_from_file_location("update_brief", SCRIPT)
update_brief = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(update_brief)

BUGBOT = "<!-- CURSOR_SUMMARY -->\n---\n\n> [!NOTE]\n> bot summary\n<!-- /CURSOR_SUMMARY -->\n"


def test_replaces_only_the_brief_and_keeps_bugbot_summary():
    body = f"## Motivation\n\nWhy.\n\n## Reviewer Brief\n\nOld brief.\n\n{BUGBOT}"

    result = update_brief.replace_brief(body, "New brief.")

    assert (
        result
        == f"## Motivation\n\nWhy.\n\n## Reviewer Brief\n\nNew brief.\n\n{BUGBOT}"
    )


def test_windows_line_endings_do_not_duplicate_the_section():
    body = "## Motivation\r\n\r\nWhy.\r\n\r\n## Reviewer Brief\r\n\r\nOld brief.\r\n"

    result = update_brief.replace_brief(body, "New brief.")

    assert result.count("## Reviewer Brief") == 1
    assert "Old brief." not in result


def test_replaces_the_older_review_record_heading():
    body = "## Changes\n\nWhat.\n\n## Review record\n\nOld record.\n\n## Notes\n\nKeep me.\n"

    result = update_brief.replace_brief(body, "New brief.")

    assert "## Review record" not in result
    assert "Old record." not in result
    assert result.endswith("## Reviewer Brief\n\nNew brief.\n\n## Notes\n\nKeep me.\n")


def test_inserts_before_bugbot_summary_when_no_section_exists():
    body = f"## Motivation\n\nWhy.\n\n{BUGBOT}"

    result = update_brief.replace_brief(body, "New brief.")

    assert result.index("## Reviewer Brief") < result.index("<!-- CURSOR_SUMMARY -->")
    assert result.endswith(BUGBOT)
