#!/usr/bin/env python3
"""Replace the Reviewer Brief section of a PR description, leaving the rest intact.

Usage: update_brief.py <pr-number> <brief-file> [--dry-run]

The brief file holds the section body, without the "## Reviewer Brief" heading.
The section runs from the heading to the next "## " heading or to Bugbot's
summary block, whichever comes first, so neither is overwritten. An older
"## Review record" heading is replaced as well. If neither heading exists, the
section is inserted before Bugbot's summary, or appended.

Requires the GitHub CLI (gh), authenticated for the repository.
"""

import argparse
import re
import subprocess
import sys
import tempfile

HEADING = "## Reviewer Brief"
SECTION = re.compile(
    r"^## (?:Reviewer Brief|Review record)[ \t]*$", re.MULTILINE | re.IGNORECASE
)
NEXT_SECTION = re.compile(r"^## |^(?:---\s*\n+)?<!-- CURSOR_SUMMARY -->", re.MULTILINE)
BUGBOT_BLOCK = re.compile(r"^(?:---\s*\n+)?<!-- CURSOR_SUMMARY -->", re.MULTILINE)


def gh(*args: str) -> str:
    result = subprocess.run(["gh", *args], capture_output=True, text=True)
    if result.returncode != 0:
        sys.exit(f"gh {' '.join(args)} failed:\n{result.stderr.strip()}")
    return result.stdout


def replace_brief(body: str, brief: str) -> str:
    body = body.replace("\r\n", "\n")
    section = f"{HEADING}\n\n{brief.strip()}\n"
    match = SECTION.search(body)
    if match:
        end = NEXT_SECTION.search(body, match.end())
        tail = body[end.start() :] if end else ""
        return body[: match.start()] + section + ("\n" + tail if tail else "")
    bugbot = BUGBOT_BLOCK.search(body)
    if bugbot:
        head = body[: bugbot.start()].rstrip()
        return f"{head}\n\n{section}\n{body[bugbot.start():]}"
    return f"{body.rstrip()}\n\n{section}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("pr", help="PR number")
    parser.add_argument(
        "brief_file", help="file holding the brief, without its heading"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the new description instead of saving it",
    )
    args = parser.parse_args()

    with open(args.brief_file, encoding="utf-8") as f:
        brief = f.read()
    brief = SECTION.sub("", brief.replace("\r\n", "\n"), count=1)

    body = gh("pr", "view", args.pr, "--json", "body", "--jq", ".body")
    new_body = replace_brief(body, brief)

    if args.dry_run:
        print(new_body)
        return
    with tempfile.NamedTemporaryFile(
        "w", suffix=".md", delete=False, encoding="utf-8"
    ) as f:
        f.write(new_body)
    gh("pr", "edit", args.pr, "--body-file", f.name)
    print(f"Updated the Reviewer Brief on PR {args.pr}.")


if __name__ == "__main__":
    main()
