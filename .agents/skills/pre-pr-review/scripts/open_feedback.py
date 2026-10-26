#!/usr/bin/env python3
"""List the review feedback on a PR that still needs a response.

Usage: open_feedback.py <pr-number> [--include-resolved]

Prints, as Markdown:
- inline review threads that aren't resolved, with path, line and every
  comment in the thread. Threads GitHub marks outdated are still listed,
  tagged [outdated], because a later push can move a line without fixing it;
- review summaries that have a body;
- top-level PR comments, skipping the release bot.

Resolution state is only available through GitHub's GraphQL API, which is why
this script exists. Requires the GitHub CLI (gh), authenticated for the
repository.
"""

import argparse
import json
import subprocess
import sys

QUERY = """
query($owner: String!, $name: String!, $number: Int!, $cursor: String) {
  repository(owner: $owner, name: $name) {
    pullRequest(number: $number) {
      reviewThreads(first: 50, after: $cursor) {
        pageInfo { hasNextPage endCursor }
        nodes {
          isResolved
          isOutdated
          path
          line
          originalLine
          comments(first: 50) { nodes { author { login } body url } }
        }
      }
      reviews(first: 100) { nodes { author { login } state body url } }
      comments(first: 100) { nodes { author { login } body url } }
    }
  }
}
"""

SKIP_AUTHORS = {"applied-ai-release-bot"}


def gh_json(*args: str) -> dict:
    result = subprocess.run(["gh", *args], capture_output=True, text=True)
    if result.returncode != 0:
        sys.exit(f"gh {' '.join(args[:2])} failed:\n{result.stderr.strip()}")
    return json.loads(result.stdout)


def first_line(text: str, limit: int = 160) -> str:
    line = next(
        (
            part.strip()
            for part in text.splitlines()
            if part.strip() and not part.lstrip().startswith("<!--")
        ),
        "",
    )
    return line if len(line) <= limit else line[: limit - 1] + "…"


def login(node: dict) -> str:
    return (node.get("author") or {}).get("login", "ghost")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("pr", type=int, help="PR number")
    parser.add_argument(
        "--include-resolved", action="store_true", help="also list resolved threads"
    )
    args = parser.parse_args()

    owner, name = gh_json("repo", "view", "--json", "nameWithOwner")[
        "nameWithOwner"
    ].split("/")
    threads, reviews, comments, cursor = [], [], [], None
    while True:
        call = [
            "api",
            "graphql",
            "-f",
            f"query={QUERY}",
            "-f",
            f"owner={owner}",
            "-f",
            f"name={name}",
            "-F",
            f"number={args.pr}",
        ]
        if cursor:
            call += ["-f", f"cursor={cursor}"]
        pr = gh_json(*call)["data"]["repository"]["pullRequest"]
        page = pr["reviewThreads"]
        threads += page["nodes"]
        reviews, comments = pr["reviews"]["nodes"], pr["comments"]["nodes"]
        if not page["pageInfo"]["hasNextPage"]:
            break
        cursor = page["pageInfo"]["endCursor"]

    unsettled = [t for t in threads if not t["isResolved"]]
    shown = threads if args.include_resolved else unsettled
    print(f"## Review threads ({len(unsettled)} open of {len(threads)})\n")
    for thread in shown:
        state = "".join(
            tag
            for flag, tag in (
                (thread["isResolved"], " [resolved]"),
                (thread["isOutdated"], " [outdated]"),
            )
            if flag
        )
        print(f"- `{thread['path']}:{thread['line'] or thread['originalLine']}`{state}")
        for comment in thread["comments"]["nodes"]:
            print(
                f"  - {login(comment)}: {first_line(comment['body'])} ({comment['url']})"
            )

    summaries = [r for r in reviews if r["body"].strip()]
    print(f"\n## Review summaries ({len(summaries)})\n")
    for review in summaries:
        print(
            f"- {login(review)} ({review['state'].lower()}): {first_line(review['body'])} ({review['url']})"
        )

    top_level = [c for c in comments if login(c) not in SKIP_AUTHORS]
    print(f"\n## Top-level comments ({len(top_level)})\n")
    for comment in top_level:
        print(f"- {login(comment)}: {first_line(comment['body'])} ({comment['url']})")


if __name__ == "__main__":
    main()
