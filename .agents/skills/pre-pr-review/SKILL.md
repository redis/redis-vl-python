---
name: pre-pr-review
description: Review a RedisVL branch before a human sees it, fix what holds up, and write a Reviewer Brief into the PR description so the approver knows where to look. Use it whenever you're about to open a PR, mark one ready for review, or push substantive changes to one, and when you're handling Bugbot or reviewer comments, even if nobody asked for a review by name.
compatibility: Works with any coding agent. Needs git, make, pytest and Python 3. Optional - gh, to read and update the PR; the public Redis Docs MCP at https://redis.io/mcp, for the Redis behaviour lens.
---

# Pre-PR review

A human approves every merge to `main`. This review runs before they see the PR, so the problems it catches never reach them. Its output, the Reviewer Brief, gives them a head start: where to begin reading, where the change could still fail, and what nobody has checked.

The approver can only rely on the brief if it's accurate, so report only what you can tie to the code, and never claim a check you didn't run.

## 1. Scope the change

- Check the starting point. Read the linked issue and every maintainer comment on it, and search open PRs that touch the same files (`gh pr list --search "<path or symbol>"`). If a maintainer has ruled an approach out, or an open PR already covers the work, stop and tell the author. If the issue proposes a fix, the change should either be that fix or say in Notes why it isn't. AGENTS.md asks for this before any code is written; do it now if it was skipped.
- Rebase on the current `origin/main`. A review of a stale branch describes code the reviewer won't see.
- Diff the work against its merge base with `main`: `git diff $(git merge-base origin/main HEAD)` covers commits and uncommitted changes, and `git status` shows untracked files to read as well.
- Read each changed file in full, not just the hunks. Bugs often sit in the unchanged lines a hunk depends on.
- For every changed public function, method or class, read its callers and its sync or async twin (`SearchIndex` and `AsyncSearchIndex`, and so on). Drift between twins is one of this project's most common defects.
- Search for sibling sites. A bug in one expression usually has copies: grep `redisvl/` for the pattern the diff fixes (the same call, the same conditional, the same constant list) and list every match. Fix them all in this PR, or name the ones left out in Notes with the reason. A PR that fixes `api_config.get(...) if api_config else os.getenv(...)` in one vectorizer and leaves it in six others comes back from review.
- Run the new and changed tests against the base code, as `references/tests.md` describes, and keep the output for the tests lens.
- In a stack of PRs you may review the whole stack at once, but keep track of which PR introduced each line. Each PR gets its own brief.

## 2. Choose lenses

Run all six lenses by default. Each catches problems the others miss, and running all of them is the most thorough option.

A lens that finds nothing relevant in the diff has still done its job; say nothing about it beyond the footer. If you need to save tokens, the "Most useful when" column says where each lens pays off. Correctness always runs. The footer of the brief lists the lenses that ran, so the reviewer can see what was covered.

| Lens | File | Most useful when |
|---|---|---|
| Correctness | `references/correctness.md` | Always |
| Security | `references/security.md` | The diff touches `redisvl/mcp/`, `redisvl/query/filter.py`, `redisvl/utils/token_escaper.py`, code that builds query strings or `SCAN` patterns, destructive methods (`delete`, `clear`, drop), `redisvl/redis/connection.py`, `.github/`, or dependencies |
| Design | `references/design.md` | Feature work, new or changed public API or MCP surface, work on deprecated APIs, new vectorizer, reranker or provider modules |
| Tests | `references/tests.md` | Tests are added or changed, or the change sits in an area CI can't exercise (cluster, live provider APIs) |
| Docs | `references/docs.md` | `docs/`, notebooks, docstrings or comments change, or public behaviour changes |
| Redis behaviour | `references/redis.md` | The diff changes how Redis commands or queries are built, relies on a server default, limit or reply shape, or states how Redis behaves |

## 3. Review, one lens at a time

Run one pass per lens. Start each pass from the diff and that lens's file, not from an earlier pass's conclusions, so that one pass's blind spots don't become the next one's.

- If your agent can start isolated sub-tasks (subagents, sub-sessions, parallel workers), give each lens its own, and run them in parallel.
- If it can't, run the lenses one after another, and write each pass's findings to a scratch file before starting the next.

Write each finding in this shape, so the challenger can check it quickly:

```
[lens] path/to/file.py:LINE: claim, in plain words. Failure: the input or state that goes wrong, and what happens. Fix: the smallest change that resolves it.
```

Skip formatting, import order and typing nits, which CI and `make lint` already catch.

## 4. Challenge the findings

Run a separate challenger pass, as described in `references/challenger.md`. Review passes produce false positives, and every one that survives costs someone time. Keep findings that score 7 or more out of 10 and have a file, a line and a concrete failure or cost. A test that proves nothing has a cost, not a failure, so the challenger judges those by the evidence rule in its file instead of looking for a failing input.

Run a second challenge round only if the first round's fixes changed the code materially.

## 5. Fix and re-check

- Fix the kept findings that are within the PR's scope, at the root cause. A real problem outside scope goes to the author as a question, not into the diff.
- Add the narrowest regression test that fails without the fix, or note why none applies. A test you've seen fail without the fix is strong evidence for the brief's "Beyond CI" line.
- Re-run whatever the fixes touched: `make format`, `make check-types`, and the affected tests (`make test ARGS="tests/unit/test_x.py"`, or `pytest` on the specific file).
- Before the PR is opened or marked ready, run `make check` on the rebased branch and see it pass. CI on a fork PR skips the service tests, so a green fork run proves less than a local one.
- Commit the fixes locally. The author pushes.

## 6. List what changes for users

Reviewers find behaviour changes the description never mentions more often than they find bugs. Before writing anything for the reviewer, write the list yourself.

- For each changed public function, name every input whose result, exception or exception order differs from `main`. Include the inputs the change wasn't aimed at: an empty string or `None` that used to raise and now falls back, a value that used to match and now doesn't, an argument error that now loses to a different check.
- Run each one against the base and the branch, and keep the two outputs. A change you only reasoned about goes under "Not verified".
- Each difference is then one of three things: intended, in which case it goes in Release Notes in words a user can act on; unintended, in which case fix it; or a trade-off, in which case take it to the author in step 7.
- Re-read the PR description against the code. Every sentence that states what the code did before, what it does now or who calls it is a claim: check it, and correct or delete the ones that don't hold. Name the most consequential change first.

## 7. Settle open questions with the author

A PR should reach the reviewer with no open questions. Before the PR is opened or marked ready, and before each later rewrite of the brief, take everything the review couldn't settle to the author and get an answer: findings outside the PR's scope, rejections you're unsure about, and trade-offs. You work for the author, and the reviewer shouldn't have to answer questions the author never saw.

Once the author has decided, record a deliberate trade-off that a reviewer might question in the PR's Notes section, as a decision with its reason. Notes say what was decided and why; they don't ask the reviewer anything.

If you can't reach the author, for example because you're running unattended, don't open the PR or mark it ready. Leave it as a draft, or unopened, and put the questions in your report to the author, never in the PR.

## 8. Write the Reviewer Brief

Every PR written with AI assistance carries a brief, and so does every maintainer PR. The PR template's AI Assistance section is where the author declares which applies; tick the AI-assisted box and say which agent did what.

Write the brief following `references/reviewer-brief.md`. The reviewer sees the change as it is now, so the brief points at the current code, not at the problems you fixed.

To put it in the PR description, save the brief to a file and run:

```
python3 .agents/skills/pre-pr-review/scripts/update_brief.py <pr-number> <brief-file>
```

The script replaces only the `## Reviewer Brief` section and keeps the rest of the description, including Bugbot's summary. Add `--dry-run` to print the result without changing the PR. If the PR doesn't exist yet, put the brief under `## Reviewer Brief` in the description you open it with. Without `gh`, give the brief to the author to paste.

## Responding to review

When Bugbot or a person comments on the PR:

1. **List what's still open:**
   ```
   python3 .agents/skills/pre-pr-review/scripts/open_feedback.py <pr-number>
   ```
   It prints unresolved, non-outdated review threads, review summaries and top-level comments, each with its author and location.
2. **Verify each point against the current code.** A reviewer's wording, human or bot, is a claim to check, not a verdict.
3. **Decide its outcome:**
   - `fixed in <sha>`
   - `already fixed`
   - `invalid`, with the evidence
   - `duplicate of <item>`
   - `needs a decision`: ask the author, and reply once they've decided
4. **Check the fix before saying so.** Re-read the exact location in the fixed tree, and search for other instances of the same pattern, before you reply "fixed".
5. **Reply on each thread with its outcome, and bring the brief up to date.** Fixes that change behaviour need a fresh review of the new commits and a rewritten brief, because the reviewer will rely on it. Trivial fixes need only a "Since the last brief" line. Anything that needs a decision goes to the author.

Ask the author before posting replies on the PR, unless they've already said you can.
