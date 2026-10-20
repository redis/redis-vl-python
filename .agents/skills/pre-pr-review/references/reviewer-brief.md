# Reviewer Brief

The Reviewer Brief is the section under `## Reviewer Brief` in the PR description. It's written for one person: the human who will approve the PR. They see the change as it is now, not the history of how it got there. They haven't seen your review, and they want to know where to spend their attention.

The brief isn't a log of your review. Leave out what you fixed, how many findings you raised, and suggestions you rejected. The reviewer never saw those problems, so reading about them costs time and changes nothing they do. The commits and the PR's review threads already hold that history for anyone who wants it.

The brief also has no questions in it, because the PR shouldn't have any open ones. Anything the review couldn't settle, such as a finding outside the PR's scope, a rejected suggestion you're unsure about, or a trade-off, goes to the author before the PR is opened or marked ready. Once they've decided, a deliberate trade-off that a reviewer might question goes in the PR's Notes section, written as a decision with its reason.

## Template

```markdown
**Start here:** `<file or function>`. <Why this is the heart of the change, in one sentence.>

**Where it could fail**
- `<path.py:LINE>`: <what could go wrong, for which input or state, and what the reviewer should check>.

**Not verified:** <what wasn't run or exercised, and why>.

**Beyond CI:** <checks CI can't show, with results>.

**Since the last brief:** <what later commits changed>.

<sub>Reviewed `<short sha>`, fixes in `<short sha>` · <agent> (<model>) · lenses: <lenses that ran></sub>
```

"Start here", "Not verified" and the footer always appear. Leave out "Where it could fail" when nothing in the diff is a credible risk, "Beyond CI" when there's nothing beyond CI to report, and "Since the last brief" until the brief has been updated after a push. An empty heading tells the reviewer nothing.

## What goes in each section

- **Start here.** Name the one file or function the rest of the change hangs off, so the reviewer reads it first. Reviewers give more attention to whatever they read first, so this is the most valuable line in the brief.
- **Where it could fail.** At most three spots in the final diff where a defect would be most likely or most costly. Use where your review found problems as a hint, but describe the risk in the code as it stands. For example, "the cancellation path records a duration; check it also records the error type", not "we fixed a bug here". Mark anything that predates the PR as pre-existing.
- **Not verified.** Anything you reasoned about without running: cluster behaviour, live provider APIs, Redis versions outside the CI matrix, other platforms. This line tells the reviewer where the remaining risk sits.
- **Beyond CI.** Only evidence CI doesn't already show. Examples: a regression test confirmed to fail without the fix, a run on a Python version CI doesn't cover, or a measurement. Passing tests that CI also runs don't belong here.
- **Since the last brief.** One line per later push, so a returning reviewer knows what's new.
- **Footer.** The commit you reviewed, the commits that hold the review's own fixes (leave that part out if there are none), which agent and model ran the review, and the lenses that ran. If you reviewed uncommitted work, write "Reviewed the working tree on `<sha>`" and replace it with the commit once the work is committed.

## Writing for the reviewer

The reviewer should finish the brief with a working picture of the change: what to look at, why, and what could go wrong. Write so that someone who has never seen your review can act on every line.

- **Consequence first, then mechanism.** "A cancelled call counts as a success, because no error type is recorded" tells the reviewer why it matters before telling them how. "Missing `error.type` on `CancelledError`" makes them work that out.
- **Use the reviewer's vocabulary.** Name the functions, files and behaviours they can see in the diff. Leave out lens names, finding numbers, scores and other words from your own process.
- **Define jargon or replace it.** If a term isn't in the codebase or in common use, describe the thing instead.
- **One sentence per item,** with numbers and units where they carry the point.
- **Cut what the diff or CI already says.** Before keeping a line, ask whether a reviewer who hasn't seen your review would do anything differently because of it. If not, delete it.
- **Scale to the change.** Aim for about 150 words on a large PR, and far fewer on a small one. A three-line fix might need only "Start here", "Not verified" and the footer.

## Keeping it honest

- **List only checks you saw run.** If "Beyond CI" claims a run you didn't watch, the reviewer will trust the brief for the wrong reasons.
- **The footer says what the brief covers.** It names the commit you reviewed and the commits that only apply that review's fixes. The reviewer compares those with the PR's head to judge whether the brief still applies, so don't list anything else there.
- **Re-review substantive changes.** After a push that changes behaviour, including fixes for review comments, re-run the review on the new commits and rewrite the brief. A "Since the last brief" line is enough only for pushes a reviewer wouldn't care about, such as a rebase, a typo fix or a comment tweak.
- **One brief per PR, even in a stack.** You may review the whole stack at once, but each PR's brief covers only that PR's diff, so it stands on its own.

## Example

The brief for an OpenTelemetry PR that adds spans and metrics to SemanticCache and the vectorizers:

```markdown
**Start here:** `redisvl/utils/vectorize/_telemetry.py`, `embeddings()`. Every vectorizer's embed call in `base.py` now runs inside it, so it decides what each provider call records.

**Where it could fail**
- `redisvl/utils/vectorize/_telemetry.py:49`: a cancelled or interrupted provider call must record an error type, or dashboards count it as a success.
- Prompt-text capture: only `SPAN_ONLY` and `SPAN_AND_EVENT` may put prompt text into spans. Check that no other value of the capture setting turns it on.

**Not verified:** live provider APIs (the Google backend mapping is tested with a stand-in); cluster mode.

**Beyond CI:** the new regression tests fail against the code before the fixes; an instrumented lookup with no SDK installed costs about 11 µs on an Apple M4 Pro.

<sub>Reviewed `feb3fc9`, fixes in `5a55eec` · Claude Code (claude-opus-5-5) · lenses: correctness, security, design, tests, docs, Redis behaviour</sub>
```
