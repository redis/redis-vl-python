# Challenger pass

The challenger's job is to remove false positives before they cost anyone time. It starts from the code and the list of findings, not from the reasoning that produced them. If your agent supports isolated sub-tasks, run the challenger as one, giving it only the diff and the findings list. If it doesn't, put the lens passes out of mind: re-open only the diff and the scratch file of findings, and judge each finding from the code alone.

For each finding:

1. Open the file at the cited line and read enough context to judge the claim.
2. Try to build the failure scenario the finding describes. If you can, write it down as an input and the result, or as a short test.
3. Look for anything that already prevents it: a guard elsewhere, a caller that never passes that input, a test that covers it.
4. Score it.

## Scale

| Score | Meaning |
|---|---|
| 9–10 | Demonstrated: the failure is reproduced or follows directly from the code |
| 7–8 | Very likely: the failure follows from the code as written, and nothing prevents it |
| 4–6 | Plausible but unproven: it depends on inputs or states you couldn't confirm can occur |
| 1–3 | Wrong, already handled, out of scope, or a style preference |

## Findings with a cost instead of a failure

A finding that says a test should go, a helper already exists or a simpler design is available has no failing input, and scoring it on the scale above would reject it as a style preference. That is the wrong result: these are what human reviewers send most PRs back for. Score them on their evidence instead.

| Finding | Scores 9–10 when | Scores 1–3 when |
|---|---|---|
| A test proves nothing | It passes against the base code (`references/tests.md`), or you broke the changed line and it still passed | It fails on the base at its own assertion |
| A test is redundant | You can name the existing test, with file and line, that fails for every change this one would catch | The named test misses a branch this one reaches |
| An axis adds no coverage | The changed code has no branch on it: cite the line where the variants meet | The variants reach different lines of the change |
| An existing helper does the job | You can name the helper and show the call that replaces the new code | The helper needs changes that outweigh the saving |
| A behaviour change is undeclared | You ran the input on the base and the branch and the results differ | The description or Release Notes already state it |

The size of a finding is not a reason to reject it. A redundant test costs the reviewer a comment and the author a round trip.

## What to keep

Keep findings that score 7 or more and have a file, a line and a concrete failure or cost. Reject the rest, each with a one-line reason in your scratch notes. If you reject a finding a reasonable person might side with, raise it with the author rather than settling it silently.

The challenger may also add a finding it noticed while checking. A finding it adds gets the same scoring, done in a fresh pass.
