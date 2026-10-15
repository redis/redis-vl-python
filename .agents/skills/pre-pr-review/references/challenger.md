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

Keep findings that score 7 or more and have a file, a line and a concrete failure or cost. Reject the rest. Each rejection gets a one-line reason, because the reasons go into the Review record and show the approver that the review discriminated.

The challenger may also add a finding it noticed while checking. A finding it adds gets the same scoring, done in a fresh pass.
