<!-- Describe the change, not the work behind it: "This PR adds", not "I added". Leave out test-plan checklists and review history; the diff and the Reviewer Brief carry those. A trivial PR, such as a typo fix or a dependency bump, can replace the other sections with one paragraph, but keeps the AI Assistance section, and the Reviewer Brief if that section calls for one. -->

## Motivation

<!-- Why this change is needed: the problem, how to reproduce it or a link to the issue, and what it unblocks. One paragraph is usually enough. It should make sense to someone who can't see any internal ticket, wiki page or chat thread. -->

## Changes

<!-- What changed and why it works this way. Give each headline change a ### subsection of one or two paragraphs. List minor changes, such as dependency bumps, doc fixes and config, as bullets after them. A before-and-after snippet of a signature or call site often lands faster than prose. -->

<!-- Optional, and most PRs don't need it. To use it, delete the comment markers around it and replace its guidance.

## Notes

Something a reviewer can't read off the diff: a trade-off made on purpose, a deferred decision, a rollout default, a stacked branch. Write one paragraph per note, each a decision with its reason. Don't ask questions here; settle them before opening the PR. If a note only restates Changes, leave it out.

-->

## Release Notes

<!-- One or two sentences for a user deciding whether to upgrade: what changed from their point of view, and what breaks. They're copied into the release notes as they are, so don't write "this PR", don't name a version number, and don't add a line saying nothing breaks. Delete this section for refactors and for test, CI and docs changes. -->

## AI Assistance

<!-- Tick exactly one box and keep this section in every PR. "AI-assisted" means an AI tool wrote or rewrote any of the code, tests, docs or this description; single-line editor autocomplete doesn't count. See CONTRIBUTING.md. -->

- [ ] No AI assistance was used for this PR.
- [ ] This PR is AI-assisted. I have reviewed and understood everything in it, and I take responsibility for it.

<!-- If AI-assisted, say which tool and model, and what it did: for example "Claude Code (claude-opus-5-5): implementation, tests and description". -->

## Reviewer Brief

<!-- Required for AI-assisted PRs and for maintainer PRs: run the pre-PR review described in AGENTS.md and put its Reviewer Brief here. It tells the reviewer where to start, where the change could fail, and what wasn't verified. A PR that needs a brief and has none isn't assigned a reviewer. If you ticked "No AI assistance" and aren't a maintainer, delete this section. -->
