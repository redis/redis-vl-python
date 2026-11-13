# Docs lens

Check that docstrings, comments, guides and notebooks say what the code actually does, and nothing more. Removing lines is a valid finding.

## Look for

- **Claims that contradict the code:** a docstring describing an old default, a comment explaining a branch that no longer exists, a guide calling a role "least privilege" when it grants more than it needs.
- **Stale guides and notebooks:** pages under `docs/` and notebooks under `docs/user_guide/` that call a changed API, rely on a changed default, or show output that would now differ. Check every usage the diff makes stale, not just the files it touches.
- **Bloat:** comments that restate the code, docstrings longer than the function they describe, and the same explanation repeated in several places when one should link to the other. A comment should explain why, or warn about something non-obvious.
- **Missing docs for new public API:** a new public class, function or parameter with no docstring, or with a docstring that doesn't state its defaults and what it raises.
- **Sphinx breakage:** wrong `currentmodule` or cross-reference targets, and new modules missing from `docs/api/`.
- **Text the change makes untrue.** Grep the docstrings, comments, guides and notebooks for the names the diff touches. Common leftovers: a return description written for the old return type, a comment explaining a caller that no longer exists, a notebook note about a workaround the fix removes, a caveat that a PR merged since has lifted.
- **Public options documented where nobody reads them.** The API reference renders the class and `__init__` docstrings, so a public keyword argument described only on a private method is undocumented.
- **The PR description.** Treat every sentence in it as a claim about the code and check it, in particular what the code did before the change. Check that it leads with the most consequential behaviour change, including one the author found on the way and didn't set out to make.
- **Release Notes.** Every difference from step 6 of `SKILL.md` that a user could notice needs a line: what used to happen, what happens now, and what to do about it. Check each line against a run, not against the intent. Notes that claim more than the change delivers are a finding, and so is a note about a change users never had.

## How to report

Quote the claim that is wrong, or the lines that should go, in a few words, and give the file and line. Don't rewrite prose for style. Report only text that misleads, is out of date, or wastes the reader's time.
