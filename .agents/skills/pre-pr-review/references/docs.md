# Docs lens

Check that docstrings, comments, guides and notebooks say what the code actually does, and nothing more. Removing lines is a valid finding.

## Look for

- **Claims that contradict the code:** a docstring describing an old default, a comment explaining a branch that no longer exists, a guide calling a role "least privilege" when it grants more than it needs.
- **Stale guides and notebooks:** pages under `docs/` and notebooks under `docs/user_guide/` that call a changed API, rely on a changed default, or show output that would now differ. Check every usage the diff makes stale, not just the files it touches.
- **Bloat:** comments that restate the code, docstrings longer than the function they describe, and the same explanation repeated in several places when one should link to the other. A comment should explain why, or warn about something non-obvious.
- **Missing docs for new public API:** a new public class, function or parameter with no docstring, or with a docstring that doesn't state its defaults and what it raises.
- **Sphinx breakage:** wrong `currentmodule` or cross-reference targets, and new modules missing from `docs/api/`.
- **The PR description:** check that it names the change's most consequential behaviour change. If public behaviour changes, check that the description has a Release Notes line a user could act on.

## How to report

Quote the claim that is wrong, or the lines that should go, in a few words, and give the file and line. Don't rewrite prose for style. Report only text that misleads, is out of date, or wastes the reader's time.
