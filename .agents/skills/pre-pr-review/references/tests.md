# Tests lens

Check that the tests prove the change works, cover the paths that matter, and add nothing that doesn't earn its place. Deleting or merging tests is a valid finding, and in this repository it's the most common one: reviewers ask for tests to be cut on most PRs, far more often than they ask for tests to be added.

## Start from the base run

Run the branch's tests against the code it branched from, before reading the tests. Don't revert the fix in your head instead. That shortcut is how a 60-case matrix with six cases that fail without the fix reaches a reviewer.

Use a separate worktree, so the working tree is never touched, and put the base's library under the branch's tests:

```
base=$(git merge-base origin/main HEAD)
git worktree add --detach "$TMPDIR/base-run" HEAD
git -C "$TMPDIR/base-run" restore --source="$base" -- redisvl
```

Then, from inside `$TMPDIR/base-run`, run `PYTHONPATH=. python -m pytest <changed test files>`, adding `--noconftest` for unit tests that need no Redis. `PYTHONPATH=.` matters: without it, an editable install can import `redisvl` from your main checkout, and every test passes for the wrong reason. Remove the worktree afterwards with `git worktree remove --force "$TMPDIR/base-run"`.

Every test the branch added that passes there needs one of three outcomes:

- **Delete it,** when it covers code the change doesn't touch, or repeats a test that already exists.
- **Fold it into an existing test,** when it adds one assertion or one parameter to something already covered.
- **Keep it, and say why in a comment or the test name,** when it pins behaviour the change must not break, such as "a key in `api_config` still wins over the environment". One such test per behaviour is enough.

A file that fails at collection on the base proves nothing either way. Check its tests by hand: revert the source change in a scratch copy and see each one fail at its own assertion, not in setup.

## Before adding a test

- **Extend before adding.** Search `tests/` for a test of the same function. A new parameter on an existing `parametrize`, or one more assertion in an existing test, beats a new test, and a new test in the existing file beats a new file.
- **Add an axis only where the code branches on it.** Don't run sync and async variants when the async method delegates to the sync one, hash and JSON when the code never reads the storage type, or both index classes when the method lives on their shared base. Say which branch each parameter reaches; if two reach the same one, drop one.
- **Put it at the lowest level that reaches the changed lines.** A change confined to one pure function needs a unit test of that function, not an integration matrix. The reverse also holds: a unit test that pins today's wire reply as a literal will keep passing when the client changes shape, so where an integration test already runs that configuration, add a parameter to it.
- **Reuse the fixtures and helpers already there.** Check `conftest.py` and the neighbouring tests for a fixture, a factory or an autospecced mock before writing scaffolding.

## Look for

- **Vacuous tests.** Anything the base run lists that has no reason to stay. Also a test that builds an object which fails before reaching the code under test, and one that only checks `isinstance` or an attribute name.
- **Tests a wrong implementation would pass.** A test value equal to the library's default, so ignoring the caller's value still passes: use a non-default. A default path that no test takes, such as a constructor never called without its optional argument. A one-sided bound where `pytest.approx` would catch an error in either direction. A plain `MagicMock` where `create_autospec` would catch a renamed method. An async test seeded through the sync method, so the async write path never runs. To check, break the changed line in the most obvious way and see whether anything fails.
- **Missing critical paths.** Error paths, boundary values (empty, one, exactly the limit, one past it), and each sync or async variant and storage type the change does branch on. Every method the change decorates or rewrites should be called by some test.
- **Tests of private details.** Calls counted on a private method, a private class imported, handlers counted, exact log text or wire format. Test through the public entry point that callers use, so the test survives a restructuring.
- **Tests that certify behaviour nobody wants to promise.** A side effect of the current implementation, or a value the server rejects in practice, pinned as if it were a contract.
- **Areas CI can't reach.** `requires_cluster` tests are skipped in CI, and `requires_api_keys` tests only run with keys. If the change is in one of those areas, ask for a hermetic unit test, using mocks or a fake client, that pins the behaviour. Otherwise, make sure the Reviewer Brief lists it under "Not verified".
- **Weakened assertions.** Loosened expectations, added skips or broader `pytest.raises` that make a failing test pass rather than fixing the code.
- **Dead scaffolding.** A mock return value that an `assert_not_called` makes unreachable, an unused parametrize column, a hand-written stand-in class where `types.SimpleNamespace` does the job, a skip condition copied four times where one module-level mark would do.
- **Placement.** Pure logic belongs in `tests/unit/`, and anything needing Redis in `tests/integration/`, in the file that already covers that class. Integration tests share a container, so check that new ones clean up their keys and indexes.

## How to report

For a missing test, name the scenario in one line ("async path with an empty result set"). For a test that should go, give the evidence: "passes on the base", or the name of the test that already covers it, or the mutation that leaves the suite green. That evidence is what the challenger keeps the finding on.
