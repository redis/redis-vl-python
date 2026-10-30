# Tests lens

Check that the tests prove the change works, cover the paths that matter, and add nothing that doesn't earn its place. Deleting or merging tests is a valid finding.

## Look for

- **Vacuous tests.** A regression test that passes without the fix. Revert the fix mentally, or actually, and check that the test fails for the right reason, not because of an unrelated error earlier in setup. A test that builds an object which fails before reaching the code under test proves nothing.
- **Missing critical paths.** Error paths, boundary values (empty, one, exactly the limit, one past it), both the sync and async variants, and both hash and JSON storage, where the change touches them.
- **Areas CI can't reach.** `requires_cluster` tests are skipped in CI, and `requires_api_keys` tests only run with keys. If the change is in one of those areas, ask for a hermetic unit test, using mocks or a fake client, that pins the behaviour. Otherwise, make sure the Reviewer Brief lists it under "Not verified".
- **Weakened assertions.** Loosened expectations, added skips or broader `pytest.raises` that make a failing test pass rather than fixing the code.
- **Bloat.** Parametrisation that multiplies cases without covering new behaviour; several tests asserting the same thing; tests of trivial getters or of library code. Also tests that pin incidental details such as exact log text or wire format, which make refactoring expensive.
- **Placement.** Pure logic belongs in `tests/unit/`, and anything needing Redis in `tests/integration/`. Integration tests share a container, so check that new ones clean up their keys and indexes.

## How to report

For a missing test, name the scenario in one line ("async path with an empty result set"). For a test that should go, say what it duplicates or why it proves nothing.
