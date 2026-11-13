# Design lens

Ask whether the change is a lasting solution at the right layer, or a patch that the next change will have to undo. In this repository, this lens more often produces a decision than a defect: for example, "remove the deprecated API instead of patching it".

## Look for

- **A decision already made.** The linked issue, its maintainer comments and the open PRs on the same files. A change that builds what a maintainer declined, or that duplicates an open PR, is the wrong change however well it's written. Where the issue proposes a fix, a larger one needs a reason the smaller one fails.
- **Something that already does the job.** Before the diff adds a helper, a stored copy of state or a second conversion path, look for the existing one: a function that already builds the query and times the call, an object that already holds the URL and connection arguments, a stdlib behaviour the new code re-creates. Name what you found and the call that replaces the new code. The same check applies inside the diff: an expression repeated at three or more sites belongs in one function, so the sites can't drift.
- **Cost on a hot path.** Work added to something that runs per query, per document or per message: re-validating on every build, constructing a helper per call, sorting a whole result set once per page. Measure it before and after with a realistic size, and give the numbers and the machine.
- **The wrong layer.** A special case bolted onto a caller when the problem lives in a shared builder, client or base class. Or a fix in one subclass that its siblings also need.
- **Patching a footgun.** Effort spent making a deprecated or error-prone API safer, when finishing its deprecation or removing it would be simpler. Use `deprecated_argument` and `deprecated_function` from `redisvl.utils.utils` for the path out.
- **Public API shape.** New or changed signatures, defaults and return types, and whether they stay consistent across sync and async classes and with sibling classes. One example is a new vectorizer that doesn't follow its siblings' constructor and method conventions. New optional parameters should be keyword-only, an explicit signature is better than `*args, **kwargs` passed through, and new code shouldn't reach into another class's private attributes when a public method does the same.
- **Compatibility.** Changes to public behaviour or defaults with no deprecation path. Compatibility branches left behind after a version floor was raised.
- **New dependencies.** A new required dependency, or an optional one imported at module level instead of through `lazy_import`.
- **Error semantics.** What the caller sees on partial failure, retry or reconnect, and whether that matches how the rest of the package behaves.

## How to report

Name the cost of leaving it as it is, such as "every new provider repeats this branch", and the smallest change that would remove that cost. If the better design is clearly out of scope, report it as an open item for the author to decide, not as something to fix in this PR.

Don't propose new abstractions for a single use. Three similar lines are cheaper than a premature helper.
