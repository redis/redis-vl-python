# Design lens

Ask whether the change is a lasting solution at the right layer, or a patch that the next change will have to undo. In this repository, this lens more often produces a decision than a defect: for example, "remove the deprecated API instead of patching it".

## Look for

- **The wrong layer.** A special case bolted onto a caller when the problem lives in a shared builder, client or base class. Or a fix in one subclass that its siblings also need.
- **Patching a footgun.** Effort spent making a deprecated or error-prone API safer, when finishing its deprecation or removing it would be simpler. Use `deprecated_argument` and `deprecated_function` from `redisvl.utils.utils` for the path out.
- **Public API shape.** New or changed signatures, defaults and return types, and whether they stay consistent across sync and async classes and with sibling classes. One example is a new vectorizer that doesn't follow its siblings' constructor and method conventions.
- **Compatibility.** Changes to public behaviour or defaults with no deprecation path. Compatibility branches left behind after a version floor was raised.
- **New dependencies.** A new required dependency, or an optional one imported at module level instead of through `lazy_import`.
- **Error semantics.** What the caller sees on partial failure, retry or reconnect, and whether that matches how the rest of the package behaves.

## How to report

Name the cost of leaving it as it is, such as "every new provider repeats this branch", and the smallest change that would remove that cost. If the better design is clearly out of scope, report it as an open item for the author to decide, not as something to fix in this PR.

Don't propose new abstractions for a single use. Three similar lines are cheaper than a premature helper.
