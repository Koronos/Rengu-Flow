# Documentation conventions

All feature documentation must be written **in English**.

## Audiences

### End user (trains models)

- **Location**: `docs/user/`
- **Content**: How to use the framework **as it works today** — config, CLI, training, signals, workflows. No implementation details.
- **Tone**: Task-oriented, examples, “how do I…”.

### Developer (extends functionality)

- **Location**: `docs/developer/`
- **Content**: Technical contract, where the code lives, APIs, how to add or extend features. Paths under `renga_flow/`.
- **Tone**: Implementation-focused, “where to change what”, extension steps.

### Specifications (not implemented yet)

- **Location**: `docs/spec/`
- **Content**: Intended behavior for features **not shipped**. Link `docs/BACKLOG.md` IDs.
- **Do not** describe spec-only features in `docs/user/` as if users can enable them today.

## Example configs (TOML)

- **Location**: `examples/` at the repo root.
- Keep runnable examples aligned with schema; mark spec-only keys with `# planned — see docs/spec/…`.

## Optional parameters

When documenting config keys, CLI flags, or API parameters that are **optional**:

- **Do not** only mention them by name.
- **Do** document each with **Purpose**, **Values**, and **Default** (tables in user docs).

## Checklist when adding or changing features

- [ ] **Implemented?** If no → `docs/spec/` + BACKLOG; not a how-to in `docs/user/`.
- [ ] **User doc**: Updated page under `docs/user/` when shipped.
- [ ] **Developer doc**: Contract and code locations under `docs/developer/`.
- [ ] **Optional options**: Purpose, values, default (not name-only lists).
- [ ] **Examples**: `examples/` updated; spec keys commented as planned.
- [ ] **Index**: `docs/README.md` links user, developer, and spec sections.
- [ ] **UI hints**: `FIELD_HELP` / `doc_path` if the field appears in the web UI.
