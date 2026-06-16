# Documentation conventions

All feature documentation must be written **in English**.

## Audiences

### End user (trains models)

- **Location**: `docs/user/`
- **Content**: How to use the framework **as it works today** — config, CLI, training, signals, workflows. No implementation details.
- **Tone**: Task-oriented, examples, “how do I…”.

### Developer (extends functionality)

- **Location**: `docs/developer/`
- **Content**: Technical contract, where the code lives, APIs, how to add or extend features. Paths under `rengu_flow/`.
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

## UI field hints (FieldHelpIcon / config_field_help.py + dataset_field_help.py)

Audience is mixed (hobbyists + advanced). Every form field gets a hint; each hint is
at most two sentences, in this order:

1. **What it does** — concrete mechanism, present tense, no theory.
2. **Effect on the user's result** — when only one thing fits, this wins.
3. **When to touch it** — an OBSERVABLE symptom ("raise it if the tagger keeps
   attaching the wrong character name"), never internal theory.

Calibration rules:

- Numbers over adjectives ("~17 GB", "0.5 = fewer but surer tags"); never a bare
  "faster/better".
- The default is visible when blank/unset is meaningful.
- Jargon only in the mechanism part, and only when it is the searchable term;
  forbidden in the symptom trigger.
- Never explain general concepts (what VRAM or a tag is) in a hint — depth lives in
  the linked `docs/user/*.md`, which the help icon opens in a drawer.
- **The code is the source of truth**: every factual claim (defaults, semantics, UI
  labels) is validated against the code; on contradiction the text gets fixed. What
  cannot be verified is not written.
- Double test before accepting a hint: (a) a hobbyist knows *when* to touch the
  option without opening the doc; (b) an expert learns the exact behavior
  (units/default/scope) without feeling tutorialized.

Canonical example: *"Includes character/series name tags in the output. Taggers are
weakest here — turn off if they keep mislabeling your characters, and put your own
trigger in Prepend tags instead."*

## Form-field anatomy (all UI forms)

Every field follows the same visual contract:

1. **Label**: human-readable name. A `*` marks required fields — nothing else does.
2. **TOML path** directly under the label, small muted mono text (e.g.
   `optimizer.lr`), for every field that maps to a TOML key.
3. **Placeholder always means "what you get if you leave this empty"**:
   - Field with a default → the placeholder IS the default value (e.g. `42`,
     `model default (0.5)`); do **not** label the field "(optional)" and do not
     repeat the default elsewhere.
   - Optional without a default → placeholder shows a realistic example prefixed
     with `e.g. ` (the prefix prevents reading it as a default); empty = unset.
   - Required without a default → same `e.g. ` placeholder plus the `*`.
4. **Hint**: every field carries a FieldHelpIcon following the hint rules above.
5. Inputs without a placeholder slot (selects, switches) communicate the default by
   their initial state; the hint states it when the state alone is ambiguous.
