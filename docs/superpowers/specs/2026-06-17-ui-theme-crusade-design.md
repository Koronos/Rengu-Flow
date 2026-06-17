# UI theme crusade — design spec

**Status:** in progress · **Branch:** `ui-theme` · **Keep LOCAL (no push) until the user says.**
**Approved direction:** Candidate **B** — cyan primary on navy surfaces (serious/professional, on-brand
with the logo + Poet). Iterate freely; this is reversible.

## Goal
Re-skin the whole rengu web UI (Element Plus + Vue 3, dark mode) into a cohesive, serious/professional
theme derived from the new logo (navy badge `#0a1b35→#15375f` + cyan flow `#1fb6e6→#5eead4`) and the Poet
mascot (teal hair, navy haori, red-flower accent). Plus per-section style/usability polish.

## Principles
One accent (cyan). Semantic color only where it means something. Consistency via shared tokens.
High legibility on navy. Subtle depth (borders/overlays, minimal shadows). On-brand.

## Color tokens (override Element Plus CSS vars in `ui/web/src/styles/app.css`, `html.dark`)
Primary (cyan):
- `--el-color-primary: #22b8e6`
- `--el-color-primary-light-3: #1f9bc2` · `-light-5: #1a7d9c` · `-light-7: #155a70`
- `--el-color-primary-light-8: #123c4b` · `-light-9: #0f2935` (hover/plain bg)
- `--el-color-primary-dark-2: #56ccf0`

Navy surfaces:
- `--el-bg-color: #0f1f33` · `--el-bg-color-page: #0a1626` · `--el-bg-color-overlay: #15263d`
- fills: `--el-fill-color-blank: #0f1f33` · `-light: #16273e` · `-lighter: #132137` · base `#1d3049`
  · `-dark: #22364f`
- borders: `--el-border-color: #26405f` · `-light: #213855` · `-lighter: #1b3049` · `-extra-light: #172841`
- text: `--el-text-color-primary: #e8eef7` · `-regular: #c4cfe1` · `-secondary: #8ba1bc`
  · `-placeholder: #6b7f99`

Semantic: keep EP success/warning/danger/info (tune only if they clash on navy).
`index.html` `theme-color` → `#0a1626`. **Poet coral accent:** deferred (start 100% cyan+navy).

## Execution
1. **Global tokens first** (biggest lever) — edit `app.css` `html.dark` + `index.html`. Verify with the
   chrome-devtools MCP across a few sections (before/after screenshots).
2. **Per-section polish** — one task per section; audit via MCP, fix, verify. Delegate mechanical/repetitive
   edits to agents, audit via `git diff`; the theme + visual verification stay centrally driven for consistency.
3. Local commits per logical step on `ui-theme`. No push until approved.

## Sections checklist (status)
- [ ] Global tokens (app.css + index.html theme-color) + shell (sidebar/header/brand, CPU/RAM/GPU chips)
- [ ] Runs (JobsView): queue, history table, run-state colors, button hierarchy, empty states
- [ ] Compare (RunComparisonView): already iterated; align to tokens (board controls, legend, hparams)
- [ ] Datasets (DatasetsListView)
- [ ] Studio / Prep (PrepJobsView, PrepJobFormView, TagEditorView)
- [ ] Run detail (RunDetailView): loss monitor, previews, lineage/timeline
- [ ] Forms (RunFormView): config form fields, hints, sections
- [ ] Docs (DocsView), Maintenance (MaintenanceView)
- [ ] Run-state color map (running/pending/finished/failed/stopped) — consistent across views
- [ ] Usability pass: focus-visible, empty states, hover affordances, table readability, button hierarchy

## Out of scope (YAGNI)
No feature/layout restructuring, no new components, no light-mode redesign (stays dark), no logo/mascot changes.

## Theming infra (baseline)
`main.ts` loads Element Plus full + `theme-chalk/dark/css-vars.css`. `app.css` `:root` has `--rf-*` tokens
(fonts/spacing/radius) and `html.dark` overrides `--el-bg-color*` (today neutral gray #141414). Re-theme =
override these EP vars in one place. Primary is EP default blue today (not overridden).
