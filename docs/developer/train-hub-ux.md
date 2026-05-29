# Train hub UX (developer note)

The full **Train hub** design (unified nav, live panel, merged run list) is a **specification**, not the current UI.

- **Specification:** [spec/train-hub-ux.md](../spec/train-hub-ux.md)
- **Shipped today:** [Web UI](web-ui.md) — nav **Datasets**, **Configs**, **Runs**; launch and queue in `ui/web/src/views/JobsView.vue` (`/runs`).

When implementing the spec, extend `rengu_flow_ui/app.py` and `JobsView.vue` per the spec’s API section; keep backward compatibility with existing `/jobs` and `/runs` routes.
