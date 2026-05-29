# Architecture and design goals

Rengu is a **TOML-driven**, **registry-based** training framework. It reimplements and extends ideas from [diffusion-pipe](https://github.com/tdrussell/diffusion-pipe) in this repository (no runtime dependency on a diffusion-pipe install).

**Implementation status:** [BACKLOG.md](../BACKLOG.md). **Upstream mapping:** [dependencies-and-upstream.md](dependencies-and-upstream.md).

---

## Objective

Provide a modular training flow where you can plug in, via config and explicit registration:

- Models (pipelines)
- Optimizers and LR schedulers
- Adapters (LoRA, LoKr, …)
- Dataset sources and cache
- Pre/post training hooks and step callbacks (planned)

Execution is launched from a main TOML (and optional dataset TOML), similar to diffusion-pipe.

---

## Design principles

1. **Configuration as contract** — TOML (+ CLI) is the source of truth for which components run.
2. **Explicit registration** — Components are registered by **string name** (decorators / registries). No filesystem auto-discovery.
3. **Single resolution path** — For each component type: read `type` from config → registry lookup → construct with that section’s kwargs.
4. **Predictable order** — Fixed phases: config load → distributed init → resolve components → pre-train (cache, load weights) → build DeepSpeed pipeline → train loop → post-train (planned) → shutdown.
5. **DeepSpeed** — Current distribution backend; no alternate backend abstraction yet.

---

## Target execution flow

1. Load main TOML and dataset TOML; apply defaults; validate.
2. Initialize distributed / DeepSpeed.
3. Resolve model, optimizer, scheduler, adapter (if any) from registries.
4. **Pre-training:** `DatasetManager.cache()` when using real data; load diffusion weights; configure adapter.
5. Build pipeline from `model.to_layers()`; create optimizer, scheduler, `PipelineDataLoader`.
6. Resume checkpoint if requested.
7. **Training loop:** batch → `train_batch` → logging, eval, `Saver`, signal files, previews per config.
8. **Post-training** (planned): optional hooks after the loop.
9. Shutdown and final saves.

---

## Registries (current and planned)

| Component | Registry / mechanism | Status |
|-----------|----------------------|--------|
| Model | `rengu_flow.registry.models` | `sdxl`, `cosmos_predict2` (+ alias `anima`) |
| Optimizer | `rengu_flow.registry.optimizers` + `optim/resolver.py` | Aliases + qualified paths + vendor optimizers |
| Scheduler | `rengu_flow.registry.schedulers` | constant, linear, cosine, paths |
| Adapter | Branches in pipeline + `rengu_flow/networks/*` | LoRA/LoKr; **no adapter registry yet** — see BACKLOG P0-3 |
| Dataset | `Dataset` / `DirectoryDataset` + `DatasetManager` | Directory + buckets |
| Step/epoch callbacks | `Saver`, eval, previews | **No generic callback registry** — BACKLOG P0-4 |
| Pre/post hooks | `DatasetManager.cache()` only | **Named hook registry** — BACKLOG P0-5, P0-6 |

---

## TOML extensibility (planned)

The schema may grow beyond diffusion-pipe. Intended pattern:

- **Core sections:** `[model]`, `[optimizer]`, dataset reference, training options (required for the standard path).
- **Optional registered blocks:** e.g. `[[post_train]]`, dataset tagging phases — resolved by name at the right hook point; unknown keys either fail validation or are ignored per policy.

Not implemented yet; tracked in [BACKLOG.md](../BACKLOG.md) (hooks + extensible phases).

---

## Naming

Migration-friendly names are kept where they match diffusion-pipe (`DatasetManager`, `PipelineDataLoader`, `ManualPipelineModule`). New code should follow existing `rengu_flow` module layout.

---

## Related docs

- [Model pipeline contract](model-pipeline-contract.md) — per-method implementation status
- [Networks](networks.md) — adapter modules and how to add types
- [Adding optimizers and schedulers](adding-optimizers-and-schedulers.md)
