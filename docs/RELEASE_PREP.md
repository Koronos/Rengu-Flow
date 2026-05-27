# Public release preparation — tracked files audit

**Updated:** 2026-05-26 after cleanup pass.

---

## Done in this pass

| Action | Detail |
|--------|--------|
| Backlog | [BACKLOG.md](BACKLOG.md) — canonical |
| Architecture | [developer/architecture.md](developer/architecture.md) — from former executive report (design + flow) |
| Removed from git | `EXECUTIVE_REPORT.md`, `docs/package-upgrade-journal.md`, `docs/training-tuning-journal.md`, `requirements-pinned.txt`, `scripts/debug_euler_preview.py`, `scripts/run_upgrade_smoke.sh` |
| Local copies | Former journals + executive report copied to **`tmp/journals/`** (gitignored) |
| License | Root [LICENSE](../LICENSE) (GPL-3.0-or-later), [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md), `pyproject.toml` `license` field |

---

## Scripts kept (supported)

| Script | Role |
|--------|------|
| `run_model_smoke.sh` | GPU smoke (documented in testing.md) |
| `smoke_training_signals.sh` | Signal-file smoke |
| `smoke_perf_ab.sh` | Cache/dataloader A/B |
| `vendor_smoke_cc0.sh` | Fixture vendor |
| `poc_cpu_ram_optimizations.py` | POC benchmark (+ pytest smoke) |
| `run_preview_once.sh` | Maintainer: Cosmos preview to PNG under `output/` |
| `tensorboard.sh` | Log viewer helper |

---

## Still large but intentional (~17 MB tracked)

- `renga_flow/model/cosmos_predict2/assets/**` — bundled tokenizers (~13 MB)
- `tests/fixtures/smoke_cc0/images/` — CC0 test images (~1 MB)
- `uv.lock` — reproducible installs

---

## Optional later

| Item | Notes |
|------|-------|
| `.cursor/rules/` | Contributor-only; can gitignore `.cursor/` if desired |
| `CHANGELOG.md` | For tagged releases |
| `CONTRIBUTING.md` | Point to developer testing + BACKLOG |

---

## Related

- [BACKLOG.md](BACKLOG.md)
- [README.md](README.md)
