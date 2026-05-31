# Rengu-Flow — Backlog (E2E UI training audit)

> Origen: auditoría E2E manejando la **UI en modo develop** (`uv run rengu ui dev`, Vite `:5173` + API `:8765`)
> con entrenamientos reales de **4 pasos** sobre GPU (RTX 3000 Ada, 8 GB), dataset CC0 de los smoke tests
> (12 imágenes @ 512 px) y modelos locales (Illustrious-XL para SDXL; anima-base + Qwen VAE/LLM para Cosmos).
> Fecha: 2026-05-31.

Cada ítem es **auto-contenido** (pensado para ser tomado por agentes distintos en paralelo). Campos:
**Sev** (severidad), **Estado**, **Área**, **Archivos**, **Repro**, **Causa raíz**, **Fix propuesto**, **Aceptación**, **Dependencias**.

## Leyenda de estado
- `TODO` — sin empezar.
- `PARTIAL-FIX-IN-TREE` — hay un parche aplicado en el working tree (sin commitear); falta completar/revisar/testear.
- `BLOCKED` — depende de otro ítem.

## Resultado de la matriz (estado real, tras los fixes RF-01…RF-13)
| Caso | ¿Entrena vía UI? | Nota |
|---|---|---|
| SDXL · LoRA | ✅ | guardó `step4/lora.safetensors` |
| SDXL · LoKr | ✅ | **arreglado** (RF-05, backend vendored) — guardó `step4/adapter_model.safetensors` |
| Cosmos · LoRA | ✅ | guardó `step4/adapter_model.safetensors` |
| Cosmos · LoKr | ✅ | backend LyCORIS OK |

---

## Prioridad sugerida
1. **P0 (lanzar entrenamiento):** RF-01, RF-02, RF-03, RF-04
2. **P1 (correctitud de estado/monitoreo):** RF-06, RF-07
3. **P1 (feature rota):** RF-05
4. **P2 (UX):** RF-08, RF-09, RF-10
5. **P2 (docs):** RF-11
6. **P3 (entorno WSL/Windows):** RF-12, RF-13

---

## RF-01 — UI: DeepSpeed se invoca con `-m` (no existe) en vez de `--module`
- **Sev:** P0 (bloqueante total — ningún entrenamiento arranca desde la UI)
- **Estado:** ✅ DONE (sin commit) — `-m`→`--module` + `--master_port`; build factorizado en `train_launcher.base_train_command`. Tests `tests/test_cli.py`. Verificado E2E (SDXL LoRA por UI, `--module --master_port`).
- **Área:** UI backend / launcher
- **Archivos:** `rengu_flow_ui/jobs.py:32` (`build_train_command`)
- **Repro:** Run now en cualquier config → `data/logs/pending.log` muestra
  `deepspeed: error: unrecognized arguments: -m`.
- **Causa raíz:** `cmd = [deepspeed, f"--num_gpus={num_gpus}", "-m", "rengu_flow.main", ...]`.
  DeepSpeed 0.19 usa `--module`, no `-m` (los smoke scripts ya usan `--module`, p.ej. `scripts/run_model_smoke.sh:86`).
- **Fix propuesto:** `-m` → `--module`. **Además añadir `--master_port`** (paridad con CLI/smoke; evita colisiones).
  *Parche en tree:* ya cambiado a `--module` (falta `--master_port`).
- **Aceptación:** `data/staging/<id>/train.toml` se lanza y el CMD del log contiene `--module rengu_flow.main`; un job llega a `step=1/...`.
- **Dependencias:** ninguna.

## RF-02 — CLI: mismo bug `-m` en `rengu train`
- **Sev:** P0 (bloqueante total — `rengu train` también roto con este DeepSpeed)
- **Estado:** ✅ DONE (sin commit) — `-m`→`--module` en `train_launcher.py`; build común con la UI (`base_train_command`). Tests `tests/test_cli.py`.
- **Área:** CLI / launcher
- **Archivos:** `rengu_flow/cli/train_launcher.py:71`
- **Repro:** `uv run rengu train --config <toml>` con DeepSpeed 0.19 → `unrecognized arguments: -m`.
- **Causa raíz:** `cmd = [deepspeed, f"--num_gpus={ngpus}", f"--master_port={port}", "-m", "rengu_flow.main", ...]`.
- **Fix propuesto:** `-m` → `--module`. Considerar factorizar el build de comando compartido con `rengu_flow_ui/jobs.py` para no duplicar el bug.
- **Aceptación:** `rengu train` arranca un entrenamiento; idealmente un test que verifique que el comando contiene `--module`.
- **Dependencias:** comparte causa con RF-01 (candidato a refactor común).

## RF-03 — UI: el staging corrompe el `train.toml` (`set_config_defaults` no idempotente)
- **Sev:** P0 (bloqueante total — todo lanzamiento desde la UI genera un staged TOML inválido)
- **Estado:** ✅ DONE (sin commit) — `materialize_staging` valida sobre `deepcopy` y vuelca el config sin mutar. Test `tests/test_configs_store.py::test_materialize_staging_does_not_persist_defaults`.
- **Área:** UI backend / staging
- **Archivos:** `rengu_flow_ui/configs_store.py:163` (`materialize_staging`); referencia: `rengu_flow/config/defaults.py:83`
- **Repro:** Config válido en biblioteca con `dtype = "bfloat16"`, `[adapter] type=lora rank=8` →
  el `data/staging/<id>/train.toml` sale con `dtype = "torch.bfloat16"` y `alpha = rank`. El trainer revienta con
  `KeyError: 'torch.bfloat16'` y/o `ConfigValidationError: Remove alpha from [adapter]`.
- **Causa raíz:** `materialize_staging` hace `set_config_defaults(config)` y luego `toml.dumps(config)`.
  `set_config_defaults` **muta** el dict: convierte strings de dtype a objetos `torch.dtype` (se serializan como `"torch.bfloat16"`)
  e **inyecta** `alpha=rank` (que el trainer luego rechaza al reaplicar defaults). No es idempotente.
- **Fix propuesto:** Validar sobre una **copia** y volcar el config sin mutar (solo conservar la resolución de `dataset`).
  *Parche en tree:* `set_config_defaults(copy.deepcopy(config))` para validación; se vuelca el `config` original (solo dataset resuelto).
- **Aceptación:** `data/staging/<id>/train.toml` conserva `dtype = "bfloat16"` y **no** tiene `alpha`; el trainer no aborta en `set_config_defaults`.
- **Dependencias:** ninguna (independiente de RF-01).

## RF-04 — SDXL no arranca sin el extra de Cosmos (import eager de cosmos → `torchvision`)
- **Sev:** P0 (SDXL no entrena en un entorno solo-`[ui]`)
- **Estado:** ✅ DONE (sin commit) — import lazy en `registry/models.py` (`get_model`) **y** `torchvision` movido a deps base en `pyproject.toml` (SDXL también lo usa vía `data/preprocess_media.py`). Test `tests/test_registry_lazy.py`. Verificado E2E.
- **Área:** registry / instalación de extras
- **Archivos:** `rengu_flow/registry/models.py` (antes `_register_builtin_models` importaba `rengu_flow.model.cosmos_predict2` eager);
  `rengu_flow/install/manager.py:121` (perfiles de extras); `pyproject.toml` (`torchvision` solo en extra `cosmos_predict2`).
- **Repro:** Entorno con solo `uv sync --extra ui`. Lanzar SDXL → `ModuleNotFoundError: No module named 'torchvision'`
  (cadena: registry → `cosmos_predict2/__init__` → `pipeline` → `dit.py:25` `from torchvision import transforms`).
- **Causa raíz:** el registry importa el pipeline de Cosmos **incondicionalmente**, arrastrando `torchvision`; pero el auto-sync
  (`manager.py`) solo añade el extra `cosmos` cuando `model.type == cosmos_predict2`, así que para SDXL nunca se instala.
- **Fix propuesto (elige uno):** (a) import **lazy** de cada modelo en el registry; o (b) mover `torchvision` a deps base; o
  (c) que `ensure_training_extras` garantice `torchvision` siempre. Contradice la doc que promete auto-sync de extras (ver RF-11).
- **Aceptación:** SDXL LoRA arranca en un entorno con solo `[ui]` (+ deps base) sin instalar el extra de Cosmos.
- **Dependencias:** ninguna.

## RF-05 — SDXL + LoKr está roto en ambos backends
- **Sev:** P1 (feature anunciada que no funciona)
- **Estado:** ✅ DONE (sin commit) — causa raíz: el backend LyCORIS cuelga el adapter del ROOT del modelo, fuera de las capas del pipeline DeepSpeed (params ni se mueven a GPU ni se entrenan). Fix: SDXL-LoKr usa siempre el backend **vendored** (registra params en cada nn.Linear, in-tree) y se corrigió su recorrido de módulos (inyecta sobre `containers.modules()`, sin `getattr(parent,"0")`). Tests `tests/test_lokr_sdxl.py`; smoke `tests/fixtures/smoke/train_sdxl_lokr.toml` + `scripts/run_model_smoke.sh sdxl_lokr`. Verificado E2E (4 pasos, guardó adapter).
- **Área:** networks / adapters
- **Archivos:** `rengu_flow/networks/lokr_sdxl.py:185` (backend LyCORIS), `:124`/`:138`/`:173` (backend vendored)
- **Repro:** Config SDXL con `[adapter] type="lokr"`, Run now (la UI instala `lycoris-lora` sola para LoKr):
  - **LyCORIS:** `RuntimeError: Expected all tensors to be on the same device, cuda:0 and cpu` en `lycoris/modules/lokr.py:560` (`base_weight + diff_weight`).
  - **Vendored** (sin `lycoris-lora`): `AttributeError: 'UNet2DConditionModel' object has no attribute '0'` en `_apply_lokr_vendored`.
- **Causa raíz:**
  - LyCORIS: la red LoKr se inyecta (`create_lycoris`+`apply_to()`) **antes** de mover el modelo a GPU (DeepSpeed); sus parámetros no migran al device.
  - Vendored: el recorrido usa `getattr(parent, part)` con índices numéricos de `ModuleList`/`Sequential` (`"0"`), debería indexar `parent[int(part)]`; además los nombres parecen relativos al contenedor equivocado (root mismatch).
- **Pista:** **Cosmos LoKr usa LyCORIS y SÍ funciona** → el bug es de la integración SDXL-LoKr, no de LyCORIS.
- **Fix propuesto:** mover la red LoKr al device tras `deepspeed.initialize` (o registrar params para que migren); corregir el recorrido de módulos del vendored (indexar ModuleList + root correcto).
- **Aceptación:** SDXL LoKr completa ≥1 paso y guarda adapter; **añadir un smoke test de GPU para SDXL-LoKr** (hoy solo existe `tests/fixtures/smoke/train_cosmos_predict2_lokr.toml`).
- **Dependencias:** RF-01/RF-03 (para poder llegar al training).

## RF-06 — Jobs fallidos se reportan como `finished` y `exit_code` siempre `null`
- **Sev:** P1 (correctitud — imposible distinguir éxito de fallo desde la UI)
- **Estado:** ✅ DONE (sin commit) — `_read_exit_code` parsea el log del job (deepspeed exit / traceback); `poll_job` marca `failed` si exit≠0. Tests `tests/test_job_lifecycle.py`. Verificado live (exit_code=0→finished).
- **Área:** UI backend / estado de jobs
- **Archivos:** `rengu_flow_ui/jobs.py:98` (`poll_job` → reconcilia a `finished`), `rengu_flow_ui/jobs.py:117` (`_read_exit_code` es stub que retorna `None`)
- **Repro:** Lanzar un config que crashea (p.ej. antes de los parches RF-01/03). El job aparece `finished` con `exit_code=null`.
- **Causa raíz:** al detectar el PID muerto, un job "running" no monitoreado se marca `finished` sin leer el código de salida; `_read_exit_code` no está implementado.
- **Fix propuesto:** capturar el `returncode` del subproceso (o leerlo de un marcador en el run dir) y distinguir estado `failed` vs `finished`; superficializar el error en la UI.
- **Aceptación:** un job que sale con código ≠ 0 queda en estado `failed` con su `exit_code`; la UI lo muestra distinto a un éxito.
- **Dependencias:** se beneficia de RF-07 (log/run_dir por job).

## RF-07 — Logs y `run_dir` no son por-job (todo va a `pending.log`)
- **Sev:** P1 (monitoreo por run no funciona: logs/métricas/señales/TensorBoard)
- **Estado:** ✅ DONE (sin commit) — causa raíz: `db.update_job` no permitía `log_path` (se descartaba). Añadido a `allowed`; `poll_job` captura `run_dir` parseando `Run dir:` del log del job; `resolve_job_run_dir` ya no usa el fallback "carpeta más nueva" para jobs terminales. Tests `tests/test_job_lifecycle.py`. Verificado live (11.log + run_dir propio).
- **Área:** UI backend / job tracking
- **Archivos:** `rengu_flow_ui/job_queue.py` (set de `log_path` a `<id>.log`), `rengu_flow_ui/jobs.py:41` (`start_job` usa `job.log_path`), `db` (`run_dir`)
- **Repro:** Tras un run exitoso, `GET /api/v1/jobs/<id>` devuelve `log_path=.../pending.log` y `run_dir=null`; `<id>.log` nunca se crea. En la lista de Runs, **todos** los jobs muestran el mismo progreso (`step 4/4 100%`) porque leen el mismo `pending.log` — incluso los que fallaron.
- **Causa raíz:** el subproceso escribe al `pending.log` original; el `log_path` por-job no se aplica y `run_dir` (parseable del log `Run dir: ...`) no se captura.
- **Fix propuesto:** escribir cada job a su propio `<id>.log`; parsear y persistir `run_dir` desde el log correcto; el progreso/estado de la lista debe leer el log/`status.json` del run, no el compartido.
- **Aceptación:** cada run tiene su `<id>.log`, su `run_dir` poblado, y la lista de Runs muestra progreso/estado **independiente** por run.
- **Dependencias:** habilita RF-06.

## RF-08 — Selector de dataset (vista Tabla) no selecciona al hacer clic en la fila
- **Sev:** P2 (UX — descubribilidad; hay workaround)
- **Estado:** ✅ NO ES BUG (verificado) — el código ya conecta `@row-click → item-click → toggle` (`DatasetPreviewTable.vue:9,98`) y el campo Dataset es `multiple`. Con un **clic real** la fila completa selecciona ("Selected: 1", fila activa) y "Add selected" añade el chip al config. El fallo del audit fue un artefacto del `.click()` sintético (no dispara la delegación de row-click de Element Plus). Sin cambios de código. (Nicety menor posible, no aplicada: un clic en la columna de acciones también dispara `row-click`.)
- **Área:** UI frontend (Configs / dataset picker)
- **Archivos:** `ui/web/src/**` (diálogo "Choose datasets", vista Table; campo Dataset del config form)
- **Repro:** Config form → Dataset → "Add dataset" → vista Table → clic en la fila `smoke_cc0`: no se selecciona y "Add selected" queda deshabilitado. Solo el ícono-check minúsculo de la 1ª columna selecciona. Además, al confirmar, la selección **no se aplicó** (el campo seguía "No training datasets yet"). *Workaround que sí funciona:* escribir la ruta `.toml` en el input "Or type a .toml path" + Enter.
- **Fix propuesto:** que el clic en toda la fila seleccione (o checkbox visible); verificar que "Add selected" persista la selección al campo `dataset`.
- **Aceptación:** seleccionar por clic de fila y que el dataset quede en el config (`dataset = "rengu-flow-dataset:<id>:..."`).
- **Dependencias:** ninguna.

## RF-09 — Defaults de `[preview]` demasiado pesados para smokes / 8 GB
- **Sev:** P2 (UX / OOM)
- **Estado:** `TODO`
- **Área:** plantillas de config / form seed
- **Archivos:** seed/plantilla de config (genera `[preview] enabled=true`, `width=1024`, `height=1024`); `rengu_flow_ui/config_schema.py`
- **Repro:** "New config" SDXL incluye `[preview] enabled=true 1024x1024`; en 8 GB un smoke corto puede OOMear al generar preview.
- **Fix propuesto:** preview desactivado por defecto (o resolución menor) en plantillas/smoke; avisar de coste de VRAM.
- **Aceptación:** un run nuevo por defecto no intenta preview 1024² salvo que el usuario lo active.
- **Dependencias:** ninguna.

## RF-10 — Pulidos varios de UI/serialización
- **Sev:** P2 (UX menor)
- **Estado:** 🟡 PARCIAL (sin commit) — hecho el punto de mayor valor; el resto diferido (bajo ROI / verificación con file-dialog inviable en el harness).
- **Área:** UI frontend + serializador TOML
- **Detalles (sub-tareas independientes):**
  1. ⬜ DEFERIDO (cosmético): el serializador reescribe arrays como `resolutions = [ 512,]`. Es comportamiento de la lib `toml` y produce TOML válido; cambiarlo (serializador custom / `tomlkit`) afecta toda la salida con poco beneficio.
  2. ⬜ DEFERIDO: Datasets sin "Import TOML" (Configs sí). Requiere composable + endpoint nuevos; existe workaround (New dataset → pestaña TOML → pegar). Verificación del file-dialog no es automatizable aquí.
  3. ✅ DONE (sin commit) — el ID de config nuevo deriva del `run_name` (`ConfigEditorView.vue:save()`). Verificado en navegador (run_name "my_sdxl_run" → id por defecto "my_sdxl_run").
  4. ⬜ DEFERIDO (menor): "New dataset" no añade un `[[directory]]` por defecto.
- **Aceptación:** punto 3 cerrado; 1/2/4 quedan como niceties menores en backlog.
- **Dependencias:** ninguna.

## RF-11 — Docs: promesas que no se cumplen + omisiones
- **Sev:** P2 (docs)
- **Estado:** `TODO`
- **Área:** documentación
- **Archivos:** `docs/user/web-ui.md`, `docs/user/training-sdxl-lora-lokr.md`
- **Detalles:**
  1. `web-ui.md` dice que la UI hace `uv sync` de extras al iniciar training → **falso para SDXL** (ver RF-04).
  2. No documenta que **SDXL-LoKr** no funciona hoy (ver RF-05).
  3. No advierte del coste de VRAM del preview por defecto (ver RF-09).
- **Aceptación:** docs alineadas con el comportamiento real (o los bugs cerrados y la doc verificada).
- **Dependencias:** idealmente tras RF-04/RF-05/RF-09.

## RF-12 — Worktrees en CRLF sin `.gitattributes` rompen `./rengu` en WSL
- **Sev:** P3 (entorno; no es bug de producto pero bloquea en WSL/Windows)
- **Estado:** ✅ DONE (sin commit) — añadido `.gitattributes` (`* text=auto`, `*.sh`/`rengu`/`*.py`/`*.toml`/`*.md` con `eol=lf`, binarios marcados). Scripts del worktree convertidos a LF; `./rengu --help` corre bajo WSL. **Nota:** los archivos ya versionados se normalizan con `git add --renormalize .` (correr desde un entorno git válido — ver RF-13).
- **Área:** repo / portabilidad
- **Archivos:** no hay `.gitattributes`; scripts shell (`rengu`, `start-ui.sh`, `scripts/*.sh`) quedan en CRLF al hacer checkout en Windows.
- **Repro:** `./rengu ...` en WSL → `/usr/bin/env: 'bash\r': No such file or directory`. *Workaround:* `uv run rengu ...`.
- **Fix propuesto:** añadir `.gitattributes` forzando `* text=auto` y `*.sh`/`rengu` con `eol=lf`.
- **Aceptación:** `./rengu` ejecuta en WSL tras un checkout limpio en Windows.
- **Dependencias:** ninguna.

## RF-13 — `.git` del worktree apunta a ruta UNC de Windows → git inutilizable desde WSL
- **Sev:** P3 (entorno; afecta a workflows de agentes en WSL)
- **Estado:** ✅ DONE — documentado (no es código de producto). Causa: el worktree lo creó el git de Windows, así que `.git` apunta a un gitdir UNC que el git de WSL no resuelve. **Workarounds:** (a) usar git desde Windows con `git config --global --add safe.directory '%(prefix)///wsl.localhost/...'`; (b) crear los worktrees desde WSL (`git -C ~/Rengu-Flow worktree add ...`) para que el gitdir sea POSIX; (c) operar git en el repo principal en WSL, no en el worktree. Para commitear los fixes de este backlog, usar (a) o (c).
- **Área:** tooling / worktrees
- **Repro:** dentro del worktree en WSL, `git status` → `fatal: not a git repository: //wsl.localhost/ubuntu/.../.git/worktrees/...`.
- **Causa raíz:** el `.git` (gitdir) del worktree fue escrito por el git de Windows con ruta UNC; el git de WSL no la resuelve.
- **Fix propuesto:** documentar/normalizar la creación de worktrees para que el gitdir use rutas POSIX cuando se vaya a usar desde WSL.
- **Dependencias:** ninguna.

---

## Apéndice — Cómo reproducir el harness de prueba (para cualquier agente)
1. **Modelos** (WSL): `~/Rengu-Flow/tmp/models/` → `Illustrious-XL-v2.0.safetensors` (SDXL), `anima-base-v1.0.safetensors`, `qwen_image_vae.safetensors`, `qwen_3_06b_base.safetensors`.
2. **`.env`** en la raíz del worktree con `RENGU_SDXL_CHECKPOINT_PATH` + `RENGU_COSMOS_{TRANSFORMER,VAE,LLM}_PATH` apuntando a esos archivos.
3. **Extras**: `uv sync --inexact --extra ui --extra cosmos_predict2 --extra lycoris` (necesario hasta cerrar RF-04).
4. **Dev server**: `uv run rengu ui dev --no-open` (Vite `:5173`, API `:8765`). Logueado en `data/dev-api.log`.
5. **Dataset de prueba**: `tmp/e2e/dataset_cc0.toml` → `[[directory]] path` (abs) a `tests/fixtures/smoke_cc0/images` (12 imgs), `resolutions=[512]`.
6. **Config E2E**: TOML mínimo con `max_steps = 4`, sin `[preview]`. SDXL usa `checkpoint_path`; Cosmos usa `transformer_path`/`vae_path`/`llm_path`.
7. **Verificación directa (smoke, evita la UI):**
   `uv run deepspeed --num_gpus=1 --master_port=29577 --module rengu_flow.main --config data/staging/<id>/train.toml`
   → debe llegar a `step=1/4 ... step=4/4`, `Reached max_steps`, `Saving model`, `Training complete`.

## Apéndice — Cambios aplicados en el working tree (SIN commit)
Código:
- `rengu_flow/cli/train_launcher.py` — `base_train_command` compartido; `-m`→`--module` (RF-02).
- `rengu_flow_ui/jobs.py` — usa `base_train_command` (`--module` + `--master_port` libre); `_read_exit_code` real; `poll_job` captura `run_dir` y marca `failed` (RF-01/06/07).
- `rengu_flow_ui/configs_store.py` — `materialize_staging` valida sobre `deepcopy`, no persiste defaults (RF-03).
- `rengu_flow_ui/db.py` — `update_job` permite `log_path` (RF-07).
- `rengu_flow_ui/training_hub.py` — `resolve_job_run_dir` no usa el fallback "carpeta más nueva" para jobs terminales (RF-07).
- `rengu_flow/registry/models.py` — import lazy por modelo (RF-04).
- `rengu_flow/networks/lokr_sdxl.py` — SDXL-LoKr siempre vendored + recorrido corregido (RF-05).
- `rengu_flow_ui/config_schema.py` — `preview.enabled` default `false` (RF-09).
- `ui/web/src/views/ConfigEditorView.vue` — id de config nuevo desde `run_name` (RF-10.3).
- `pyproject.toml` — `torchvision` a deps base (RF-04).

Tests/fixtures: `tests/test_cli.py`, `tests/test_configs_store.py`, `tests/test_job_lifecycle.py`, `tests/test_registry_lazy.py`, `tests/test_lokr_sdxl.py`, `tests/test_preview_default.py`, `tests/fixtures/smoke/train_sdxl_lokr.toml`, `scripts/run_model_smoke.sh`.

Repo/portabilidad: `.gitattributes` (nuevo); scripts shell del worktree convertidos a LF.

Docs: `docs/user/{web-ui,previews,training-sdxl-lora-lokr,training-cosmos-predict2-lora-lokr-finetune,full-model-training-sdxl}.md`, `docs/developer/{cli,testing}.md`.

**Pendiente:** revisar, **commitear** (desde un entorno git válido — ver RF-13), y opcional `git add --renormalize .` para aplicar `.gitattributes` a archivos ya versionados.

---

# Plan de ejecución

Fases en orden de dependencia. Cada `T-x.y` es una tarea individual y accionable. Marca `[x]` al cerrar.
Convención: cada fase termina con **verificación** (re-correr el harness) + **tests** + **commit/PR**.

> **Estado global (2026-05-31):** 12/13 ítems cerrados + el sub-punto de mayor valor de RF-10. Todos los
> cambios están en el **working tree sin commitear** (git no opera desde WSL en este worktree, RF-13 — usar el
> workaround de Windows/`safe.directory` o el repo principal para commitear). Suite: **29 tests verdes** + `vue-tsc` limpio.

## Fase 0 — Desbloquear el lanzamiento (P0) ✅
> Meta: que un entrenamiento arranque desde la UI **y** el CLI. Hasta cerrar esto, nada entrena out-of-the-box.
- [x] **T-0.1 (RF-01)** UI: `-m`→`--module` + `--master_port` en `rengu_flow_ui/jobs.py`.
- [x] **T-0.2 (RF-02)** CLI: `-m`→`--module` en `rengu_flow/cli/train_launcher.py`.
- [x] **T-0.3 (refactor)** Build unificado en `train_launcher.base_train_command` (UI + CLI lo comparten).
- [x] **T-0.4 (RF-03)** `materialize_staging` valida sobre `deepcopy`; `train.toml` sale con `dtype="bfloat16"` y sin `alpha`.
- [x] **T-0.5 (RF-04)** Import lazy en `registry/models.py` **+** `torchvision` movido a deps base en `pyproject.toml`.
- [x] **T-0.6 (verificación)** E2E vía UI: SDXL·LoRA, Cosmos·LoRA, Cosmos·LoKr → `step 4/4` + adapter guardado.
- [x] **T-0.7** Tests (`--module`; staging no muta dtype/alpha; lazy import). *(commit/PR pendiente — ver estado global.)*

## Fase 1 — Correctitud de estado / monitoreo (P1) ✅
- [x] **T-1.1 (RF-07)** `db.update_job` ahora permite `log_path` (era la causa); `poll_job` parsea/persiste `run_dir` por job.
- [x] **T-1.2 (RF-06)** `_read_exit_code` parsea el log; `poll_job` marca `failed` si exit≠0.
- [x] **T-1.3 (verificación)** Live: job exit 0 → `finished` con su `run_dir`/`11.log` propios. (Camino `failed` cubierto por unit tests.)
- [x] **T-1.4** Tests (`tests/test_job_lifecycle.py`). *(commit pendiente.)*

## Fase 2 — Arreglar SDXL + LoKr (P1) ✅
- [x] **T-2.1 (RF-05a)** N/A — el backend LyCORIS es incompatible con el pipeline (params fuera de las capas); SDXL-LoKr usa **vendored** siempre.
- [x] **T-2.2 (RF-05b)** Vendored: recorrido corregido (inyecta sobre `containers.modules()`, sin `getattr(parent,"0")`) en `lokr_sdxl.py`.
- [x] **T-2.3 (RF-05c)** Smoke GPU: `tests/fixtures/smoke/train_sdxl_lokr.toml` + `scripts/run_model_smoke.sh sdxl_lokr` (también `cosmos_lokr`).
- [x] **T-2.4 (verificación)** E2E SDXL·LoKr vía UI → `step 4/4` + `adapter_model.safetensors`. *(commit pendiente.)*

## Fase 3 — UX (P2)
- [x] **T-3.1 (RF-08)** No era bug — el clic real de fila ya selecciona y "Add selected" persiste (artefacto del click sintético del audit).
- [x] **T-3.2 (RF-09)** `preview.enabled` por defecto `false` + nota de VRAM (`config_schema.py`); test `tests/test_preview_default.py`.
- [x] **T-3.3 (RF-10)** Parcial: (3) ID de config desde `run_name` ✅ (verificado). (1) formato arrays, (2) Import TOML en Datasets, (4) `[[directory]]` por defecto → DEFERIDOS (menor/cosmético).

## Fase 4 — Docs (P2) ✅
- [x] **T-4.1 (RF-11)** Alineadas `web-ui.md` (extras/torchvision base), `previews.md` (preview opt-in/VRAM), `training-sdxl-lora-lokr.md` (LoKr vendored). Corregidos **todos** los ejemplos de docs con el bug `deepspeed -m` (7 bloques + guía en `developer/cli.md`).

## Fase 5 — Entorno WSL/Windows (P3) ✅
- [x] **T-5.1 (RF-12)** `.gitattributes` añadido (`* text=auto`, `*.sh`/`rengu`/etc `eol=lf`); scripts del worktree convertidos a LF (`./rengu` corre en WSL).
- [x] **T-5.2 (RF-13)** Documentado (gitdir UNC; workarounds para git/commit desde WSL).
