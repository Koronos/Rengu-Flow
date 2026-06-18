# Toolbox

Toolbox is a top-level section in the web UI for **user-authored Python tools** — scripts
you write yourself to do custom dataset treatment, ad-hoc image processing, or any
one-off task that doesn't fit a built-in prep stage. Each tool runs in its own isolated
`uv` environment and is completely separate from Studio (`/prep`) and from rengu's own
venv.

## Enable execution

Script execution is **off by default**. You can create, edit, and save tools at any time,
but the **Run** button only works when execution is enabled. The UI shows a banner when it
is off.

Add this to `rengu.local.toml` on the machine that runs `./rengu ui serve`:

```toml
[toolbox]
enabled = true   # default: false
```

| Key | Values | Default | Effect |
|-----|--------|---------|--------|
| `toolbox.enabled` | `true` / `false` | `false` | Unlocks the Run button; authoring is always available |

Restart the UI server after changing the flag.

## Authoring a tool

Each tool is a Python script with one **entrypoint function** (default name `run`). You
declare inputs in the UI form — one per parameter of that function. When you hit **Run**,
the UI calls `run(...)` with the values you entered.

### Inputs

Each declared input maps to one parameter of the entrypoint:

| Control type | Maps to | Notes |
|--------------|---------|-------|
| **number** | numeric parameter | Integer or float |
| **text** | string parameter | Single-line |
| **textarea** | string parameter | Multi-line |
| **switch** | bool parameter | On / Off |
| **select** | string parameter | User-defined option list |

The **parameter name** field must match exactly the name in the function signature.

### Required packages

The **Required packages** field lists pip-style dependencies (one per line). These become
[PEP 723 inline script metadata](https://peps.python.org/pep-0723/) and are resolved by
`uv run --no-project --isolated` at run time. `uv` caches the resolved environment; the
first run may take a few seconds while packages download. rengu's own venv is never
touched.

### Run record

Each tool keeps a **single last-run record**: the inputs you used, the exit status, and
the full log output. This record is overwritten on every run — there is no history. The
log updates live during the run (REST snapshot + WebSocket stream).

## Example: a two-number adder

**Script**

```python
def add(num1: float, num2: float) -> None:
    print(f"{num1} + {num2} = {num1 + num2}")
```

**Tool configuration**

| Field | Value |
|-------|-------|
| Entrypoint | `add` |
| Required packages | *(empty)* |

**Declared inputs**

| Label | Parameter | Control type |
|-------|-----------|--------------|
| First number | `num1` | number |
| Second number | `num2` | number |

Enter `3` and `7`, click **Run**, and the log shows:

```
3.0 + 7.0 = 10.0
```
