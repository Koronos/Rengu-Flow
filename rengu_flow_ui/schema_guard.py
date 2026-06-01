"""Startup guard: block on an incompatible UI DB schema until the user consents.

Two kinds of schema change are handled in different places, on purpose:

* **Additive, backward-compatible** changes (a new column with a DEFAULT) are healed
  transparently by ``db.init_db`` — no version bump, no consent, no data loss.
* **Incompatible** changes (a column removed/renamed, semantics changed) bump
  ``SCHEMA_VERSION``. Those are destructive to migrate, so this guard refuses to touch the
  DB until the user agrees to wipe-and-recreate. We never silently destroy a user's library.

Call :func:`ensure_schema_compatible` once, at the terminal-attached entrypoint, *before* any
app construction or ``init_db`` work — so the user can decide before we make our mess.
"""

from __future__ import annotations

import sys


def ensure_schema_compatible() -> None:
    """Abort startup (interactively, when possible) on an incompatible DB schema version.

    Fresh and legacy-unstamped DBs, and DBs already at the current version, pass silently.
    """
    from rengu_flow_ui.db import (
        SCHEMA_VERSION,
        reset_ui_database,
        schema_action,
        stored_schema_version,
    )

    stored = stored_schema_version()
    if schema_action(stored, SCHEMA_VERSION) == "ok":
        return

    print(
        f"The UI database schema changed (file is v{stored}, this build needs "
        f"v{SCHEMA_VERSION}). The existing library (datasets, job history) is "
        "incompatible. To keep your data, abort and run "
        "`rengu-flow-ui export-library <dir>` on the previous app version, then "
        "`import-library <dir>` after upgrading."
    )
    if not sys.stdin.isatty():
        raise SystemExit(
            "Refusing to touch an incompatible database in non-interactive mode. "
            "Run `rengu ui reset-db` to wipe it, or use the previous app version."
        )
    answer = input(
        "Wipe and recreate the database now? All saved data is lost. [y/N] "
    ).strip().lower()
    if answer in ("y", "yes"):
        path = reset_ui_database()
        print(f"Recreated empty database: {path}")
    else:
        raise SystemExit("Aborted. Use the previous app version, or reset-db when ready.")
