"""GPU arbitration: one SQLite row per device, whose PRIMARY KEY is the mutex.

Inserting over an occupied ``device`` raises ``sqlite3.IntegrityError`` — a free, atomic
compare-and-swap that is correct even across processes. In-process memory cannot describe a
resource that outlives the process holding it (a detached child survives a server restart *still
using the GPU*), and a lock file would mean re-inventing ``O_EXCL``, parsing and stale cleanup that
SQLite already provides.

**Reaping is by holder validity, not by timing.** A lease is never freed because a timer expired;
it is freed because its holder is provably gone — the job row is missing or terminal, the workflow
row was deleted, or the bound pid is dead, a zombie, or reused. There is deliberately no timeout on a ``pid IS NULL`` lease: the
window between :func:`acquire` and :func:`bind_pid` contains ``ensure_training_extras`` ->
``uv sync``, which on a cold extra takes minutes. A timer there would free the lease mid-install and
let training spawn with no lease at all — the exact regression this table exists to prevent. An
unbound lease is legitimate for as long as its job row says ``pending``; it simply cannot survive
the process that created it, which is what :func:`reconcile_on_start` is for.
"""

from __future__ import annotations

import sqlite3

from rengu_flow.platform_compat import pid_alive
from rengu_flow_ui import db, workflow_db
from rengu_flow_ui._time import now_utc_iso

#: Sentinel device for a host with no enumerable GPU. Arbitration still works there: every holder
#: contends for the same single row.
HOST_DEVICE = -1

#: Job states in which a lease is legitimately held. Anything else — terminal, back to a ``new``
#: draft, or a deleted row — means the holder is gone. Expressed as an allow-list so a state added
#: later defaults to *reap*, never to *leak*.
_LEASE_HOLDING_STATES = frozenset({"pending", "running", "stopping"})

_JOB_PREFIX = "job:"

#: ``wf:<workflow_id>:<node_id>`` — the workflow lane's holder id (``workflow_runner._holder_id``).
_WF_PREFIX = "wf:"

_devices_cache: list[int] | None = None


def _connect() -> sqlite3.Connection:
    """Same DB file and same pragmas (WAL, busy_timeout) as every other UI connection."""
    return db._connect()


# ------------------------------------------------------------------------------ enumeration


def enumerate_devices() -> list[int]:
    """Physical GPU indices on this host, cached for the life of the process.

    Topology does not change while the server lives, and this sits in the queue tick's hot path
    where spawning ``nvidia-smi`` every second is unacceptable (~1.6 s cold). A host with no
    enumerable GPU reports ``[-1]`` so arbitration still has exactly one row to contend for.
    """
    global _devices_cache
    if _devices_cache is None:
        from rengu_track import system_stats

        found = [int(d["index"]) for d in system_stats.list_gpu_devices()]
        _devices_cache = found or [HOST_DEVICE]
    return list(_devices_cache)


def reset_device_cache() -> None:
    """Force re-enumeration on the next call (tests, driver reload)."""
    global _devices_cache
    _devices_cache = None
    from rengu_track import system_stats

    system_stats.reset_device_cache()


def _resolve_devices(devices: list[int] | None) -> list[int]:
    """The device set an acquisition actually takes: ``None``/empty means host-exclusive.

    Every requested index is validated against :func:`enumerate_devices`; an unknown one falls back
    to host-exclusive. Without that check, a host where enumeration failed caches ``[-1]``, an
    "auto" holder takes ``-1``, a node pinned to ``device: 0`` takes ``0``, the two do not conflict,
    and both land on the same physical GPU.
    """
    known = enumerate_devices()
    if not devices:
        return known
    wanted = sorted({int(d) for d in devices})
    if any(d not in known for d in wanted):
        return known
    return wanted


# ------------------------------------------------------------------------------ acquire / release


def acquire(holder_kind: str, holder_id: str, devices: list[int] | None) -> bool:
    """Take ``devices`` (or every enumerated device when ``None``) for ``holder_id``.

    All-or-nothing: the multi-row insert runs inside ``with conn:`` so a conflict on any device
    rolls the whole acquisition back. Under autocommit an "auto" acquisition that collides on the
    second GPU would leave the first one taken forever.

    Not idempotent: the table is a mutex, not a refcount, so re-acquiring a device this same holder
    already owns fails like any other conflict (and changes nothing).
    """
    rows = [
        (device, holder_kind, holder_id, now_utc_iso())
        for device in _resolve_devices(devices)
    ]
    conn = _connect()
    try:
        with conn:
            conn.executemany(
                "INSERT INTO gpu_leases"
                " (device, holder_kind, holder_id, pid, pid_create_time, acquired_at)"
                " VALUES (?, ?, ?, NULL, NULL, ?)",
                rows,
            )
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()
    return True


def bind_pid(
    holder_id: str, pid: int | None, pid_create_time: float | None = None
) -> bool | None:
    """Attach the launched process to ``holder_id``'s lease. Three outcomes, all distinct:

    * ``True`` — bound.
    * ``False`` — the UPDATE matched zero rows, i.e. the lease vanished underneath the launch
      (it was reaped while ``start_job`` ran). The caller is then holding a process with no
      lease, another holder may already own the GPU, and it **must kill what it just spawned**.
    * ``None`` — there was no pid to bind, so nothing was attempted and nothing is wrong with
      the lease. Deliberately NOT ``False``: ``False`` is a kill-the-process signal, and
      conflating "no pid was reported" with "the lease evaporated" would kill on a launcher
      that simply does not return a pid. The lease stays unbound for
      :func:`reconcile_on_start` to sweep.

    ``pid_create_time`` is captured here from ``psutil`` when not supplied: without it a PID
    reused after a reboot leaves the lease taken forever.
    """
    if pid is None:
        return None
    if pid_create_time is None:
        pid_create_time = _pid_create_time(pid)
    conn = _connect()
    try:
        with conn:
            cur = conn.execute(
                "UPDATE gpu_leases SET pid = ?, pid_create_time = ? WHERE holder_id = ?",
                (int(pid), pid_create_time, holder_id),
            )
        return cur.rowcount > 0
    finally:
        conn.close()


def release(holder_id: str) -> None:
    """Free every device held by ``holder_id``. A no-op when it holds none."""
    conn = _connect()
    try:
        with conn:
            conn.execute("DELETE FROM gpu_leases WHERE holder_id = ?", (holder_id,))
    finally:
        conn.close()


# ------------------------------------------------------------------------------ reaping


def _pid_create_time(pid: int) -> float | None:
    try:
        import psutil

        return float(psutil.Process(int(pid)).create_time())
    except Exception:  # noqa: BLE001 - psutil missing, process already gone, access denied
        return None


def _pid_is_gone(pid: int, stored_create_time: float | None) -> bool:
    """True when ``pid`` is dead, a zombie, or a *different* process that reused the number."""
    if not pid_alive(pid):
        return True
    try:
        import psutil
    except ImportError:
        return False  # cannot refine without psutil; pid_alive said it is alive
    try:
        proc = psutil.Process(int(pid))
        # os.kill(pid, 0) returns True for a zombie and nothing ever wait()s these detached
        # children, so without this check a finished holder keeps its lease indefinitely.
        if proc.status() == psutil.STATUS_ZOMBIE:
            return True
        if stored_create_time is not None:
            return abs(proc.create_time() - float(stored_create_time)) >= 1.0
    except psutil.NoSuchProcess:
        return True
    except psutil.Error:
        return False  # access denied and friends: assume alive rather than steal the GPU
    return False


def _job_row_is_gone(job_id: str) -> bool:
    try:
        job = db.get_job(job_id)
    except KeyError:
        return True
    return job.state not in _LEASE_HOLDING_STATES


def _workflow_row_is_gone(holder_id: str) -> bool:
    """True when the workflow a ``wf:`` lease belongs to no longer exists.

    Same reconciliation the training lane gets from :func:`_job_row_is_gone`, and for the same
    reason: ``DELETE /workflows/{id}`` guards on ``running``/``cancelling``, but a lease taken in
    the window before the node's status is written — or a row removed by any path that never goes
    through the route — would otherwise be freed by nothing. ``reconcile_on_start`` cannot help:
    it walks ``list_workflows()``, so a deleted row is never visited, and a lease still inside the
    ``acquire`` -> ``bind_pid`` window (minutes, on a cold extras install) has ``pid IS NULL`` and
    is deliberately never timed out. Without this, restarting the server is the only thing that
    frees it, and the training lane cannot acquire until then.

    **Existence only, never the run status.** A node's status lives in ``state_json`` and is
    written around the launch, so reaping on "not running" would free the lease of a node that is
    mid-spawn — the exact regression the no-timeout rule exists to prevent.
    """
    workflow_id = holder_id[len(_WF_PREFIX):].split(":", 1)[0]
    try:
        workflow_db.get_workflow(workflow_id)
    except KeyError:  # deleted, or an unparseable id that can never match a row
        return True
    return False


def _holder_is_gone(holder_id: str, pid: int | None, pid_create_time: float | None) -> bool:
    if holder_id.startswith(_JOB_PREFIX) and _job_row_is_gone(holder_id[len(_JOB_PREFIX):]):
        return True
    if holder_id.startswith(_WF_PREFIX) and _workflow_row_is_gone(holder_id):
        return True
    if pid is None:
        return False  # unbound but still legitimate: the launch has not reached bind_pid yet
    return _pid_is_gone(pid, pid_create_time)


def _holders() -> list[tuple[str, int | None, float | None]]:
    conn = _connect()
    try:
        rows = conn.execute(
            "SELECT holder_id, pid, pid_create_time FROM gpu_leases"
        ).fetchall()
    finally:
        conn.close()
    seen: dict[str, tuple[str, int | None, float | None]] = {}
    for row in rows:
        seen.setdefault(
            row["holder_id"],
            (row["holder_id"], row["pid"], row["pid_create_time"]),
        )
    return list(seen.values())


def reap_dead() -> list[str]:
    """Free every lease whose holder is provably gone. Returns the freed holder ids.

    This — not the eager ``release()`` in ``poll_job`` — is what makes the training lane leak-proof.
    A job reaches a terminal state through at least five paths that never touch ``poll_job``
    (``stop_job`` with no pid, the config-missing branch of ``try_start_next``, ``delete_job_record``
    and ``dequeue_job`` on a pending row, a hard server kill). Sprinkling ``release()`` across those
    call sites is a checklist that will be wrong the first time a sixth is added; reconciling
    against the job row is correct by construction.
    """
    freed = [
        holder_id
        for holder_id, pid, create_time in _holders()
        if _holder_is_gone(holder_id, pid, create_time)
    ]
    for holder_id in freed:
        release(holder_id)
    return freed


def reconcile_on_start() -> list[str]:
    """Startup sweep: :func:`reap_dead` plus every unbound lease. Returns the freed holder ids.

    A ``pid IS NULL`` lease cannot survive the process that created it — the launch it belonged to
    never happened.
    """
    freed = reap_dead()
    conn = _connect()
    try:
        with conn:
            rows = conn.execute(
                "SELECT DISTINCT holder_id FROM gpu_leases WHERE pid IS NULL"
            ).fetchall()
            conn.execute("DELETE FROM gpu_leases WHERE pid IS NULL")
    finally:
        conn.close()
    return freed + [row["holder_id"] for row in rows]


# ------------------------------------------------------------------------------ inspection


def snapshot() -> list[dict]:
    """Every live lease, ordered by device."""
    conn = _connect()
    try:
        rows = conn.execute(
            "SELECT device, holder_kind, holder_id, pid, pid_create_time, acquired_at"
            " FROM gpu_leases ORDER BY device"
        ).fetchall()
    finally:
        conn.close()
    return [dict(row) for row in rows]


def _device_label(device: int) -> str:
    return "the GPU" if device == HOST_DEVICE else f"GPU {device}"


def _holder_label(row: dict) -> str:
    holder_id = row["holder_id"]
    if row["holder_kind"] == "train" and holder_id.startswith(_JOB_PREFIX):
        return f"training job {holder_id[len(_JOB_PREFIX):]}"
    return holder_id


def wait_reason(devices: list[int] | None) -> str:
    """Human-readable explanation of why ``devices`` cannot be acquired, or ``""`` if they can."""
    wanted = _resolve_devices(devices)
    held = {row["device"]: row for row in snapshot()}
    blocking = [held[device] for device in wanted if device in held]
    if not blocking:
        return ""
    row = blocking[0]
    return f"Waiting for {_device_label(row['device'])} — held by {_holder_label(row)}."
