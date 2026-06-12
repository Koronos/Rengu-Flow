"""rengu_track — local experiment-tracking core (hub).

Decoupled from rengu_flow / rengu_flow_ui: both are clients. Producers feed data through a
``MetricsSink`` (the connectable/disconnectable client); the store is the local filesystem
(``run_dir``) — TensorBoard event files for time-series, ``run.json`` for structured metadata,
``run_events.jsonl`` for the lifecycle/config timeline. No daemon, no HTTP API in the producer
path.

Importing this package is cheap: backend classes defer their heavy imports (torch, wandb) until
a sink is actually built.
"""

from rengu_track.events import (
    EVENT_CONFIG_EDITED,
    EVENT_CONFIG_RELOADED,
    EVENT_FAILED,
    EVENT_FINISHED,
    EVENT_RESTARTED_FROM_SCRATCH,
    EVENT_RESUMED,
    EVENT_RUN_STARTED,
    EVENT_STOP_REQUESTED,
    append_event,
    config_diff,
    read_events,
)
from rengu_track.run import RunManifest, flatten_hparams, read_manifest, write_manifest
from rengu_track.sink import MetricsSink, NullSink, build_sink

__all__ = [
    "MetricsSink",
    "NullSink",
    "build_sink",
    "RunManifest",
    "read_manifest",
    "write_manifest",
    "flatten_hparams",
    "append_event",
    "read_events",
    "config_diff",
    "EVENT_RUN_STARTED",
    "EVENT_RESUMED",
    "EVENT_RESTARTED_FROM_SCRATCH",
    "EVENT_CONFIG_RELOADED",
    "EVENT_CONFIG_EDITED",
    "EVENT_STOP_REQUESTED",
    "EVENT_FINISHED",
    "EVENT_FAILED",
]
