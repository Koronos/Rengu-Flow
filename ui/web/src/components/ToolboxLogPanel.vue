<!--
  A tool run's console: the live log, its status and its exit code.

  Extracted from `ToolboxRunPanel.vue` — the `.output` block plus the whole streaming stack behind
  it (WebSocket first, byte-offset REST polling as the fallback, ANSI SGR stripping). The panel
  owns the run status because the status *is* what the stream reports; the parent reads it back
  through `update:status` for its Run/Cancel buttons.

  Initialisation is driven by `lastRun`, which the parent already fetched with the tool: `undefined`
  means "not loaded yet" and the panel waits, `null` means "loaded, never run". A run that is still
  `running` is streamed from the start over the socket (no REST snapshot, which would duplicate the
  first chunk); anything else loads its final log once over REST.
-->
<template>
  <div class="output">
    <div class="output__bar">
      <span class="output__title">Output</span>
      <el-tag v-if="showStatus && status" :type="statusType" size="small" effect="dark">
        {{ status }}
      </el-tag>
      <span v-if="exitInfo" class="output__meta">{{ exitInfo }}</span>
    </div>
    <pre v-if="cleanLog" class="output__log">{{ cleanLog }}</pre>
    <div v-else class="output__empty">{{ emptyLabel }}</div>
  </div>
</template>

<script setup lang="ts">
import { computed, onUnmounted, ref, watch } from "vue";
import { api, type ToolboxRun } from "../api";
import { wsBaseUrl } from "../lib/wsLog";

const props = defineProps<{
  toolId: string;
  /**
   * The tool's `last_run`. Left `undefined` while the parent is still loading the tool — that is
   * the signal to wait; `null` means "loaded, never run" and is acted on immediately.
   */
  lastRun?: ToolboxRun | null;
  /** Render the status tag inside the output bar (the Toolbox page shows its own, in the head). */
  showStatus?: boolean;
  emptyText?: string;
}>();

const emptyLabel = computed(() => props.emptyText || "Run the tool to see output here.");

const emit = defineEmits<{
  (e: "update:status", status: string): void;
}>();

const log = ref("");
const status = ref("");
const exitCode = ref<number | null | undefined>(undefined);

const statusType = computed(() =>
  status.value === "done" ? "success" : status.value === "failed" ? "danger" : "info",
);

// Strip ANSI SGR color codes so the log reads as plain text. Built from a string rather than a
// literal so the ESC byte stays an escape sequence in the source instead of a raw control char.
const ANSI_SGR = new RegExp("\\u001b\\[[0-9;]*m", "g");
const cleanLog = computed(() => log.value.replace(ANSI_SGR, ""));
const exitInfo = computed(() => {
  if (status.value !== "done" && status.value !== "failed") return "";
  return exitCode.value === null || exitCode.value === undefined
    ? status.value
    : `exit ${exitCode.value}`;
});

watch(status, (value) => emit("update:status", value));

let ws: WebSocket | null = null;
let pollTimer: ReturnType<typeof setTimeout> | null = null;
let pollOffset = 0;

function isTerminal(s: string) {
  return s !== "" && s !== "running" && s !== "idle";
}

function stopPoll() {
  if (pollTimer !== null) {
    clearTimeout(pollTimer);
    pollTimer = null;
  }
}

async function loadSnapshot() {
  const r = await api.toolboxLog(props.toolId, 0);
  log.value = r.chunk;
  status.value = r.status;
}

async function pollLog() {
  try {
    const r = await api.toolboxLog(props.toolId, pollOffset);
    if (r.chunk) log.value += r.chunk;
    pollOffset = r.offset;
    status.value = r.status;
    if (isTerminal(r.status)) {
      stopPoll();
      await refreshStatus();
      return;
    }
  } catch {
    // network hiccup — keep polling
  }
  pollTimer = setTimeout(pollLog, 1500);
}

function startPollFallback(fromOffset: number) {
  stopPoll();
  pollOffset = fromOffset;
  pollTimer = setTimeout(pollLog, 1500);
}

function openWs() {
  ws?.close();
  ws = new WebSocket(`${wsBaseUrl()}/api/v1/toolbox/tools/${encodeURIComponent(props.toolId)}/log/ws`);
  let wsOffset = 0;
  ws.onmessage = (ev) => {
    const chunk = ev.data as string;
    log.value += chunk;
    wsOffset += new TextEncoder().encode(chunk).length;
  };
  ws.onerror = () => {
    if (status.value === "running") startPollFallback(wsOffset);
  };
  ws.onclose = () => {
    if (status.value === "running") startPollFallback(wsOffset);
    else refreshStatus();
  };
}

async function refreshStatus() {
  const r = await api.toolboxRunStatus(props.toolId);
  status.value = r.status;
  exitCode.value = r.exit_code;
}

/** Called right after a run is launched: clear the console and stream the new run. */
function start(): void {
  log.value = "";
  exitCode.value = undefined;
  status.value = "running";
  openWs();
}

let initialized = false;

watch(
  () => props.lastRun,
  async (lastRun) => {
    if (initialized || lastRun === undefined) return;
    initialized = true;
    exitCode.value = lastRun?.exit_code;
    if (lastRun?.status === "running") {
      // Stream the live log from the start via WS (no REST snapshot — avoids duplication).
      log.value = "";
      status.value = "running";
      openWs();
    } else {
      // Idle/done/failed: load the final log once via REST.
      await loadSnapshot();
    }
  },
  { immediate: true },
);

onUnmounted(() => {
  ws?.close();
  stopPoll();
});

defineExpose({ start, refreshStatus });
</script>

<style scoped>
.output {
  border: 1px solid var(--el-border-color-lighter);
  border-radius: var(--el-border-radius-base);
  overflow: hidden;
}
.output__bar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: var(--rf-space-sm);
  padding: 6px 10px;
  background: var(--el-fill-color-light);
  border-bottom: 1px solid var(--el-border-color-lighter);
}
.output__title {
  font-size: 12px;
  font-weight: 600;
  letter-spacing: 0.04em;
  text-transform: uppercase;
  color: var(--el-text-color-secondary);
  margin-right: auto;
}
.output__meta {
  font-size: 12px;
  font-family: var(--rf-font-mono);
  color: var(--el-text-color-secondary);
}
.output__log {
  margin: 0;
  padding: var(--rf-space-sm);
  font-family: var(--rf-font-mono);
  font-size: 12px;
  line-height: 1.55;
  white-space: pre-wrap;
  word-break: break-word;
  max-height: 320px;
  overflow: auto;
  color: var(--el-text-color-primary);
}
.output__empty {
  padding: var(--rf-space-md);
  text-align: center;
  font-size: 13px;
  color: var(--el-text-color-secondary);
}
</style>
