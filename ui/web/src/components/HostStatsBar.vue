<template>
  <div class="host-stats-root" v-bind="$attrs">
    <div class="host-stats" @click="drawerOpen = true">
    <template v-if="loading && !stats">
      <span class="chip muted">Host…</span>
    </template>
    <template v-else-if="stats">
      <span class="chip chip-meter" :class="heatClass(summary.cpu_temp_c)">
        <span
          class="chip-fill"
          :class="levelClass(summary.cpu_percent)"
          :style="{ width: `${clampPct(summary.cpu_percent)}%` }"
        />
        <span class="chip-text">
          <span class="lbl">CPU</span>
          {{ fmtPct(summary.cpu_percent) }}
          <span v-if="summary.cpu_temp_c != null" class="sep">·</span>
          <span v-if="summary.cpu_temp_c != null">{{ fmtTemp(summary.cpu_temp_c) }}</span>
        </span>
      </span>
      <span class="chip chip-meter" :class="loadClass(summary.ram_percent)">
        <span
          class="chip-fill"
          :class="levelClass(summary.ram_percent)"
          :style="{ width: `${clampPct(summary.ram_percent)}%` }"
        />
        <span class="chip-text">
          <span class="lbl">RAM</span>
          {{ fmtGb(summary.ram_used_gb) }}/{{ fmtGb(summary.ram_total_gb) }}
          <span class="pct">({{ fmtPct(summary.ram_percent) }})</span>
        </span>
      </span>
      <span
        v-for="gpu in summary.gpus || []"
        :key="gpu.index"
        class="chip chip-meter"
        :class="[heatClass(gpu.temp_c), loadClass(gpuVramPct(gpu))]"
      >
        <span
          class="chip-fill"
          :class="levelClass(gpuVramPct(gpu))"
          :style="{ width: `${clampPct(gpuVramPct(gpu))}%` }"
        />
        <span class="chip-text">
          <span class="lbl">GPU{{ gpu.index }}</span>
          {{ fmtPct(gpu.util_percent) }}
          <span class="sep">·</span>
          VRAM {{ fmtGb(gpu.vram_used_gb) }}/{{ fmtGb(gpu.vram_total_gb) }}
          <span v-if="gpu.temp_c != null" class="sep">·</span>
          <span v-if="gpu.temp_c != null">{{ fmtTemp(gpu.temp_c) }}</span>
        </span>
      </span>
      <span v-if="!summary.gpus?.length && gpuHint" class="chip muted" title="GPU metrics need nvidia-smi">
        GPU —
      </span>
    </template>
    <span v-else class="chip muted">Host unavailable</span>
    <el-icon class="expand-hint"><ArrowDown /></el-icon>
    </div>

    <el-drawer
    v-model="drawerOpen"
    title="Host metrics"
    direction="rtl"
    :size="isMobile ? '100%' : '420px'"
    class="host-stats-drawer"
  >
    <div v-if="stats" class="detail-body">
      <p class="updated">Updated {{ updatedLabel }}</p>
      <el-alert
        v-for="(w, i) in warnings"
        :key="i"
        type="warning"
        :title="w"
        show-icon
        :closable="false"
        class="mb-8"
      />

      <h4>CPU</h4>
      <div class="meter-block mb-8">
        <div class="meter-label">
          <span>Load</span>
          <span>{{ fmtPct(detail.cpu?.percent) }}</span>
        </div>
        <div class="meter-track">
          <div
            class="meter-fill"
            :class="levelClass(detail.cpu?.percent)"
            :style="{ width: `${clampPct(detail.cpu?.percent)}%` }"
          />
        </div>
      </div>
      <el-descriptions :column="1" border size="small" class="mb-16">
        <el-descriptions-item v-if="detail.cpu?.temp_c != null" label="Temperature">
          {{ fmtTemp(detail.cpu.temp_c) }}
        </el-descriptions-item>
        <el-descriptions-item label="Cores">
          {{ detail.cpu?.physical_count ?? "?" }} physical /
          {{ detail.cpu?.logical_count ?? "?" }} logical
        </el-descriptions-item>
        <el-descriptions-item
          v-if="detail.cpu?.freq_mhz?.current"
          label="Frequency"
        >
          {{ detail.cpu.freq_mhz.current }} MHz
        </el-descriptions-item>
      </el-descriptions>

      <div v-if="detail.cpu?.per_core?.length" class="core-grid mb-16">
        <div
          v-for="(pct, i) in detail.cpu.per_core"
          :key="i"
          class="core-cell"
          :title="`Core ${i}: ${pct}%`"
        >
          <div class="core-bar" :style="{ height: `${Math.min(100, pct)}%` }" />
          <span class="core-idx">{{ i }}</span>
        </div>
      </div>

      <el-table
        v-if="detail.cpu?.temperatures?.length"
        :data="detail.cpu.temperatures"
        size="small"
        border
        class="mb-16"
      >
        <el-table-column prop="label" label="Sensor" min-width="120" />
        <el-table-column label="°C" width="72">
          <template #default="{ row }">{{ row.current_c }}</template>
        </el-table-column>
      </el-table>

      <h4>Memory</h4>
      <div class="meter-block mb-8">
        <div class="meter-label">
          <span>RAM</span>
          <span>{{ fmtGb(detail.ram?.used_gb) }} / {{ fmtGb(detail.ram?.total_gb) }} ({{ fmtPct(detail.ram?.percent) }})</span>
        </div>
        <div class="meter-track">
          <div
            class="meter-fill"
            :class="levelClass(detail.ram?.percent)"
            :style="{ width: `${clampPct(detail.ram?.percent)}%` }"
          />
        </div>
      </div>
      <div v-if="detail.ram?.swap?.total_gb" class="meter-block mb-16">
        <div class="meter-label">
          <span>Swap</span>
          <span>{{ fmtGb(detail.ram.swap.used_gb) }} / {{ fmtGb(detail.ram.swap.total_gb) }} ({{ fmtPct(detail.ram.swap.percent) }})</span>
        </div>
        <div class="meter-track">
          <div
            class="meter-fill"
            :class="levelClass(detail.ram.swap.percent)"
            :style="{ width: `${clampPct(detail.ram.swap.percent)}%` }"
          />
        </div>
      </div>

      <h4>GPUs</h4>
      <template v-if="detail.gpus?.devices?.length">
        <el-card
          v-for="gpu in detail.gpus.devices"
          :key="gpu.index"
          shadow="never"
          class="gpu-card mb-8"
        >
          <template #header>GPU {{ gpu.index }} — {{ gpu.name }}</template>
          <div class="meter-block mb-8">
            <div class="meter-label">
              <span>VRAM</span>
              <span>{{ fmtGb(gpu.vram_used_gb) }} / {{ fmtGb(gpu.vram_total_gb) }} ({{ fmtPct(gpu.vram_percent) }})</span>
            </div>
            <div class="meter-track">
              <div
                class="meter-fill"
                :class="levelClass(gpu.vram_percent)"
                :style="{ width: `${clampPct(gpu.vram_percent)}%` }"
              />
            </div>
          </div>
          <div class="meter-block mb-8">
            <div class="meter-label">
              <span>GPU load</span>
              <span>{{ fmtPct(gpu.util_percent) }}</span>
            </div>
            <div class="meter-track">
              <div
                class="meter-fill"
                :class="levelClass(gpu.util_percent)"
                :style="{ width: `${clampPct(gpu.util_percent)}%` }"
              />
            </div>
          </div>
          <el-descriptions :column="1" border size="small">
            <el-descriptions-item v-if="gpu.temp_c != null" label="Temperature">
              {{ fmtTemp(gpu.temp_c) }}
            </el-descriptions-item>
            <el-descriptions-item v-if="gpu.power_w != null" label="Power">
              {{ gpu.power_w }} W
            </el-descriptions-item>
            <el-descriptions-item v-if="gpu.fan_percent != null" label="Fan">
              {{ gpu.fan_percent }}%
            </el-descriptions-item>
            <el-descriptions-item v-if="gpu.clock_sm_mhz != null" label="SM clock">
              {{ gpu.clock_sm_mhz }} MHz
            </el-descriptions-item>
          </el-descriptions>
        </el-card>
      </template>
      <el-empty v-else :description="detail.gpus?.error || 'No GPU data'" :image-size="48" />
    </div>
    <el-skeleton v-else :rows="6" animated />
    </el-drawer>
  </div>
</template>

<script setup>
import { computed, onMounted, onUnmounted, ref } from "vue";

defineOptions({ inheritAttrs: false });
import { ArrowDown } from "@element-plus/icons-vue";
import { api } from "../api";
import { useBreakpoint } from "../composables/useBreakpoint";

const POLL_MS = 2000;

const { isMobile } = useBreakpoint();
const stats = ref(null);
const loading = ref(true);
const drawerOpen = ref(false);
let timer = null;

const summary = computed(() => stats.value?.summary || {});
const detail = computed(() => stats.value?.detail || {});
const warnings = computed(() => detail.value.warnings || []);
const gpuHint = computed(() => detail.value.gpus?.error);

const updatedLabel = computed(() => {
  const ts = stats.value?.ts;
  if (!ts) return "—";
  return new Date(ts * 1000).toLocaleTimeString();
});

function fmtPct(v) {
  if (v == null || Number.isNaN(v)) return "—";
  return `${Math.round(v)}%`;
}

function fmtGb(v) {
  if (v == null || Number.isNaN(v)) return "—";
  return `${v}G`;
}

function fmtTemp(v) {
  if (v == null) return "";
  return `${Math.round(v)}°C`;
}

function clampPct(v) {
  if (v == null || Number.isNaN(v)) return 0;
  return Math.min(100, Math.max(0, v));
}

function levelClass(pct) {
  const p = clampPct(pct);
  if (p >= 90) return "level-crit";
  if (p >= 75) return "level-warn";
  if (p >= 50) return "level-mid";
  return "level-ok";
}

function gpuVramPct(gpu) {
  return gpu?.vram_percent ?? (gpu?.vram_used_gb && gpu?.vram_total_gb
    ? (100 * gpu.vram_used_gb) / gpu.vram_total_gb
    : null);
}

function heatClass(temp) {
  if (temp == null) return "";
  if (temp >= 85) return "hot";
  if (temp >= 75) return "warm";
  return "";
}

function loadClass(pct) {
  if (pct == null) return "";
  if (pct >= 95) return "hot";
  if (pct >= 80) return "warm";
  return "";
}

async function poll() {
  try {
    stats.value = await api.getSystemStats();
  } catch {
    if (!stats.value) stats.value = null;
  } finally {
    loading.value = false;
  }
}

onMounted(() => {
  poll();
  timer = setInterval(poll, POLL_MS);
});

onUnmounted(() => {
  if (timer) clearInterval(timer);
});
</script>

<style scoped>
.host-stats {
  display: flex;
  align-items: center;
  gap: 6px;
  flex-shrink: 0;
  max-width: min(100%, 720px);
  overflow-x: auto;
  cursor: pointer;
  padding: 4px 8px;
  border-radius: var(--el-border-radius-base);
  scrollbar-width: thin;
}
.host-stats:hover {
  background: var(--el-fill-color-light);
}
.chip {
  flex-shrink: 0;
  font-size: 11px;
  font-family: ui-monospace, monospace;
  border-radius: 4px;
  border: 1px solid var(--el-border-color-lighter);
  white-space: nowrap;
}
.chip-meter {
  position: relative;
  overflow: hidden;
  background: var(--el-fill-color-darker);
  min-width: 4.5rem;
}
.chip-fill {
  position: absolute;
  left: 0;
  top: 0;
  bottom: 0;
  z-index: 0;
  transition: width 0.45s ease;
  opacity: 0.55;
}
.chip-fill.level-ok {
  background: linear-gradient(90deg, #16a34a, #22c55e);
}
.chip-fill.level-mid {
  background: linear-gradient(90deg, #ca8a04, #eab308);
}
.chip-fill.level-warn {
  background: linear-gradient(90deg, #ea580c, #f97316);
}
.chip-fill.level-crit {
  background: linear-gradient(90deg, #dc2626, #ef4444);
}
.chip-text {
  position: relative;
  z-index: 1;
  display: inline-flex;
  align-items: center;
  gap: 4px;
  padding: 3px 8px;
}
.chip.warm {
  border-color: var(--el-color-warning);
}
.chip.hot {
  border-color: var(--el-color-danger);
}
.chip.hot .chip-text {
  color: var(--el-color-danger-light-3);
}
.chip.muted {
  opacity: 0.6;
  display: inline-flex;
  padding: 3px 8px;
}
.meter-block {
  width: 100%;
}
.meter-label {
  display: flex;
  justify-content: space-between;
  font-size: 12px;
  margin-bottom: 4px;
  color: var(--el-text-color-secondary);
}
.meter-track {
  height: 10px;
  border-radius: 5px;
  background: var(--el-fill-color-darker);
  overflow: hidden;
  border: 1px solid var(--el-border-color-lighter);
}
.meter-fill {
  height: 100%;
  border-radius: 4px;
  transition: width 0.45s ease;
  min-width: 2px;
}
.meter-fill.level-ok {
  background: linear-gradient(90deg, #16a34a, #22c55e);
}
.meter-fill.level-mid {
  background: linear-gradient(90deg, #ca8a04, #eab308);
}
.meter-fill.level-warn {
  background: linear-gradient(90deg, #ea580c, #f97316);
}
.meter-fill.level-crit {
  background: linear-gradient(90deg, #dc2626, #ef4444);
}
.lbl {
  font-weight: 600;
  color: var(--el-text-color-secondary);
  font-size: 10px;
  text-transform: uppercase;
}
.sep,
.pct {
  opacity: 0.75;
}
.expand-hint {
  flex-shrink: 0;
  opacity: 0.5;
  font-size: 12px;
}
.detail-body h4 {
  margin: 16px 0 8px;
  font-size: 13px;
  font-weight: 600;
}
.detail-body h4:first-of-type {
  margin-top: 0;
}
.updated {
  margin: 0 0 12px;
  font-size: 12px;
  color: var(--el-text-color-secondary);
}
.mb-8 {
  margin-bottom: 8px;
}
.mb-16 {
  margin-bottom: 16px;
}
.core-grid {
  display: flex;
  flex-wrap: wrap;
  gap: 4px;
  align-items: flex-end;
  max-height: 80px;
}
.core-cell {
  width: 14px;
  height: 48px;
  background: var(--el-fill-color-darker);
  border-radius: 2px;
  position: relative;
  display: flex;
  align-items: flex-end;
  overflow: hidden;
}
.core-bar {
  width: 100%;
  background: var(--el-color-primary);
  min-height: 2px;
}
.core-idx {
  position: absolute;
  bottom: 0;
  left: 0;
  right: 0;
  font-size: 7px;
  text-align: center;
  color: var(--el-text-color-secondary);
  pointer-events: none;
}
.gpu-card :deep(.el-card__header) {
  padding: 8px 12px;
  font-size: 12px;
}
</style>
