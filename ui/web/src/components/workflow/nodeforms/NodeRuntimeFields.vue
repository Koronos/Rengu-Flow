<!--
  The runtime block every non-`folder` node shares: where its input comes from, whether it takes a
  GPU, whether it queues for one, which one, and whether it runs at all.

  These five fields are *not* stage configuration — they live on the node (`from`, `gpu`,
  `enabled`), not in `config` — which is why they are one component instead of five copies inside
  each per-type form.

  The device list comes from the system-stats WebSocket `HostStatsBar.vue` already consumes: the
  picker deliberately adds **no endpoint**, per the spec's "Device selection" section.
-->
<template>
  <el-form label-position="top" class="node-runtime" :disabled="disabled">
    <!-- From ------------------------------------------------------------------ -->
    <el-form-item v-if="consumes" label="From">
      <el-select
        v-model="fromId"
        :clearable="sourceOptional"
        placeholder="Pick the step this one reads"
        class="w-full"
      >
        <el-option
          v-for="source in sources"
          :key="source.id"
          :label="`${ordinalGlyph(ordinal[source.id])} ${source.title}`"
          :value="source.id"
        >
          <span class="opt-title">
            {{ ordinalGlyph(ordinal[source.id]) }} {{ source.title }}
          </span>
          <span class="opt-path">{{ sourcePaths[source.id] || "folder not resolved yet" }}</span>
        </el-option>
      </el-select>
      <el-text v-if="!sources.length" size="small" type="warning" class="hint-text">
        No earlier step emits a folder. Add a source folder above this one.
      </el-text>
      <el-text v-else-if="fromId" size="small" type="info" class="hint-text">
        {{ sourcePaths[fromId] || "This step has not produced a folder yet." }}
      </el-text>
    </el-form-item>

    <!-- Needs GPU -------------------------------------------------------------- -->
    <el-form-item label="Needs GPU">
      <el-switch v-model="needsGpu" />
      <el-text class="ml-8" size="small">Run this step on the GPU</el-text>
      <el-text size="small" type="info" class="hint-text">
        Off means the step never asks for a GPU lease, so it runs alongside training.
        <template v-if="node.type === 'prep.quality'">
          Quality filtering follows its metric: <code>blur</code> is pure CPU, the model-based
          metrics are not.
        </template>
      </el-text>
    </el-form-item>

    <!-- Wait for the GPU queue -------------------------------------------------- -->
    <el-form-item v-if="needsGpu" label="Wait for the GPU queue">
      <el-switch v-model="waitForQueue" />
      <el-text class="ml-8" size="small">Queue behind whatever holds the GPU</el-text>
      <el-text size="small" type="info" class="hint-text">
        Runs only when no other job holds the GPU. Turn it off to start immediately alongside a
        training run — both will share VRAM.
      </el-text>
      <el-text size="small" type="info" class="hint-text">
        Waiting retries opportunistically and has no ordering guarantee, so a stream of short
        training jobs can keep this step waiting indefinitely.
      </el-text>
    </el-form-item>

    <!-- Device ------------------------------------------------------------------ -->
    <el-form-item v-if="needsGpu" label="Device">
      <el-select v-model="device" class="w-full" placeholder="Auto">
        <el-option label="Auto — let the app pick" :value="AUTO_DEVICE" />
        <el-option
          v-for="dev in devices"
          :key="dev.index"
          :label="dev.label"
          :value="dev.index"
        />
      </el-select>
      <el-text size="small" type="info" class="hint-text">
        Sets <code>CUDA_VISIBLE_DEVICES</code> for this step only. With the queue toggle on it
        still waits its turn, so picking a second GPU buys parallelism only with the toggle off.
      </el-text>
      <el-text v-if="!devices.length" size="small" type="warning" class="hint-text">
        No GPUs reported yet — the list fills in from the host stats stream.
      </el-text>
    </el-form-item>

    <!-- Enabled ----------------------------------------------------------------- -->
    <el-form-item label="Enabled">
      <el-switch v-model="enabled" />
      <el-text class="ml-8" size="small">Include this step when the workflow runs</el-text>
      <el-text size="small" type="info" class="hint-text">
        A disabled step is skipped, and the step below it reads this one's last saved folder
        instead. Without one, the workflow refuses to start rather than failing mid-run.
      </el-text>
    </el-form-item>
  </el-form>
</template>

<script setup lang="ts">
import { computed, ref, watch } from "vue";
import type { PropType } from "vue";
import { legalSources, ordinalGlyph, ordinals } from "../../../lib/workflowGraph";
import { consumesInput, defaultNeedsGpu, sourceMayBeEmpty } from "../../../lib/workflowNodeTypes";
import { useSystemStatsStream } from "../../../composables/useSystemStatsStream";
import type { WorkflowGraph, WorkflowNode, WorkflowNodeGpu } from "../../../types/workflow";

/** `el-select` cannot bind `null`, so "auto" travels as a sentinel and maps back on write. */
const AUTO_DEVICE = -1;

const node = defineModel<WorkflowNode>({ required: true });

const props = defineProps({
  graph: { type: Object as PropType<WorkflowGraph>, required: true },
  /** Node id -> the folder that node emits, resolved by the drawer. Shown as the option subtitle. */
  sourcePaths: { type: Object as PropType<Record<string, string>>, default: () => ({}) },
  /**
   * Read-only: the runner owns the workflow. `el-form` hands this to every control under it, so
   * `from`, the GPU switches, the device picker and `enabled` all go inert together.
   */
  disabled: { type: Boolean, default: false },
});

function patch(changes: Partial<WorkflowNode>): void {
  node.value = { ...node.value, ...changes };
}

function patchGpu(changes: Partial<WorkflowNodeGpu>): void {
  patch({ gpu: { ...node.value.gpu, ...changes } });
}

const consumes = computed(() => consumesInput(node.value.type));
const sourceOptional = computed(() => sourceMayBeEmpty(node.value.type));
const sources = computed(() => legalSources(props.graph, node.value.id));
const ordinal = computed(() => ordinals(props.graph));

const fromId = computed<string | null>({
  get: () => node.value.from,
  set: (value) => patch({ from: value || null }),
});

const enabled = computed<boolean>({
  get: () => node.value.enabled,
  set: (value) => patch({ enabled: value }),
});

const needsGpu = computed<boolean>({
  get: () => node.value.gpu.required,
  set: (value) => {
    gpuTouched.value = true;
    patchGpu({ required: value });
  },
});

const waitForQueue = computed<boolean>({
  get: () => node.value.gpu.wait,
  set: (value) => patchGpu({ wait: value }),
});

const device = computed<number>({
  get: () => node.value.gpu.device ?? AUTO_DEVICE,
  set: (value) => patchGpu({ device: value === AUTO_DEVICE ? null : value }),
});

/**
 * `prep.quality`'s GPU default follows its metric, so flipping `blur` -> `iqa` must flip the
 * switch too — but only while the user has not overridden it. Once they have, their choice wins
 * and the metric stops moving it.
 */
const gpuTouched = ref(false);
watch(
  () => node.value.id,
  () => {
    gpuTouched.value = false;
  },
);
watch(
  () => defaultNeedsGpu(node.value.type, node.value.config),
  (fallback) => {
    if (gpuTouched.value) return;
    if (node.value.gpu.required !== fallback) patchGpu({ required: fallback });
  },
);

const { stats } = useSystemStatsStream();

const devices = computed(() => {
  const detailed = stats.value?.detail?.gpus?.devices ?? [];
  const listed = detailed.length
    ? detailed.map((gpu) => ({
        index: gpu.index ?? 0,
        label: gpu.name ? `GPU ${gpu.index} — ${gpu.name}` : `GPU ${gpu.index}`,
      }))
    : (stats.value?.summary?.gpus ?? []).map((gpu) => ({
        index: gpu.index,
        label: `GPU ${gpu.index}`,
      }));
  // A saved device the host no longer reports must still show, or the select renders blank and
  // the next write silently drops a setting the user made deliberately.
  const saved = node.value.gpu.device;
  if (saved != null && !listed.some((entry) => entry.index === saved)) {
    return [...listed, { index: saved, label: `GPU ${saved} — not reported` }];
  }
  return listed;
});
</script>

<style scoped>
.node-runtime :deep(.el-form-item) {
  margin-bottom: 18px;
}
.hint-text {
  display: block;
  margin-top: 4px;
  line-height: 1.45;
}
.ml-8 {
  margin-left: 8px;
}
.opt-title {
  margin-right: 12px;
}
.opt-path {
  float: right;
  color: var(--el-text-color-secondary);
  font-size: 12px;
  font-family: var(--rf-font-mono);
}
</style>
