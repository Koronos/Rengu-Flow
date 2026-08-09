<template>
  <div class="wf-bar">
    <div class="wf-bar__top">
      <el-button link :icon="ArrowLeft" @click="$emit('back')">Workflows</el-button>

      <el-input
        v-if="editingName"
        ref="nameInput"
        v-model="draftName"
        class="wf-bar__name-input"
        size="small"
        maxlength="120"
        @keydown.enter.prevent="commitName"
        @keydown.esc="editingName = false"
        @blur="commitName"
      />
      <template v-else>
        <h2 class="wf-bar__name">{{ name || "Untitled workflow" }}</h2>
        <el-button
          link
          size="small"
          :icon="EditPen"
          :disabled="readOnly"
          v-bind="ariaLabel('Rename workflow')"
          @click="startRename"
        />
      </template>

      <span class="wf-bar__spacer" />

      <el-badge :value="undefinedCount" :hidden="!undefinedCount" type="danger">
        <el-button size="small" @click="$emit('variables')">
          {x} Variables{{ variableCount ? ` ${variableCount}` : "" }}
        </el-button>
      </el-badge>

      <el-button v-if="running" type="danger" :loading="busy" @click="$emit('stop')">
        ■ Stop
      </el-button>
      <el-tooltip v-else :content="runTooltip" :disabled="!runTooltip" placement="bottom">
        <span class="wf-bar__run">
          <el-dropdown
            split-button
            type="primary"
            :disabled="runDisabled"
            @click="$emit('run')"
            @command="onCommand"
          >
            ▸ Run{{ runCount ? ` ${runCount}` : "" }}
            <template #dropdown>
              <el-dropdown-menu>
                <el-dropdown-item command="force">
                  <span class="rf-dropdown-item-label">
                    <el-icon><RefreshRight /></el-icon><span>Run all (force)</span>
                  </span>
                </el-dropdown-item>
                <el-dropdown-item command="validate">
                  <span class="rf-dropdown-item-label">
                    <el-icon><CircleCheck /></el-icon><span>Validate only</span>
                  </span>
                </el-dropdown-item>
              </el-dropdown-menu>
            </template>
          </el-dropdown>
        </span>
      </el-tooltip>

      <el-dropdown trigger="click" @command="onCommand">
        <el-button size="small" circle :icon="MoreFilled" v-bind="ariaLabel('Workflow actions')" />
        <template #dropdown>
          <el-dropdown-menu>
            <el-dropdown-item command="duplicate">
              <span class="rf-dropdown-item-label">
                <el-icon><CopyDocument /></el-icon><span>Duplicate</span>
              </span>
            </el-dropdown-item>
            <el-dropdown-item command="delete" divided :disabled="running">
              <span class="rf-dropdown-item-label rf-dropdown-item-label--danger">
                <el-icon><Delete /></el-icon><span>Delete</span>
              </span>
            </el-dropdown-item>
          </el-dropdown-menu>
        </template>
      </el-dropdown>
    </div>

    <div class="wf-bar__status">
      <span :class="`wf-bar__dot wf-bar__dot--${status}`" aria-hidden="true">●</span>
      <span>{{ statusLabel }}</span>
      <span v-if="lastRunLabel">· {{ lastRunLabel }}</span>
      <span v-if="resultPath" class="wf-bar__result">· Result: {{ resultPath }}</span>
      <span class="wf-bar__spacer" />
      <span class="wf-bar__save">{{ saveLabel }}</span>
      <span v-if="streamStatus !== 'connected'" class="wf-bar__stream">· live updates offline</span>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed, nextTick, ref, useTemplateRef } from "vue";
import {
  ArrowLeft,
  CircleCheck,
  CopyDocument,
  Delete,
  EditPen,
  MoreFilled,
  RefreshRight,
} from "@element-plus/icons-vue";
import { ariaLabel } from "../../lib/aria";
import type { WorkflowStatus } from "../../types/workflow";

/**
 * The editor's header: identity on the left, the split Run button on the right, one status line
 * underneath.
 *
 * `[ ▸ Run ]` carries the count of what it will actually do, because "Run" on a chain where five of
 * six steps are already done is a different action from "Run" on a fresh one, and the number is
 * the cheapest way to say which.
 */

const props = withDefaults(
  defineProps<{
    name: string;
    status: WorkflowStatus;
    running?: boolean;
    busy?: boolean;
    readOnly?: boolean;
    /** How many steps `[ ▸ Run ]` would execute. Zero means everything is done and fresh. */
    runCount?: number;
    /** Non-empty disables Run and is shown as the tooltip — undefined variables, no steps, … */
    blockedReason?: string;
    variableCount?: number;
    undefinedCount?: number;
    resultPath?: string;
    lastRunLabel?: string;
    saving?: boolean;
    dirty?: boolean;
    streamStatus?: string;
  }>(),
  {
    running: false,
    busy: false,
    readOnly: false,
    runCount: 0,
    blockedReason: "",
    variableCount: 0,
    undefinedCount: 0,
    resultPath: "",
    lastRunLabel: "",
    saving: false,
    dirty: false,
    streamStatus: "connected",
  }
);

const emit = defineEmits<{
  back: [];
  run: [];
  "run-force": [];
  validate: [];
  stop: [];
  variables: [];
  duplicate: [];
  delete: [];
  rename: [name: string];
}>();

const editingName = ref(false);
const draftName = ref("");
const nameInput = useTemplateRef<{ focus: () => void }>("nameInput");

async function startRename(): Promise<void> {
  draftName.value = props.name;
  editingName.value = true;
  await nextTick();
  nameInput.value?.focus();
}

function commitName(): void {
  if (!editingName.value) return;
  editingName.value = false;
  const next = draftName.value.trim();
  if (next && next !== props.name) emit("rename", next);
}

const STATUS_LABELS: Record<WorkflowStatus, string> = {
  idle: "Idle",
  running: "Running",
  cancelling: "Stopping",
  done: "Done",
  failed: "Failed",
  stopped: "Stopped",
};

const statusLabel = computed(() => STATUS_LABELS[props.status] ?? props.status);

const runDisabled = computed(() => props.busy || !!props.blockedReason);

const runTooltip = computed(() => {
  if (props.blockedReason) return props.blockedReason;
  if (!props.runCount) return "Every enabled step is done and up to date. Use Run all (force).";
  return "";
});

const saveLabel = computed(() => {
  if (props.readOnly) return "Read-only while running";
  if (props.saving) return "Saving…";
  return props.dirty ? "Unsaved changes" : "Saved";
});

function onCommand(command: string | number): void {
  if (command === "force") emit("run-force");
  else if (command === "validate") emit("validate");
  else if (command === "duplicate") emit("duplicate");
  else if (command === "delete") emit("delete");
}
</script>

<style scoped>
.wf-bar {
  border: 1px solid var(--el-border-color);
  border-radius: 8px;
  background: var(--el-bg-color-overlay);
  padding: 10px 12px;
  margin-bottom: 16px;
}

.wf-bar__top {
  display: flex;
  align-items: center;
  gap: 8px;
  flex-wrap: wrap;
}

.wf-bar__name {
  margin: 0;
  font-size: 16px;
  font-weight: 600;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  max-width: 40vw;
}

.wf-bar__name-input {
  max-width: 320px;
}

.wf-bar__spacer {
  flex: 1 1 auto;
}

.wf-bar__run {
  display: inline-flex;
}

.wf-bar__status {
  display: flex;
  align-items: center;
  gap: 6px;
  flex-wrap: wrap;
  margin-top: 8px;
  font-size: 12px;
  color: var(--el-text-color-secondary);
}

.wf-bar__dot--running,
.wf-bar__dot--cancelling {
  color: var(--el-color-primary);
}
.wf-bar__dot--done {
  color: var(--el-color-success);
}
.wf-bar__dot--failed {
  color: var(--el-color-danger);
}
.wf-bar__dot--stopped {
  color: var(--el-color-warning);
}

.wf-bar__result {
  min-width: 0;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.wf-bar__stream {
  color: var(--el-color-warning);
}
</style>
