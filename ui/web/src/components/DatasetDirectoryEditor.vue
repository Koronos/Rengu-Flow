<template>
  <div class="dir-editor">
    <el-row :gutter="12">
      <el-col :xs="24" :sm="8" class="dir-list-col">
        <div class="dir-list-header">
          <span class="dir-list-title">Folders</span>
          <el-button size="small" :icon="Plus" @click="addDirectory">Add</el-button>
        </div>
        <el-scrollbar max-height="420px" class="dir-list-scroll">
          <div
            v-for="(row, index) in directories"
            :key="index"
            class="dir-list-item"
            :class="{ active: index === selectedIndex }"
            @click="selectDirectory(index)"
          >
            <div class="dir-list-item-main">
              <span class="dir-list-label">{{ directoryLabel(row, index) }}</span>
              <el-tag v-if="overrideCount(row) > 0" size="small" type="warning">
                {{ overrideCount(row) }} override{{ overrideCount(row) === 1 ? "" : "s" }}
              </el-tag>
              <el-tag v-if="rowAugEnabled(row)" size="small" type="success">aug</el-tag>
              <el-tag v-if="row.num_repeats > 1" size="small" type="info">×{{ row.num_repeats }}</el-tag>
            </div>
            <el-text v-if="row.path" type="info" size="small" class="dir-list-path">{{ row.path }}</el-text>
            <el-text v-else type="warning" size="small">Path not set</el-text>
          </div>
          <el-empty v-if="!directories.length" description="No directories" :image-size="48" />
        </el-scrollbar>
      </el-col>

      <el-col :xs="24" :sm="16">
        <el-card v-if="selectedRow" shadow="never" class="dir-detail-card">
          <template #header>
            <div class="dir-detail-header">
              <span>Directory {{ selectedIndex + 1 }}</span>
              <el-button
                type="danger"
                size="small"
                :icon="Delete"
                :disabled="directories.length <= 1"
                @click="removeDirectory"
              >
                Remove
              </el-button>
            </div>
          </template>

          <el-alert type="info" :closable="false" show-icon class="mb-12">
            Only fields you override here differ from the <strong>global defaults above</strong>.
            Use <strong>Remove override</strong> to drop a per-folder value and inherit global again.
          </el-alert>

          <el-form label-position="top" class="dir-fields">
            <template v-for="field in visibleFields" :key="field.path">
              <div v-if="field.show_if_set && !hasOwnOverride(field.path)" class="override-row">
                <el-button size="small" @click="enableOverride(field.path)">
                  + Override {{ field.label }}
                </el-button>
              </div>
              <div v-else class="dir-field-block">
                <ConfigFormField
                  :field="fieldWithInheritHint(field)"
                  :form="directoryForm"
                  @update:path="onDirectoryField"
                />
                <el-button
                  v-if="canRemoveOverride(field.path) && hasOwnOverride(field.path)"
                  type="primary"
                  link
                  size="small"
                  class="clear-override"
                  @click="clearOverride(field.path)"
                >
                  Remove override (use global)
                </el-button>
              </div>
            </template>
          </el-form>

          <el-collapse v-if="augmentationFields.length" class="aug-collapse">
            <el-collapse-item title="Augmentation (this folder)" name="augmentation">
              <el-form label-position="top" class="dir-fields">
                <ConfigFormField
                  v-for="field in augmentationFields"
                  :key="'aug-' + field.path"
                  :field="field"
                  :form="augmentationForm"
                  @update:path="onAugmentationField"
                />
              </el-form>
            </el-collapse-item>
          </el-collapse>

          <div class="dir-actions">
            <el-button size="small" :loading="scanning" @click="scanSelected">Scan folder</el-button>
          </div>
          <el-alert
            v-if="scanResult"
            :type="scanResult.ok ? 'success' : 'warning'"
            :closable="true"
            show-icon
            class="mt-12"
            @close="scanResult = null"
          >
            <template v-if="scanResult.ok">
              {{ scanResult.image_count }} images
              <span v-if="scanResult.video_count">, {{ scanResult.video_count }} videos</span>
              <span v-if="scanResult.has_captions_json"> · captions.json</span>
              <span v-else-if="scanResult.caption_txt_files">
                · {{ scanResult.caption_txt_files }} .txt
              </span>
            </template>
            <template v-else>{{ scanResult.error }}</template>
          </el-alert>
        </el-card>
        <el-empty v-else description="Select a directory" :image-size="64" />
      </el-col>
    </el-row>
  </div>
</template>

<script setup>
import { computed, ref, watch } from "vue";
import { Delete, Plus } from "@element-plus/icons-vue";
import { api } from "../api";
import { fieldEffectiveValue } from "../lib/formUtils";
import ConfigFormField from "./ConfigFormField.vue";

const props = defineProps({
  modelValue: { type: Array, default: () => [] },
  globalForm: { type: Object, default: () => ({}) },
  directoryFields: { type: Array, default: () => [] },
  augmentationFields: { type: Array, default: () => [] },
});

const emit = defineEmits(["update:modelValue", "select"]);

const directories = ref([]);
const selectedIndex = ref(0);
const scanResult = ref(null);
const scanning = ref(false);

const REQUIRED_PATHS = new Set(["path", "num_repeats"]);
const BASIC_PATHS = new Set(["path", "num_repeats", "directory_caption"]);

const AR_OVERRIDE_PATHS = ["min_ar", "max_ar", "num_ar_buckets", "ar_buckets"];
const CAPTION_PATHS = new Set([
  "shuffle_tags",
  "cache_shuffle_num",
  "cache_shuffle_delimiter",
  "shuffle_metadata",
  "online_captions",
]);

function syncFromProps() {
  const rows = (props.modelValue || []).map((r) => ({ ...r }));
  directories.value = rows.length ? rows : [emptyRow()];
  if (selectedIndex.value >= directories.value.length) {
    selectedIndex.value = Math.max(0, directories.value.length - 1);
  }
}

function emptyRow() {
  return { path: "", num_repeats: 1, directory_caption: "" };
}

watch(() => props.modelValue, syncFromProps, { immediate: true, deep: true });

const selectedRow = computed(() => directories.value[selectedIndex.value] ?? null);

const visibleFields = computed(() => {
  const row = selectedRow.value || {};
  return (props.directoryFields || []).filter((field) => {
    if (BASIC_PATHS.has(field.path) || CAPTION_PATHS.has(field.path)) {
      return true;
    }
    if (field.show_when_field) {
      const en = fieldEffectiveValue(
        { path: field.show_when_field, default: false, type: "boolean" },
        directoryForm.value
      );
      if (!en && !hasOwnOverride(field.path)) {
        return false;
      }
    }
    if (field.show_if_set && !hasOwnOverride(field.path)) {
      return true;
    }
    if (field.show_if_set) {
      return hasOwnOverride(field.path);
    }
    return true;
  });
});

/** Effective values for controls: directory override, else global default. */
const directoryForm = computed(() => {
  const row = selectedRow.value || {};
  const g = props.globalForm || {};
  const merged = { ...g };
  for (const field of props.directoryFields || []) {
    const key = field.path;
    if (key in row) {
      merged[key] = row[key];
    }
  }
  return merged;
});

const augmentationForm = computed(() => {
  const row = selectedRow.value || {};
  const aug = row.augmentation;
  if (aug && typeof aug === "object" && !Array.isArray(aug)) {
    return { ...aug };
  }
  return {};
});

function rowAugEnabled(row) {
  const aug = row?.augmentation;
  return aug && typeof aug === "object" && !!aug.enabled;
}

function onAugmentationField({ path, value }) {
  const row = directories.value[selectedIndex.value];
  if (!row) return;
  if (!row.augmentation || typeof row.augmentation !== "object") {
    row.augmentation = {};
  }
  if (value === "" || value === undefined || value === null) {
    if (path === "enabled") {
      row.augmentation.enabled = false;
    } else {
      delete row.augmentation[path];
    }
  } else {
    row.augmentation[path] = value;
  }
  if (
    Object.keys(row.augmentation).length === 0 ||
    (Object.keys(row.augmentation).length === 1 &&
      row.augmentation.enabled === false &&
      !row.augmentation.preset)
  ) {
    delete row.augmentation;
  }
  emitDirectories();
}

function hasOwnOverride(path) {
  const row = selectedRow.value;
  return row != null && Object.prototype.hasOwnProperty.call(row, path);
}

function canRemoveOverride(path) {
  return !REQUIRED_PATHS.has(path);
}

function overrideCount(row) {
  if (!row) return 0;
  return Object.keys(row).filter(
    (k) =>
      k !== "augmentation" &&
      canRemoveOverride(k) &&
      Object.prototype.hasOwnProperty.call(row, k)
  ).length;
}

function clearOverride(path) {
  const row = directories.value[selectedIndex.value];
  if (!row || !canRemoveOverride(path)) return;
  delete row[path];
  if (path === "enable_ar_bucket") {
    for (const p of AR_OVERRIDE_PATHS) {
      delete row[p];
    }
  }
  emitDirectories();
}

function globalValue(path) {
  const g = props.globalForm || {};
  if (path in g) return g[path];
  const field = (props.directoryFields || []).find((f) => f.path === path);
  return field?.default;
}

function fieldWithInheritHint(field) {
  if (BASIC_PATHS.has(field.path) || !hasOwnOverride(field.path)) {
    if (BASIC_PATHS.has(field.path) || field.path === "directory_caption") {
      return field;
    }
    const gv = globalValue(field.path);
    if (gv === undefined || gv === null || gv === "") {
      return field;
    }
    const hint =
      typeof gv === "boolean"
        ? gv
          ? "true"
          : "false"
        : Array.isArray(gv)
          ? gv.join(", ")
          : String(gv);
    return {
      ...field,
      description: `${field.description || field.label} (global: ${hint})`,
    };
  }
  return {
    ...field,
    description: `${field.description || ""} Overriding global default.`.trim(),
  };
}

function directoryLabel(row, index) {
  if (!row.path?.trim()) return `Directory ${index + 1}`;
  const parts = row.path.replace(/\\/g, "/").split("/").filter(Boolean);
  return parts[parts.length - 1] || row.path;
}

function emitDirectories() {
  emit(
    "update:modelValue",
    directories.value.map((r) => ({ ...r }))
  );
  emit("select", selectedIndex.value);
}

function selectDirectory(index) {
  selectedIndex.value = index;
  scanResult.value = null;
  emit("select", index);
}

function addDirectory() {
  directories.value.push(emptyRow());
  selectedIndex.value = directories.value.length - 1;
  emitDirectories();
}

function removeDirectory() {
  if (directories.value.length <= 1) return;
  directories.value.splice(selectedIndex.value, 1);
  if (selectedIndex.value >= directories.value.length) {
    selectedIndex.value = directories.value.length - 1;
  }
  emitDirectories();
}

function onDirectoryField({ path, value }) {
  const row = directories.value[selectedIndex.value];
  if (!row) return;
  if (value === "" || value === undefined || value === null) {
    if (!BASIC_PATHS.has(path)) {
      delete row[path];
    } else {
      row[path] = value;
    }
  } else {
    row[path] = value;
  }
  emitDirectories();
}

function enableOverride(path) {
  const row = directories.value[selectedIndex.value];
  if (!row) return;
  const field = (props.directoryFields || []).find((f) => f.path === path);
  const gv = globalValue(path);
  if (field?.type === "boolean") {
    row[path] = gv ?? false;
  } else if (field?.type === "integer_list" || field?.type === "number_list") {
    row[path] = Array.isArray(gv) ? [...gv] : gv ?? [];
  } else if (gv !== undefined) {
    row[path] = gv;
  } else if ("default" in (field || {})) {
    row[path] = field.default;
  } else {
    row[path] = "";
  }
  emitDirectories();
}

async function scanSelected() {
  const path = selectedRow.value?.path?.trim();
  if (!path) return;
  scanning.value = true;
  try {
    scanResult.value = await api.scanDatasetPath(path);
  } finally {
    scanning.value = false;
  }
}

watch(selectedIndex, (i) => emit("select", i));
</script>

<style scoped>
.dir-list-col {
  border-right: 1px solid var(--el-border-color-lighter);
  padding-right: 8px;
}
.dir-list-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 8px;
}
.dir-list-title {
  font-weight: 600;
  font-size: 13px;
}
.dir-list-item {
  padding: 10px 12px;
  margin-bottom: 6px;
  border-radius: var(--el-border-radius-base);
  border: 1px solid var(--el-border-color-lighter);
  cursor: pointer;
  transition: background 0.15s, border-color 0.15s;
}
.dir-list-item:hover {
  background: var(--el-fill-color-light);
}
.dir-list-item.active {
  border-color: var(--el-color-primary);
  background: var(--el-color-primary-light-9);
}
.dir-list-item-main {
  display: flex;
  align-items: center;
  gap: 8px;
}
.dir-list-label {
  font-weight: 600;
  font-size: 13px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.dir-list-path {
  display: block;
  margin-top: 4px;
  font-family: ui-monospace, monospace;
  font-size: 11px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.dir-detail-card {
  min-height: 280px;
}
.dir-detail-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
}
.mb-12 {
  margin-bottom: 12px;
}
.mt-12 {
  margin-top: 12px;
}
.override-row {
  margin-bottom: 12px;
}
.aug-collapse {
  margin-top: 16px;
}
.dir-actions {
  margin-top: 8px;
}
.dir-field-block {
  margin-bottom: 4px;
}
.clear-override {
  margin: -4px 0 12px;
  padding-left: 0;
}
</style>
