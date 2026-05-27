<template>
  <el-alert v-if="loadError" type="error" :title="loadError" show-icon />
  <el-skeleton v-else-if="!schema" :rows="4" animated />

  <div v-else class="dataset-form">
    <el-alert type="info" :closable="false" show-icon class="mb-12">
      Set <strong>global defaults</strong> first (resolutions, captions, shuffle). Each
      <strong>[[directory]]</strong> below can override those values for one folder only.
      Use <strong>Compose</strong> to merge library datasets into one file.
      <el-button
        v-for="doc in schema.doc_links || []"
        :key="doc.path"
        type="primary"
        link
        size="small"
        class="doc-link"
        @click="openDoc(doc.path)"
      >
        {{ doc.title }}
      </el-button>
    </el-alert>

    <el-tabs v-model="activeSection" class="section-tabs">
      <el-tab-pane
        v-for="sec in schema.sections.filter((s) => !s.is_directories)"
        :key="sec.id"
        :label="sec.title"
        :name="sec.id"
      />
    </el-tabs>

    <el-card
      v-for="sec in schema.sections.filter((s) => !s.is_directories && s.id === activeSection)"
      :key="sec.id"
      shadow="never"
      class="section-card"
    >
      <p v-if="sec.description" class="sec-desc">{{ sec.description }}</p>
      <el-form label-position="top">
        <ConfigFormField
          v-for="field in sec.fields"
          :key="field.path"
          :field="field"
          :form="form"
          @update:path="onFieldUpdate"
        />
      </el-form>
    </el-card>

    <el-card shadow="never" class="section-card section-card--directories">
      <template #header>Directories (per-folder overrides)</template>
      <p class="sec-desc">
        Select a folder to set its path and optional overrides. Values not set here use the
        global options above.
      </p>
      <DatasetDirectoryEditor
        v-model="form._directories"
        :global-form="form"
        :directory-fields="schema.directory_fields || []"
        :augmentation-fields="schema.augmentation_directory_fields || []"
        @update:model-value="onDirsChange"
        @select="onDirectorySelect"
      />
    </el-card>
  </div>

  <DocMarkdownDrawer v-model="docOpen" :doc-path="docPath" />
</template>

<script setup>
import { onMounted, ref, watch } from "vue";
import { api } from "../api";
import ConfigFormField from "./ConfigFormField.vue";
import DatasetDirectoryEditor from "./DatasetDirectoryEditor.vue";
import DocMarkdownDrawer from "./DocMarkdownDrawer.vue";

const props = defineProps({
  modelValue: { type: String, default: "" },
});

const emit = defineEmits(["update:modelValue", "preview", "directory-select"]);

const schema = ref(null);
const loadError = ref("");
const form = ref({ _directories: [] });
const activeSection = ref("resolutions");
const docOpen = ref(false);
const docPath = ref("");
let syncing = false;

function openDoc(path) {
  docPath.value = path;
  docOpen.value = true;
}

async function loadSchema() {
  schema.value = await api.getDatasetSchema();
}

async function syncFromToml(content) {
  if (!content.trim()) return;
  syncing = true;
  try {
    const r = await api.parseDatasetToml(content);
    if (r.ok) form.value = { ...r.form };
  } catch (e) {
    loadError.value = String(e);
  } finally {
    syncing = false;
  }
}

async function emitToml() {
  if (syncing) return;
  try {
    const r = await api.renderDatasetToml(form.value);
    if (r.ok) {
      emit("update:modelValue", r.content);
      refreshPreview(r.content);
    }
  } catch (e) {
    loadError.value = String(e);
  }
}

async function refreshPreview(content) {
  try {
    const r = await api.previewDataset(content);
    if (r.ok) emit("preview", r.preview);
  } catch {
    /* ignore */
  }
}

function onFieldUpdate({ path, value }) {
  form.value = { ...form.value, [path]: value };
  emitToml();
}

function onDirsChange() {
  emitToml();
}

function onDirectorySelect(index) {
  emit("directory-select", index);
}

onMounted(async () => {
  await loadSchema();
  await syncFromToml(props.modelValue);
});

watch(
  () => props.modelValue,
  (v) => {
    if (!syncing) syncFromToml(v);
  }
);

defineExpose({ reloadFromToml: () => syncFromToml(props.modelValue) });
</script>

<style scoped>
.mb-12 {
  margin-bottom: 12px;
}
.section-card {
  margin-bottom: 12px;
}
.sec-desc {
  margin: 0 0 12px;
  color: var(--el-text-color-secondary);
  font-size: 13px;
}
.doc-link {
  margin-left: 8px;
}
.section-tabs {
  margin-bottom: 8px;
}
.section-card--directories {
  margin-top: 4px;
}
</style>
