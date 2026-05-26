<template>
  <el-alert v-if="loadError" type="error" :title="loadError" show-icon :closable="false" />
  <el-skeleton v-else-if="!schema" :rows="6" animated />

  <div v-else>
    <el-tabs v-model="activeSection" tab-position="top" class="section-tabs">
      <el-tab-pane
        v-for="sec in schema.sections"
        :key="sec.id"
        :label="sec.title"
        :name="sec.id"
      >
        <template #label>
          <span>{{ sec.title }}</span>
          <el-badge
            v-if="attentionCount(sec) > 0"
            :value="attentionCount(sec)"
            type="warning"
            class="tab-badge"
          />
        </template>
      </el-tab-pane>
    </el-tabs>

    <el-card v-for="sec in visibleSections" :key="sec.id" shadow="never" class="section-card">
      <template #header>
        <div class="sec-header">
          <span>{{ sec.title }}</span>
          <el-text v-if="unfilledCount(sec).required" type="danger" size="small">
            {{ unfilledCount(sec).required }} required missing
          </el-text>
          <el-text v-if="unfilledCount(sec).recommended" type="warning" size="small">
            {{ unfilledCount(sec).recommended }} important missing
          </el-text>
        </div>
      </template>
      <p v-if="sec.description" class="sec-desc">{{ sec.description }}</p>

      <div v-if="adapterModeField(sec)" class="adapter-mode-row">
        <ConfigFormField
          :field="adapterModeField(sec)"
          :form="form"
          :capabilities="modelCapabilities"
          @update:path="onFieldUpdate"
        />
      </div>

      <el-alert
        v-if="sec.id === 'model' && selectedCapability"
        type="info"
        :closable="false"
        show-icon
        class="registry-alert"
      >
        <strong>{{ selectedCapability.display_name }}</strong>
        — supported training: {{ trainingModesText }}
        <template v-if="selectedCapability.branding_note">
          <br />
          <span class="branding-note">{{ selectedCapability.branding_note }}</span>
        </template>
      </el-alert>

      <el-alert
        v-if="sec.id === 'adapter' && selectedCapability"
        type="info"
        :closable="false"
        show-icon
        class="registry-alert"
      >
        <template v-if="modelSupportsAdapters(selectedCapability)">
          Adapter types: <strong>{{ selectedCapability.adapters.join(", ") }}</strong>
          <span v-if="selectedCapability.full_finetune"> — or disable adapter for full finetune.</span>
        </template>
        <template v-else-if="selectedCapability.full_finetune">
          Full-model finetune only (no LoRA / LoKr).
        </template>
      </el-alert>

      <el-form label-position="top" class="config-form">
        <div v-if="partition(sec).required.length" class="field-group">
          <div class="group-title">Required</div>
          <ConfigFormField
            v-for="field in partition(sec).required"
            :key="field.path"
            :field="field"
            :form="form"
            :capabilities="modelCapabilities"
            @update:path="onFieldUpdate"
          />
        </div>

        <div v-if="partition(sec).recommended.length" class="field-group">
          <div class="group-title group-title--important">
            Important
            <el-text type="info" size="small" class="group-hint">
              Has a default in the trainer — review before running
            </el-text>
          </div>
          <ConfigFormField
            v-for="field in partition(sec).recommended"
            :key="field.path"
            :field="field"
            :form="form"
            :capabilities="modelCapabilities"
            @update:path="onFieldUpdate"
          />
        </div>

        <div v-if="partition(sec).advanced.length" class="field-group">
          <template v-if="sec.flat_optional">
            <ConfigFormField
              v-for="field in partition(sec).advanced"
              :key="field.path"
              :field="field"
              :form="form"
              :capabilities="modelCapabilities"
              @update:path="onFieldUpdate"
            />
          </template>
          <el-collapse v-else v-model="advancedOpen[sec.id]" class="optional-collapse">
            <el-collapse-item :name="sec.id">
              <template #title>
                <span class="group-title optional-title">
                  Advanced / optional ({{ partition(sec).advanced.length }})
                </span>
              </template>
              <ConfigFormField
                v-for="field in partition(sec).advanced"
                :key="field.path"
                :field="field"
                :form="form"
                :capabilities="modelCapabilities"
                @update:path="onFieldUpdate"
              />
            </el-collapse-item>
          </el-collapse>
        </div>

        <el-empty
          v-if="
            !partition(sec).required.length &&
            !partition(sec).recommended.length &&
            !partition(sec).advanced.length
          "
          description="No fields for this section with the current model."
          :image-size="64"
        />
      </el-form>
    </el-card>
  </div>
</template>

<script setup>
import { computed, onMounted, reactive, ref, watch } from "vue";
import { api } from "../api";
import ConfigFormField from "./ConfigFormField.vue";
import {
  fieldIsFilled,
  fieldVisible,
  getModelCapability,
  modelSupportsAdapters,
  pruneFormForModel,
  trainingModesLabel,
} from "../lib/formUtils";

const props = defineProps({
  modelValue: { type: String, default: "" },
});

const emit = defineEmits(["update:modelValue"]);

const schema = ref(null);
const loadError = ref("");
const form = ref({ _has_adapter: true });
const activeSection = ref("general");
const advancedOpen = reactive({});
let syncing = false;

const modelCapabilities = computed(
  () => schema.value?.registries?.model_capabilities ?? {}
);

const selectedCapability = computed(() =>
  getModelCapability(modelCapabilities.value, form.value["model.type"])
);

const trainingModesText = computed(() => trainingModesLabel(selectedCapability.value));

const visibleSections = computed(() => {
  if (!schema.value) return [];
  return schema.value.sections.filter((s) => s.id === activeSection.value);
});

function fieldImportance(field) {
  if (field.importance) return field.importance;
  if (field.required) return "required";
  if (field.recommended) return "recommended";
  return "advanced";
}

const ADAPTER_MODE_PATH = "_has_adapter";

function isPinnedAdapterField(sec, field) {
  return sec.id === "adapter" && field.path === ADAPTER_MODE_PATH;
}

function adapterModeField(sec) {
  if (sec.id !== "adapter") return null;
  const caps = modelCapabilities.value;
  const field = (sec.fields || []).find((f) => f.path === ADAPTER_MODE_PATH);
  if (!field || !fieldVisible(field, form.value, caps)) return null;
  return field;
}

function partition(sec) {
  const caps = modelCapabilities.value;
  const visible = (sec.fields || []).filter(
    (f) => fieldVisible(f, form.value, caps) && !isPinnedAdapterField(sec, f)
  );
  const required = visible.filter((f) => fieldImportance(f) === "required");
  const recommended = visible.filter((f) => fieldImportance(f) === "recommended");
  const advanced = visible.filter((f) => fieldImportance(f) === "advanced");
  return { required, recommended, advanced };
}

function unfilledCount(sec) {
  const p = partition(sec);
  const values = form.value;
  return {
    required: p.required.filter((f) => !fieldIsFilled(f, values)).length,
    recommended: p.recommended.filter((f) => !fieldIsFilled(f, values)).length,
  };
}

function attentionCount(sec) {
  const u = unfilledCount(sec);
  return u.required + u.recommended;
}

function applyModelCapabilityDefaults() {
  const cap = selectedCapability.value;
  if (!cap) return;

  const next = { ...form.value };
  let changed = false;

  if (!modelSupportsAdapters(cap)) {
    if (next._has_adapter !== false) {
      next._has_adapter = false;
      changed = true;
    }
  } else if (!cap.full_finetune && !next._has_adapter) {
    next._has_adapter = true;
    changed = true;
  }

  if (next._has_adapter && cap.adapters?.length) {
    const allowed = cap.adapters;
    const current = next["adapter.type"];
    if (!current || !allowed.includes(current)) {
      next["adapter.type"] = allowed[0];
      changed = true;
    }
  }

  if (changed) {
    form.value = next;
    emitToml();
  }
}

async function loadSchema() {
  try {
    schema.value = await api.getSchema();
    for (const sec of schema.value.sections || []) {
      if (partition(sec).advanced.length) {
        advancedOpen[sec.id] = [];
      }
    }
  } catch (e) {
    loadError.value = String(e);
  }
}

async function syncFromToml(content) {
  if (!content.trim()) return;
  syncing = true;
  try {
    const r = await api.parseToml(content);
    if (r.ok) {
      form.value = pruneFormForModel({ ...r.form }, modelCapabilities.value);
      applyModelCapabilityDefaults();
    }
  } catch (e) {
    loadError.value = String(e);
  } finally {
    syncing = false;
  }
}

async function emitToml() {
  if (syncing) return;
  try {
    const r = await api.renderToml(form.value);
    if (r.ok) emit("update:modelValue", r.content);
  } catch (e) {
    loadError.value = String(e);
  }
}

function onFieldUpdate({ path, value }) {
  let next = { ...form.value, [path]: value };
  if (path === "model.type") {
    next = pruneFormForModel(next, modelCapabilities.value);
    form.value = next;
    applyModelCapabilityDefaults();
  } else {
    form.value = next;
  }
  emitToml();
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

defineExpose({
  reloadFromToml: () => syncFromToml(props.modelValue),
  flushToml: () => emitToml(),
});
</script>

<style scoped>
.section-tabs {
  margin-bottom: 12px;
}
.tab-badge {
  margin-left: 6px;
  vertical-align: middle;
}
.section-card {
  margin-bottom: 12px;
}
.sec-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
}
.sec-desc {
  margin: 0 0 12px;
  color: var(--el-text-color-secondary);
  font-size: 13px;
}
.adapter-mode-row {
  margin-bottom: 12px;
  padding: 10px 12px;
  border-radius: var(--el-border-radius-base);
  background: var(--el-fill-color-light);
  border: 1px solid var(--el-border-color);
}
.adapter-mode-row :deep(.el-form-item) {
  margin-bottom: 0;
}
.registry-alert {
  margin-bottom: 12px;
}
.branding-note {
  font-size: 12px;
  opacity: 0.9;
}
.config-form {
  max-width: 720px;
}
.field-group {
  margin-bottom: 8px;
}
.group-title {
  font-size: 13px;
  font-weight: 600;
  margin-bottom: 8px;
  color: var(--el-text-color-primary);
}
.group-title--important {
  color: var(--el-color-warning);
}
.group-hint {
  margin-left: 8px;
  font-weight: 400;
}
.optional-title {
  font-weight: 500;
  color: var(--el-text-color-secondary);
}
.optional-collapse {
  border: none;
}
.optional-collapse :deep(.el-collapse-item__header) {
  border: none;
  height: 36px;
  line-height: 36px;
}
.optional-collapse :deep(.el-collapse-item__wrap) {
  border: none;
}
</style>
