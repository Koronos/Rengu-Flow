<template>
  <div class="settings-view page-shell">
    <div class="page-head">
      <div class="page-head-text">
        <p class="page-subtitle">
          Edit <code>rengu.local.toml</code> ({{ form?.path }}). Training changes apply to the next
          run; maintenance applies immediately; server fields need a restart.
        </p>
      </div>
    </div>

    <el-alert v-if="error" type="error" show-icon class="mb-12" :title="error" :closable="false" />

    <el-skeleton v-if="loading" :rows="8" animated />

    <template v-else-if="form">
      <el-card shadow="never" class="mb-12">
        <template #header>Appearance</template>
        <div class="field">
          <label class="field-label">Color theme</label>
          <div class="field-name">browser preference (not stored in TOML)</div>
          <ThemeToggle />
          <p class="field-hint">Per-browser; saved locally, applied before paint to avoid a flash.</p>
        </div>
      </el-card>

      <el-card shadow="never" class="mb-12">
        <template #header>Training</template>
        <el-form label-position="top">
          <el-form-item>
            <template #label>GPUs <code class="toml-key">training.num_gpus</code></template>
            <el-input-number v-model="form.editable.training.num_gpus" :min="1" />
            <p class="field-hint">DeepSpeed <code>--num_gpus</code> for <code>rengu train</code>. CLI flags override this.</p>
          </el-form-item>
          <el-form-item>
            <template #label>Master port <code class="toml-key">training.master_port</code></template>
            <el-input-number v-model="form.editable.training.master_port" :min="1" :max="65535" />
            <p class="field-hint">Rendezvous port for the local DeepSpeed launcher. Default 29500.</p>
          </el-form-item>
          <el-form-item>
            <template #label>Extra args <code class="toml-key">training.extra_args</code></template>
            <el-input v-model="form.editable.training.extra_args" placeholder="e.g. --validate-only" />
            <p class="field-hint">Space-separated args appended after <code>--config</code>.</p>
          </el-form-item>
        </el-form>
      </el-card>

      <el-card shadow="never" class="mb-12">
        <template #header>Training environment</template>
        <div class="field">
          <label class="field-label">Subprocess env vars <code class="toml-key">training.env</code></label>
          <KeyValueListField
            v-model="form.editable.training.env"
            hint="Literal os.environ keys for the training subprocess (values are strings). Empty = inherit only the parent env."
          />
        </div>
      </el-card>

      <el-card shadow="never" class="mb-12">
        <template #header>Maintenance</template>
        <el-form label-position="top">
          <el-form-item>
            <template #label>Enable maintenance tools <code class="toml-key">maintenance.enabled</code></template>
            <el-switch v-model="form.editable.maintenance.enabled" />
            <p class="field-hint">Shows the Maintenance page (destructive DB reset, submodule update). Applies immediately.</p>
          </el-form-item>
          <el-form-item>
            <template #label>Allow pip in maintenance <code class="toml-key">maintenance.allow_pip</code></template>
            <el-switch v-model="form.editable.maintenance.allow_pip" />
            <p class="field-hint">Lets the deps-install action run pip. Off by default.</p>
          </el-form-item>
        </el-form>
      </el-card>

      <el-card shadow="never" class="mb-12">
        <template #header>Server</template>
        <el-form label-position="top">
          <el-form-item>
            <template #label>
              Expose on local network <code class="toml-key">ui.public</code>
              <el-tag size="small" type="warning" class="ml-6">restart to apply</el-tag>
            </template>
            <el-switch v-model="form.restartRequired.ui.public" />
            <p class="field-hint">Binds 0.0.0.0 so other devices can reach the UI. Set a token when on.</p>
          </el-form-item>
          <el-form-item>
            <template #label>
              API token <code class="toml-key">ui.token</code>
              <el-tag size="small" type="warning" class="ml-6">restart to apply</el-tag>
            </template>
            <el-input
              v-model="tokenField"
              type="password"
              show-password
              placeholder="empty = no token required"
            />
            <p class="field-hint">Required on every request (header X-Rengu-Flow-Token). Strongly recommended when public.</p>
          </el-form-item>
          <el-descriptions :column="1" size="small" border class="mt-12">
            <el-descriptions-item label="host (ui.host)">{{ form.readOnly.ui.host }}</el-descriptions-item>
            <el-descriptions-item label="port (ui.port)">{{ form.readOnly.ui.port }}</el-descriptions-item>
            <el-descriptions-item label="data_dir (ui.data_dir)">{{ form.readOnly.ui.data_dir }}</el-descriptions-item>
          </el-descriptions>
          <p class="field-hint">Read-only here — they only take effect at startup. Edit the file directly to change them.</p>
        </el-form>
      </el-card>

      <div class="actions">
        <el-button type="primary" :loading="saving" @click="onSave">Save changes</el-button>
        <el-text v-if="savedAt" type="success" size="small" class="ml-12">Saved.</el-text>
      </div>
    </template>
  </div>
</template>

<script setup lang="ts">
import { computed, onMounted, ref, toRaw } from "vue";
import { ElMessage } from "element-plus";
import { api } from "../api";
import type { LocalSettings, LocalSettingsPatch } from "../types/api";
import ThemeToggle from "../components/ThemeToggle.vue";
import KeyValueListField from "../components/KeyValueListField.vue";

const form = ref<LocalSettings | null>(null);
const loading = ref(true);
const saving = ref(false);
const error = ref("");
const savedAt = ref(false);

const tokenField = computed<string>({
  get: () => form.value?.restartRequired.ui.token ?? "",
  set: (v: string) => {
    if (form.value) form.value.restartRequired.ui.token = v ? v : null;
  },
});

async function load(): Promise<void> {
  loading.value = true;
  error.value = "";
  try {
    form.value = await api.getSettings();
  } catch (e) {
    error.value = e instanceof Error ? e.message : "Failed to load settings";
  } finally {
    loading.value = false;
  }
}

async function onSave(): Promise<void> {
  if (!form.value) return;
  saving.value = true;
  savedAt.value = false;
  error.value = "";
  // KeyValueListField emits "" when the last row is cleared; normalize to a
  // valid Record<string,string> so the backend does not reject with HTTP 422.
  // toRaw unwraps the Vue reactive proxy before iterating entries.
  const rawEnv = toRaw(form.value.editable.training.env) as unknown;
  const env: Record<string, string> = {};
  if (rawEnv && typeof rawEnv === "object") {
    for (const [k, v] of Object.entries(rawEnv as Record<string, unknown>)) env[k] = String(v);
  }
  const patch: LocalSettingsPatch = {
    training: { ...form.value.editable.training, env },
    maintenance: { ...form.value.editable.maintenance },
    ui: { ...form.value.restartRequired.ui },
  };
  try {
    form.value = await api.updateSettings(patch);
    savedAt.value = true;
    ElMessage.success("Settings saved");
  } catch (e) {
    error.value = e instanceof Error ? e.message : "Failed to save settings";
  } finally {
    saving.value = false;
  }
}

onMounted(load);

// Expose internals needed by unit tests (does not affect template or public API).
defineExpose({ form, onSave });
</script>

<style scoped>
.field { margin-bottom: 8px; }
.field-label { font-weight: 600; display: block; margin-bottom: 2px; }
.field-name, .toml-key { font-size: 12px; color: var(--el-text-color-secondary); }
.field-hint { margin: 4px 0 0; font-size: 12px; color: var(--el-text-color-secondary); }
.actions { margin-top: 16px; }
.ml-6 { margin-left: 6px; }
.ml-12 { margin-left: 12px; }
.mt-12 { margin-top: 12px; }
.mb-12 { margin-bottom: 12px; }
</style>
