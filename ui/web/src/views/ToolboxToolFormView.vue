<template>
  <div class="toolbox-form">
    <header class="form-head">
      <div class="form-head__title">
        <h2>{{ isEdit ? "Edit tool" : "New tool" }}</h2>
        <span class="form-head__sub">{{ form.name || "Untitled tool" }}</span>
      </div>
      <div class="form-head__actions">
        <el-button @click="$router.push('/toolbox')">Cancel</el-button>
        <el-button type="primary" @click="save">Save</el-button>
      </div>
    </header>

    <div class="editor-grid">
      <!-- Authoring -->
      <section class="authoring">
        <el-form label-position="top">
          <div class="meta-row">
            <el-form-item label="Name" class="meta-name">
              <el-input v-model="form.name" placeholder="e.g. Resize dataset images" />
            </el-form-item>
            <el-form-item label="Entrypoint function" class="meta-entry">
              <el-input v-model="form.entrypoint" placeholder="run" />
            </el-form-item>
          </div>
          <el-form-item label="Description">
            <el-input v-model="form.description" placeholder="What this tool does" />
          </el-form-item>
          <el-form-item label="Required packages">
            <el-input
              v-model="requirementsText"
              type="textarea"
              :rows="2"
              placeholder="one per line — uv resolves them inline, isolated from this app&#10;e.g. pillow&#10;numpy>=2.0"
            />
          </el-form-item>
        </el-form>

        <!-- Script: the heart of the tool -->
        <div class="code-block">
          <div class="code-block__bar">
            <span class="code-block__title">Script</span>
            <span class="code-block__hint">
              define <code>{{ (form.entrypoint || "run").trim() }}(…)</code> — its parameters are the inputs below
            </span>
          </div>
          <el-input
            v-model="form.script"
            type="textarea"
            :autosize="{ minRows: 12, maxRows: 30 }"
            class="code-area"
            placeholder="def run(num1, num2):&#10;    print(num1 + num2)"
          />
        </div>

        <!-- Inputs builder -->
        <div class="inputs-section">
          <div class="inputs-section__head">
            <h3>Inputs</h3>
            <el-button size="small" :icon="Plus" @click="addInput">Add input</el-button>
          </div>
          <p class="hint">Each input becomes a keyword argument of your entrypoint function.</p>

          <el-empty
            v-if="!form.inputs?.length"
            description="No inputs — the function runs with no arguments"
            :image-size="56"
          />
          <div v-for="(inp, i) in form.inputs" :key="i" class="input-card">
            <div class="input-card__main">
              <el-input v-model="inp.param" placeholder="param" class="f-param" />
              <el-input v-model="inp.label" placeholder="label" class="f-label" />
              <el-select v-model="inp.control" class="f-type">
                <el-option v-for="c in controls" :key="c" :label="c" :value="c" />
              </el-select>
              <el-tooltip content="Remove input" placement="top">
                <el-button text :icon="Delete" class="input-card__del" @click="removeInput(i)" />
              </el-tooltip>
            </div>
            <div class="input-card__extra">
              <el-input
                v-if="inp.control === 'select'"
                v-model="optionsText[i]"
                placeholder="options, comma separated"
                class="f-options"
                @input="syncOptions(i)"
              />
              <el-input v-model="inp.hint" placeholder="hint (optional)" class="f-hint" />
            </div>
          </div>
        </div>
      </section>

      <!-- Run: stays beside the code -->
      <aside class="run-col">
        <ToolboxRunPanel v-if="isEdit && savedId" :key="savedId" :tool-id="savedId" />
        <div v-else class="run-placeholder">
          <h3>Run</h3>
          <p class="hint">Save the tool to run it. Its inputs and live output appear here.</p>
        </div>
      </aside>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed, onMounted, reactive, ref } from "vue";
import { useRoute, useRouter } from "vue-router";
import { Delete, Plus } from "@element-plus/icons-vue";
import { api, type ToolboxInput, type ToolboxToolWrite } from "../api";
import ToolboxRunPanel from "../components/ToolboxRunPanel.vue";

const route = useRoute();
const router = useRouter();
const controls = ["number", "text", "textarea", "switch", "select"] as const;

const isEdit = computed(() => Boolean(route.params.id));
const savedId = ref<string | null>((route.params.id as string) || null);

const form = reactive<ToolboxToolWrite>({
  name: "",
  description: "",
  entrypoint: "run",
  requirements: [],
  script: "",
  inputs: [],
});
const requirementsText = ref("");
const optionsText = reactive<Record<number, string>>({});

function addInput() {
  form.inputs!.push({ param: "", label: "", control: "text", hint: "" } as ToolboxInput);
}
function removeInput(i: number) {
  form.inputs!.splice(i, 1);
  // Rebuild optionsText: drop key i, shift keys > i down by one.
  const next: Record<number, string> = {};
  for (const k in optionsText) {
    const idx = Number(k);
    if (idx < i) next[idx] = optionsText[idx];
    else if (idx > i) next[idx - 1] = optionsText[idx];
    // idx === i is dropped
  }
  Object.keys(optionsText).forEach((k) => delete optionsText[Number(k)]);
  Object.assign(optionsText, next);
}
function syncOptions(i: number) {
  form.inputs![i].options = (optionsText[i] || "").split(",").map((s) => s.trim()).filter(Boolean);
}

async function save() {
  form.requirements = requirementsText.value.split("\n").map((s) => s.trim()).filter(Boolean);
  if (isEdit.value && savedId.value) {
    await api.updateToolboxTool(savedId.value, form);
  } else {
    const created = await api.createToolboxTool(form);
    savedId.value = created.id;
    router.replace(`/toolbox/${created.id}/edit`);
  }
}

onMounted(async () => {
  if (isEdit.value && savedId.value) {
    const t = await api.getToolboxTool(savedId.value);
    form.name = t.name;
    form.description = t.description;
    form.entrypoint = t.entrypoint;
    form.script = t.script;
    form.inputs = t.inputs;
    requirementsText.value = t.requirements.join("\n");
    t.inputs.forEach((inp, i) => {
      if (inp.options) optionsText[i] = inp.options.join(", ");
    });
  }
});
</script>

<style scoped>
.form-head {
  display: flex;
  align-items: flex-end;
  justify-content: space-between;
  gap: var(--rf-space-md);
  flex-wrap: wrap;
  margin-bottom: var(--rf-space-md);
}
.form-head__title {
  display: flex;
  align-items: baseline;
  gap: var(--rf-space-sm);
}
.form-head__title h2 {
  margin: 0;
}
.form-head__sub {
  color: var(--el-text-color-secondary);
  font-size: 13px;
}
.form-head__actions {
  display: flex;
  gap: var(--rf-space-xs);
}

.editor-grid {
  display: grid;
  grid-template-columns: minmax(0, 1fr) minmax(360px, 460px);
  gap: var(--rf-space-md);
  align-items: start;
}
@media (max-width: 980px) {
  .editor-grid {
    grid-template-columns: 1fr;
  }
}

.meta-row {
  display: flex;
  gap: var(--rf-space-sm);
}
.meta-name {
  flex: 1 1 auto;
}
.meta-entry {
  flex: 0 0 200px;
}
@media (max-width: 560px) {
  .meta-row {
    flex-direction: column;
  }
  .meta-entry {
    flex-basis: auto;
  }
}

/* Code surface */
.code-block {
  border: 1px solid var(--el-border-color);
  border-radius: var(--el-border-radius-base);
  overflow: hidden;
  margin-bottom: var(--rf-space-md);
}
.code-block__bar {
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  gap: var(--rf-space-sm);
  padding: 6px 10px;
  background: var(--el-fill-color-light);
  border-bottom: 1px solid var(--el-border-color-lighter);
  flex-wrap: wrap;
}
.code-block__title {
  font-weight: 600;
  font-size: 12px;
  letter-spacing: 0.04em;
  text-transform: uppercase;
  color: var(--el-text-color-regular);
}
.code-block__hint {
  font-size: 12px;
  color: var(--el-text-color-secondary);
}
.code-block__hint code {
  font-family: var(--rf-font-mono);
}
.code-area :deep(textarea) {
  font-family: var(--rf-font-mono);
  font-size: 13px;
  line-height: 1.6;
  border: none;
  border-radius: 0;
  box-shadow: none;
  tab-size: 4;
  resize: vertical;
}

/* Inputs builder */
.inputs-section__head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: var(--rf-space-sm);
}
.inputs-section__head h3 {
  margin: var(--rf-space-sm) 0;
}
.input-card {
  border: 1px solid var(--el-border-color-lighter);
  border-radius: var(--el-border-radius-base);
  padding: var(--rf-space-xs);
  margin-bottom: var(--rf-space-xs);
  background: var(--el-bg-color);
}
.input-card__main {
  display: flex;
  gap: var(--rf-space-xs);
  align-items: center;
}
.f-param {
  flex: 0 0 130px;
}
.f-label {
  flex: 1 1 auto;
}
.f-type {
  flex: 0 0 120px;
}
.input-card__del {
  flex: 0 0 auto;
  color: var(--el-text-color-secondary);
}
.input-card__del:hover {
  color: var(--el-color-danger);
}
.input-card__extra {
  display: flex;
  gap: var(--rf-space-xs);
  margin-top: var(--rf-space-xs);
}
.f-options {
  flex: 0 0 240px;
}
.f-hint {
  flex: 1 1 auto;
}
.hint {
  color: var(--el-text-color-secondary);
  font-size: 12px;
  margin: 4px 0 var(--rf-space-xs);
}

/* Run column */
.run-col {
  position: sticky;
  top: calc(var(--rf-topbar-height, 56px) + var(--rf-space-md));
}
.run-placeholder {
  border: 1px dashed var(--el-border-color);
  border-radius: var(--el-border-radius-base);
  padding: var(--rf-space-md);
}
.run-placeholder h3 {
  margin-top: 0;
}
</style>
