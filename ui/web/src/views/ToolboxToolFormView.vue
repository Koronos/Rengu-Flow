<template>
  <div class="toolbox-form">
    <h2>{{ isEdit ? "Edit tool" : "New tool" }}</h2>
    <el-form label-position="top">
      <el-form-item label="Name">
        <el-input v-model="form.name" placeholder="e.g. Resize dataset images" />
      </el-form-item>
      <el-form-item label="Description">
        <el-input v-model="form.description" placeholder="What this tool does" />
      </el-form-item>
      <el-form-item label="Entrypoint function">
        <el-input v-model="form.entrypoint" placeholder="run" />
      </el-form-item>
      <el-form-item label="Required packages (one per line — uv resolves inline)">
        <el-input v-model="requirementsText" type="textarea" :rows="3" placeholder="e.g. pillow&#10;numpy>=2.0" />
      </el-form-item>
      <el-form-item label="Python script">
        <el-input v-model="form.script" type="textarea" :rows="14" placeholder="def run(num1, num2):&#10;    return num1 + num2" />
      </el-form-item>

      <h3>Inputs</h3>
      <p class="hint">Each input maps to a keyword argument of your entrypoint function.</p>
      <div v-for="(inp, i) in form.inputs" :key="i" class="input-row">
        <el-input v-model="inp.param" placeholder="param name" style="width: 140px" />
        <el-input v-model="inp.label" placeholder="label" style="width: 160px" />
        <el-select v-model="inp.control" style="width: 130px">
          <el-option v-for="c in controls" :key="c" :label="c" :value="c" />
        </el-select>
        <el-input
          v-if="inp.control === 'select'"
          v-model="optionsText[i]"
          placeholder="opt1, opt2"
          style="width: 180px"
          @input="syncOptions(i)"
        />
        <el-input v-model="inp.hint" placeholder="hint" style="flex: 1" />
        <el-button size="small" type="danger" @click="removeInput(i)">×</el-button>
      </div>
      <el-button size="small" @click="addInput">Add input</el-button>

      <div class="form-actions">
        <el-button type="primary" @click="save">Save</el-button>
        <el-button @click="$router.push('/toolbox')">Cancel</el-button>
      </div>
    </el-form>

    <ToolboxRunPanel v-if="isEdit && savedId" :tool-id="savedId" />
  </div>
</template>

<script setup lang="ts">
import { computed, onMounted, reactive, ref } from "vue";
import { useRoute, useRouter } from "vue-router";
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
