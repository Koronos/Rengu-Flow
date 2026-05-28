<template>
  <el-drawer
    :model-value="modelValue"
    :title="title || 'Documentation'"
    size="min(720px, 92vw)"
    direction="rtl"
    destroy-on-close
    @update:model-value="$emit('update:modelValue', $event)"
  >
    <div v-loading="loading" class="doc-drawer-body">
      <el-alert v-if="error" type="error" :title="error" show-icon :closable="false" />
      <article
        v-else-if="html"
        class="md-article"
        v-html="html"
        @click="onArticleClick"
      />
    </div>
  </el-drawer>
</template>

<script setup lang="ts">
import { ref, watch } from "vue";
import { ElLoadingDirective } from "element-plus";
import { api } from "../api";
import { formatError } from "../lib/formatError";
import { renderMarkdown } from "../lib/markdown";

const props = defineProps({
  modelValue: { type: Boolean, default: false },
  docPath: { type: String, default: "" },
});

defineEmits(["update:modelValue"]);
const vLoading = ElLoadingDirective;

const loading = ref(false);
const error = ref("");
const html = ref("");
const title = ref("");
const activePath = ref("");

async function load(path: string) {
  if (!path) return;
  loading.value = true;
  error.value = "";
  html.value = "";
  try {
    const data = (await api.getDoc(path)) as { title?: string; path?: string; content?: string };
    title.value = data.title || path;
    activePath.value = data.path || path;
    html.value = renderMarkdown(data.content || "", {
      docPath: activePath.value,
    });
  } catch (e) {
    error.value = formatError(e);
  } finally {
    loading.value = false;
  }
}

function onArticleClick(event: MouseEvent) {
  const target = event.target;
  if (!(target instanceof Element)) return;
  const link = target.closest("a.md-doc-link");
  if (!link) return;
  event.preventDefault();
  const next = link.getAttribute("data-doc-path");
  if (next && next !== activePath.value) {
    load(next);
  }
}

watch(
  () => (props.modelValue ? props.docPath : ""),
  (path) => {
    if (path) {
      activePath.value = path;
      load(path);
    }
  }
);
</script>

<style scoped>
.doc-drawer-body {
  min-height: 200px;
  padding-right: 8px;
}

.md-article :deep(h1),
.md-article :deep(h2),
.md-article :deep(h3),
.md-article :deep(h4) {
  margin: 1.1em 0 0.45em;
  line-height: 1.3;
  color: var(--el-text-color-primary);
}

.md-article :deep(h1) {
  font-size: 1.35rem;
}
.md-article :deep(h2) {
  font-size: 1.15rem;
}
.md-article :deep(h3) {
  font-size: 1.05rem;
}

.md-article :deep(p),
.md-article :deep(ul),
.md-article :deep(ol) {
  margin: 0.55em 0;
  line-height: 1.6;
  color: var(--el-text-color-regular);
}

.md-article :deep(ul),
.md-article :deep(ol) {
  padding-left: 1.4em;
}

.md-article :deep(li) {
  margin: 0.25em 0;
}

.md-article :deep(code) {
  font-family: ui-monospace, monospace;
  font-size: 0.88em;
  background: var(--el-fill-color-light);
  padding: 0.12em 0.35em;
  border-radius: 4px;
}

.md-article :deep(pre) {
  background: var(--el-fill-color-darker);
  padding: 12px 14px;
  border-radius: 8px;
  overflow-x: auto;
  font-size: 12px;
  line-height: 1.45;
  margin: 0.75em 0;
}

.md-article :deep(pre code) {
  background: transparent;
  padding: 0;
}

.md-article :deep(blockquote) {
  margin: 0.75em 0;
  padding: 0.35em 0 0.35em 12px;
  border-left: 3px solid var(--el-color-primary);
  color: var(--el-text-color-secondary);
}

.md-article :deep(hr) {
  border: none;
  border-top: 1px solid var(--el-border-color);
  margin: 1em 0;
}

.md-article :deep(a) {
  color: var(--el-color-primary);
  text-decoration: none;
}
.md-article :deep(a:hover) {
  text-decoration: underline;
}
.md-article :deep(a.md-doc-link) {
  color: var(--el-color-warning);
  font-weight: 500;
}

.md-article :deep(.md-table-wrap) {
  overflow-x: auto;
  margin: 0.75em 0;
}

.md-article :deep(table) {
  width: 100%;
  border-collapse: collapse;
  font-size: 13px;
}

.md-article :deep(th),
.md-article :deep(td) {
  border: 1px solid var(--el-border-color);
  padding: 8px 10px;
  text-align: left;
  vertical-align: top;
}

.md-article :deep(th) {
  background: var(--el-fill-color);
  font-weight: 600;
  color: var(--el-text-color-primary);
}

.md-article :deep(tr:nth-child(even) td) {
  background: var(--el-fill-color-lighter);
}

.md-article :deep(strong) {
  color: var(--el-text-color-primary);
}
</style>
