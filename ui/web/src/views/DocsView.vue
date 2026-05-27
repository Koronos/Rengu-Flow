<template>
  <div class="docs-view page-shell">
    <div class="page-head">
      <div class="page-head-text">
        <p class="page-subtitle">User guides from the repository</p>
      </div>
    </div>

    <el-alert v-if="error" type="error" :title="error" show-icon class="mb-12" />

    <el-row :gutter="16">
      <el-col :xs="24" :md="7">
        <el-card shadow="never" class="docs-index-card">
          <template #header>Index</template>
          <el-skeleton v-if="loadingIndex" :rows="8" animated />
          <div v-else class="docs-index-list">
            <button
              v-for="item in indexItems"
              :key="item.path"
              type="button"
              class="docs-index-item"
              :class="{ 'is-active': item.path === activePath }"
              @click="onSelectDoc(item.path)"
            >
              {{ item.title }}
            </button>
          </div>
        </el-card>
      </el-col>
      <el-col :xs="24" :md="17">
        <el-card shadow="never" class="docs-body-card">
          <div v-loading="loadingDoc" class="doc-body">
            <el-empty v-if="!activePath && !loadingDoc" description="Select a guide from the index" />
            <article
              v-else-if="html"
              class="md-article"
              v-html="html"
              @click="onArticleClick"
            />
          </div>
        </el-card>
      </el-col>
    </el-row>
  </div>
</template>

<script setup>
import { onMounted, ref } from "vue";
import { useRoute, useRouter } from "vue-router";
import { api } from "../api";
import { renderMarkdown } from "../lib/markdown";

const route = useRoute();
const router = useRouter();

const loadingIndex = ref(true);
const loadingDoc = ref(false);
const error = ref("");
const indexItems = ref([]);
const activePath = ref("");
const html = ref("");

async function loadIndex() {
  loadingIndex.value = true;
  error.value = "";
  try {
    const data = await api.getDocsIndex();
    indexItems.value = data.items || [];
    const q = route.query.doc;
    if (typeof q === "string" && q) {
      await loadDoc(q);
    } else if (indexItems.value.length) {
      await loadDoc(indexItems.value[0].path);
    }
  } catch (e) {
    error.value = String(e);
  } finally {
    loadingIndex.value = false;
  }
}

async function loadDoc(path) {
  loadingDoc.value = true;
  error.value = "";
  html.value = "";
  try {
    const data = await api.getDoc(path);
    activePath.value = data.path || path;
    html.value = renderMarkdown(data.content || "", { docPath: activePath.value });
    router.replace({ name: "docs", query: { doc: activePath.value } });
  } catch (e) {
    error.value = String(e);
  } finally {
    loadingDoc.value = false;
  }
}

function onSelectDoc(path) {
  loadDoc(path);
}

function onArticleClick(event) {
  const link = event.target.closest("a.md-doc-link");
  if (!link) return;
  event.preventDefault();
  const next = link.getAttribute("data-doc-path");
  if (next && next !== activePath.value) {
    loadDoc(next);
  }
}

onMounted(() => {
  loadIndex();
});
</script>

<style scoped>
.docs-index-card {
  margin-bottom: var(--rf-space-sm);
}
.docs-index-card :deep(.el-card__body) {
  padding: 10px 12px;
}
@media (min-width: 768px) {
  .docs-index-card {
    margin-bottom: 0;
  }
}
.docs-index-list {
  display: flex;
  flex-direction: column;
  gap: 4px;
  max-height: calc(100vh - 250px);
  overflow-y: auto;
  overflow-x: hidden;
}
.docs-index-item {
  border: none;
  background: transparent;
  text-align: left;
  color: var(--el-text-color-regular);
  font-size: 14px;
  line-height: 1.35;
  padding: 8px 10px;
  border-radius: 6px;
  cursor: pointer;
  white-space: normal;
  word-break: break-word;
}
.docs-index-item:hover {
  background: var(--el-fill-color-light);
}
.docs-index-item.is-active {
  background: color-mix(in srgb, var(--el-color-primary) 18%, transparent);
  color: var(--el-color-primary-light-2);
  font-weight: 600;
}
.docs-body-card :deep(.el-card__body) {
  min-height: 320px;
}
.doc-body {
  min-height: 280px;
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
.md-article :deep(p),
.md-article :deep(ul),
.md-article :deep(ol) {
  margin: 0.55em 0;
  line-height: 1.6;
  color: var(--el-text-color-regular);
}
.md-article :deep(code) {
  font-family: var(--rf-font-mono, ui-monospace, monospace);
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
}
.md-article :deep(a) {
  color: var(--el-color-primary);
}
.md-article :deep(a.md-doc-link) {
  color: var(--el-color-warning);
  font-weight: 500;
}
</style>
