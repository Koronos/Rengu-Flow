<template>
  <div class="caption-editor">
    <div class="page-head caption-editor__head">
      <el-button :icon="ArrowLeft" @click="$router.push('/prep')">Dataset Studio</el-button>
      <h1 class="caption-editor__title">Caption editor</h1>
    </div>

    <el-card shadow="never">
      <template #header>Open a folder</template>
      <p class="page-hint">
        Review and fix captions one image at a time: every caption line is editable, and for an edit
        dataset the control images sit next to their target so a wrong instruction is easy to spot.
      </p>
      <form class="caption-editor__open-row" @submit.prevent="openFolder">
        <div class="caption-editor__field caption-editor__path">
          <label class="caption-editor__label" for="caption-editor-path">Dataset folder (targets)</label>
          <PathFieldControl
            v-bind="{ id: 'caption-editor-path' }"
            v-model="form.path"
            placeholder="e.g. /data/edit/targets"
            expect="dir"
            required
            @enter="openFolder"
          />
        </div>
        <div class="caption-editor__field caption-editor__path">
          <label class="caption-editor__label" for="caption-editor-controls">
            Control folder (edit datasets only)
          </label>
          <PathFieldControl
            v-bind="{ id: 'caption-editor-controls' }"
            v-model="form.control_path"
            placeholder="optional, e.g. /data/edit/controls"
            expect="dir"
            @enter="openFolder"
          />
        </div>
        <div class="caption-editor__field caption-editor__format">
          <label class="caption-editor__label" for="caption-editor-format">Caption format</label>
          <el-select id="caption-editor-format" v-model="form.format" class="w-full">
            <el-option label="Sidecar files" value="sidecar" />
            <el-option label="captions.json" value="json" />
            <el-option label="Auto (as the trainer)" value="auto" />
          </el-select>
        </div>
        <div v-if="form.format !== 'json'" class="caption-editor__field caption-editor__ext">
          <label class="caption-editor__label" for="caption-editor-ext">Extension</label>
          <el-input id="caption-editor-ext" v-model="form.ext" placeholder=".txt" />
        </div>
        <el-button type="primary" native-type="submit" :loading="loading && !page">Open</el-button>
      </form>
    </el-card>

    <el-alert
      v-if="readOnly && page"
      type="warning"
      :closable="false"
      show-icon
      :title="`Read-only: ${page.active.map((a) => a.label).join(', ')} is writing this folder.`"
      description="Captions can be reviewed but not saved until it finishes — a save now would race the stage."
    >
      <el-button size="small" class="mt-8" @click="reload">Check again</el-button>
    </el-alert>

    <el-alert v-if="error" type="error" :closable="false" show-icon :title="error">
      <el-button v-if="page" size="small" class="mt-8" @click="reload">Reload from disk (discard my edit)</el-button>
    </el-alert>

    <el-card v-if="!page" shadow="never">
      <el-empty description="Open a dataset folder above to review its captions." :image-size="64" />
    </el-card>

    <template v-else>
      <div class="caption-editor__toolbar">
        <label class="sr-only" for="caption-editor-search">Search captions and file names</label>
        <el-input
          id="caption-editor-search"
          v-model="search"
          clearable
          class="caption-editor__search"
          placeholder="Search captions and file names (Enter)"
          :prefix-icon="Search"
          @keydown.enter.prevent="applyQuery"
          @clear="applyQuery"
        />
        <el-radio-group v-model="filter" size="small" v-bind="ariaLabel('Show')" @change="applyQuery">
          <el-radio-button value="all">All ({{ page.image_count }})</el-radio-button>
          <el-radio-button value="uncaptioned">No caption ({{ page.uncaptioned_count }})</el-radio-button>
          <el-radio-button value="unpaired" :disabled="!page.control_path">
            Unpaired ({{ page.unpaired_count }})
          </el-radio-button>
        </el-radio-group>
        <el-text size="small" type="info">
          {{ page.format === "json" ? "captions.json" : `sidecar ${page.ext}` }}
          <template v-if="page.control_path"> · controls: {{ page.control_path }}</template>
        </el-text>
      </div>

      <div class="caption-editor__body">
        <el-card shadow="never" class="caption-editor__list-card">
          <template #header>
            <span>Images <el-tag size="small" effect="plain">{{ page.total }}</el-tag></span>
          </template>
          <el-empty v-if="!items.length" description="No images match" :image-size="56" />
          <ul v-else class="caption-editor__list" aria-label="Images">
            <li v-for="(item, i) in items" :key="item.key">
              <button
                :id="`caption-item-${i}`"
                type="button"
                class="caption-editor__item"
                :class="{ 'caption-editor__item--selected': i === selected }"
                :aria-current="i === selected ? 'true' : undefined"
                @click="select(i)"
              >
                <img
                  class="caption-editor__thumb"
                  :src="api.datasetPreviewImageUrl(item.token)"
                  :alt="`Thumbnail of ${item.key}`"
                  loading="lazy"
                />
                <span class="caption-editor__item-text">
                  <span class="caption-editor__item-name">
                    {{ item.key }}
                    <span v-if="i === selected && dirty" class="caption-editor__dot" title="Unsaved changes">
                      <span class="sr-only">(unsaved changes)</span>
                    </span>
                  </span>
                  <span class="caption-editor__item-caption">{{ item.lines[0] || "(no caption)" }}</span>
                  <span class="caption-editor__item-tags">
                    <el-tag v-if="item.unpaired" size="small" type="warning">unpaired</el-tag>
                    <el-tag v-if="item.lines.length > 1" size="small" type="info" effect="plain">
                      {{ item.lines.length }} lines
                    </el-tag>
                  </span>
                </span>
              </button>
            </li>
          </ul>
          <el-pagination
            v-if="page.total > page.limit"
            size="small"
            :current-page="Math.floor(page.offset / page.limit) + 1"
            :page-size="page.limit"
            :total="page.total"
            layout="prev, pager, next"
            class="caption-editor__pagination"
            @current-change="goToPage"
          />
        </el-card>

        <el-card v-if="current" shadow="never" class="caption-editor__detail">
          <template #header>
            <div class="caption-editor__detail-head">
              <span class="caption-editor__detail-name" :title="current.key">{{ current.key }}</span>
              <el-text size="small" type="info">{{ page.offset + selected + 1 }} of {{ page.total }}</el-text>
              <el-button-group>
                <el-button size="small" :icon="ArrowUp" v-bind="ariaLabel('Previous image (Alt+Up)')" @click="move(-1)" />
                <el-button size="small" :icon="ArrowDown" v-bind="ariaLabel('Next image (Alt+Down)')" @click="move(1)" />
              </el-button-group>
            </div>
          </template>

          <el-alert
            v-if="current.unpaired"
            type="warning"
            :closable="false"
            show-icon
            :title="`Not paired with its controls: ${current.unpaired}`"
            description="The trainer rejects an unpaired target; fix the control folder (stem.<ext> or stem_0.<ext>, stem_1.<ext>, …)."
            class="mb-8"
          />

          <div class="caption-editor__images">
            <figure v-for="(control, n) in current.controls" :key="control.name" class="caption-editor__figure">
              <img
                :src="api.datasetPreviewImageUrl(control.token)"
                :alt="`${controlLabel(n)} of ${current.key}: ${control.name}`"
                @click="openViewer(n)"
              />
              <figcaption>{{ controlLabel(n) }} · {{ control.name }}</figcaption>
            </figure>
            <figure class="caption-editor__figure caption-editor__figure--target">
              <img
                :src="api.datasetPreviewImageUrl(current.token)"
                :alt="`Target image ${current.key}`"
                @click="openViewer(current.controls.length)"
              />
              <figcaption>{{ current.controls.length ? "Target (result)" : "Image" }} · {{ current.key }}</figcaption>
            </figure>
          </div>

          <label class="caption-editor__label" for="caption-editor-text">
            Caption — one variant per line{{ page.control_path ? "; line 1 is the edit instruction" : "" }}
          </label>
          <el-input
            id="caption-editor-text"
            v-model="draft"
            type="textarea"
            :autosize="{ minRows: 3, maxRows: 14 }"
            :readonly="readOnly"
            placeholder="(no caption)"
          />
          <div class="caption-editor__actions">
            <el-button
              type="primary"
              :loading="saving"
              :disabled="!dirty || readOnly"
              @click="saveNow"
            >
              Save (Ctrl+Enter)
            </el-button>
            <el-button :disabled="!dirty" @click="resetDraft">Revert</el-button>
            <span class="caption-editor__status" role="status" aria-live="polite">
              <el-tag v-if="dirty" size="small" type="warning">Unsaved changes — saved when you move on</el-tag>
              <el-text v-else-if="saving" size="small" type="info">Saving…</el-text>
              <el-text v-else size="small" type="success">Saved</el-text>
            </span>
          </div>
          <el-text v-if="backupName" size="small" type="info" class="caption-editor__backup">
            Backup {{ backupName }} was taken before your first edit — restore it from Tag editor → Backups.
          </el-text>
          <el-text size="small" type="info" class="caption-editor__keys">
            Alt+↑/↓ (or ↑/↓ outside the text box): previous/next image · Ctrl+Enter: save. Blank lines are dropped, as the trainer reads them.
          </el-text>
        </el-card>
      </div>
    </template>
  </div>
</template>

<script setup lang="ts">
import { nextTick, onBeforeUnmount, onMounted, ref, watch } from "vue";
import { onBeforeRouteLeave, useRoute, useRouter } from "vue-router";
import { ElMessage, ElMessageBox } from "element-plus";
import { ArrowDown, ArrowLeft, ArrowUp, Search } from "@element-plus/icons-vue";
import { api } from "../api";
import { ariaLabel } from "../lib/aria";
import PathFieldControl from "../components/PathFieldControl.vue";
import { useCaptionEditor } from "../composables/useCaptionEditor";
import {
  captionEditorLocation,
  captionKeyAction,
  parseCaptionEditorQuery,
  type CaptionEditorTarget,
} from "../lib/captionEditor";
import { useDatasetImageViewerStore } from "../stores/datasetImageViewer";

const route = useRoute();
const router = useRouter();
const {
  page,
  items,
  selected,
  current,
  draft,
  dirty,
  readOnly,
  search,
  filter,
  loading,
  saving,
  error,
  backupName,
  open,
  select,
  move,
  save,
  flush,
  goToPage,
  applyQuery,
  reload,
  resetDraft,
} = useCaptionEditor();
const { openDatasetImageViewer } = useDatasetImageViewerStore();

const form = ref<CaptionEditorTarget>(parseCaptionEditorQuery(route.query));

async function openFolder(): Promise<void> {
  const path = form.value.path.trim();
  if (!path) return;
  const next: CaptionEditorTarget = {
    path,
    control_path: form.value.control_path.trim(),
    format: form.value.format,
    ext: form.value.ext.trim() || ".txt",
  };
  if (await open(next)) {
    void router.replace(captionEditorLocation(next)); // the URL reopens the same folder
  }
}

function controlLabel(n: number): string {
  const count = current.value?.controls.length ?? 0;
  return count > 1 ? `Image ${n + 1} (source)` : "Control (source)";
}

function openViewer(index: number): void {
  const item = current.value;
  if (!item) return;
  const urls = [...item.controls.map((c) => c.token), item.token].map(api.datasetPreviewImageUrl);
  openDatasetImageViewer(urls, index);
}

async function saveNow(): Promise<void> {
  if (await save()) ElMessage.success({ message: "Caption saved", duration: 1200 });
}

function isTextField(el: EventTarget | null): boolean {
  if (!(el instanceof HTMLElement)) return false;
  return el.tagName === "TEXTAREA" || el.tagName === "INPUT" || el.tagName === "SELECT" || el.isContentEditable;
}

function onKeydown(e: KeyboardEvent): void {
  // The lightbox and message boxes own the keyboard while they are open.
  if (!page.value || document.querySelector(".el-image-viewer__wrapper, .el-message-box")) return;
  const action = captionKeyAction({
    key: e.key,
    ctrlKey: e.ctrlKey,
    metaKey: e.metaKey,
    altKey: e.altKey,
    inTextField: isTextField(e.target),
  });
  if (!action) return;
  e.preventDefault();
  if (action === "save") void saveNow();
  else void move(action === "next" ? 1 : -1);
}

function onBeforeUnload(e: BeforeUnloadEvent): void {
  if (!dirty.value) return;
  e.preventDefault();
  e.returnValue = "";
}

// Keep the selected row visible while walking the list with the keyboard.
watch(selected, (i) => {
  void nextTick(() => document.getElementById(`caption-item-${i}`)?.scrollIntoView({ block: "nearest" }));
});

onBeforeRouteLeave(async () => {
  if (await flush()) return true;
  try {
    await ElMessageBox.confirm(
      "The last edit could not be saved. Leave and lose it?",
      "Unsaved caption",
      { type: "warning", confirmButtonText: "Leave", cancelButtonText: "Stay" }
    );
    return true;
  } catch {
    return false;
  }
});

onMounted(() => {
  window.addEventListener("keydown", onKeydown);
  window.addEventListener("beforeunload", onBeforeUnload);
  if (form.value.path) void openFolder();
});

onBeforeUnmount(() => {
  window.removeEventListener("keydown", onKeydown);
  window.removeEventListener("beforeunload", onBeforeUnload);
});

</script>

<style scoped>
.caption-editor {
  display: flex;
  flex-direction: column;
  gap: 12px;
}
.caption-editor__head {
  justify-content: flex-start;
  align-items: center;
  gap: var(--rf-space-sm);
  margin-bottom: 0;
}
.caption-editor__title {
  font-size: 16px;
  font-weight: 600;
  margin: 0;
}
.caption-editor__open-row {
  display: flex;
  gap: var(--rf-space-xs);
  align-items: flex-end;
  flex-wrap: wrap;
}
.caption-editor__field {
  display: flex;
  flex-direction: column;
  gap: 4px;
}
.caption-editor__label {
  font-size: 12px;
  color: var(--el-text-color-secondary);
}
.caption-editor__path {
  flex: 1;
  min-width: 240px;
}
.caption-editor__format {
  width: 180px;
}
.caption-editor__ext {
  width: 90px;
}
.w-full {
  width: 100%;
}
.mt-8 {
  margin-top: 8px;
}
.mb-8 {
  margin-bottom: 8px;
}
.sr-only {
  position: absolute;
  width: 1px;
  height: 1px;
  overflow: hidden;
  clip: rect(0 0 0 0);
  white-space: nowrap;
}
.caption-editor__toolbar {
  display: flex;
  gap: 10px;
  align-items: center;
  flex-wrap: wrap;
}
.caption-editor__search {
  max-width: 360px;
}
.caption-editor__body {
  display: grid;
  grid-template-columns: 340px 1fr;
  gap: 12px;
  align-items: start;
}
.caption-editor__list {
  list-style: none;
  margin: 0;
  padding: 0;
  display: flex;
  flex-direction: column;
  gap: 4px;
  max-height: calc(100vh - 330px);
  min-height: 240px;
  overflow: auto;
}
.caption-editor__item {
  display: flex;
  gap: 8px;
  width: 100%;
  padding: 4px;
  border: 1px solid transparent;
  border-radius: 6px;
  background: none;
  color: inherit;
  font: inherit;
  text-align: left;
  cursor: pointer;
}
.caption-editor__item:hover {
  background: var(--el-fill-color-light);
}
.caption-editor__item:focus-visible {
  outline: 2px solid var(--el-color-primary);
  outline-offset: 1px;
}
.caption-editor__item--selected {
  border-color: var(--el-color-primary);
  background: var(--el-color-primary-light-9);
}
.caption-editor__thumb {
  width: 56px;
  height: 56px;
  object-fit: cover;
  border-radius: 4px;
  flex-shrink: 0;
  background: var(--el-fill-color);
}
.caption-editor__item-text {
  display: flex;
  flex-direction: column;
  gap: 2px;
  min-width: 0;
}
.caption-editor__item-name {
  font-size: 12px;
  font-weight: 600;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
.caption-editor__item-caption {
  font-size: 12px;
  color: var(--el-text-color-secondary);
  display: -webkit-box;
  -webkit-line-clamp: 2;
  -webkit-box-orient: vertical;
  overflow: hidden;
}
.caption-editor__item-tags {
  display: flex;
  gap: 4px;
}
.caption-editor__dot {
  display: inline-block;
  width: 8px;
  height: 8px;
  margin-left: 4px;
  border-radius: 50%;
  background: var(--el-color-warning);
}
.caption-editor__pagination {
  margin-top: 8px;
  justify-content: center;
}
.caption-editor__detail {
  min-width: 0;
}
.caption-editor__detail-head {
  display: flex;
  align-items: center;
  gap: 10px;
}
.caption-editor__detail-name {
  font-weight: 600;
  flex: 1;
  min-width: 0;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
.caption-editor__images {
  display: flex;
  gap: 10px;
  flex-wrap: wrap;
  margin-bottom: 12px;
}
.caption-editor__figure {
  margin: 0;
  flex: 1 1 220px;
  min-width: 0;
  display: flex;
  flex-direction: column;
  gap: 4px;
}
.caption-editor__figure img {
  width: 100%;
  max-height: 44vh;
  object-fit: contain;
  border-radius: 6px;
  background: var(--el-fill-color-light);
  cursor: zoom-in;
}
.caption-editor__figure figcaption {
  font-size: 12px;
  color: var(--el-text-color-secondary);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
.caption-editor__figure--target figcaption {
  font-weight: 600;
  color: var(--el-text-color-primary);
}
.caption-editor__actions {
  margin-top: 10px;
  display: flex;
  gap: 8px;
  align-items: center;
  flex-wrap: wrap;
}
.caption-editor__backup,
.caption-editor__keys {
  display: block;
  margin-top: 8px;
}

@media (max-width: 1000px) {
  .caption-editor__body {
    grid-template-columns: 1fr;
  }
  .caption-editor__list {
    max-height: 40vh;
  }
}
</style>
