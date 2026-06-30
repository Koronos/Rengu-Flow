<template>
  <div class="tag-editor">
    <div class="page-head tag-editor__head">
      <el-button :icon="ArrowLeft" @click="$router.push('/prep')">Dataset Studio</el-button>
      <span class="tag-editor__title">Tag editor</span>
    </div>

    <el-card shadow="never" class="tag-editor__open">
      <template #header>Open a folder</template>
      <p class="page-hint">
        Browse tag frequencies, bulk add / remove / rename tags, and quarantine images — changes are
        staged and reviewed before they touch your caption files.
      </p>
      <div class="tag-editor__open-row">
        <div class="tag-editor__field tag-editor__path">
          <label class="tag-editor__label">Dataset folder</label>
          <PathFieldControl
            v-model="path"
            placeholder="e.g. /path/to/dataset/images"
            expect="dir"
            required
            @enter="openSession"
          />
        </div>
        <div class="tag-editor__field tag-editor__format">
          <label class="tag-editor__label">Caption format</label>
          <el-select v-model="format" size="default" class="w-full">
            <el-option label="Sidecar files" value="sidecar" />
            <el-option label="captions.json" value="json" />
          </el-select>
        </div>
        <div v-if="format === 'sidecar'" class="tag-editor__field tag-editor__ext">
          <label class="tag-editor__label">Extension</label>
          <el-input v-model="ext" class="w-full" placeholder=".txt" />
        </div>
        <el-button type="primary" :loading="loading" @click="openSession">Open</el-button>
        <el-button :disabled="!path" @click="openBackups">Backups</el-button>
      </div>
      <div v-if="session" class="tag-editor__session-info">
        <el-tag size="small" effect="plain">{{ session.image_count }} images</el-tag>
        <el-tag size="small" effect="plain">{{ session.format }}{{ session.format === "sidecar" ? ` (${session.ext})` : "" }}</el-tag>
        <el-tag
          v-if="session.changed_count"
          size="small"
          type="warning"
        >
          {{ session.changed_count }} pending change(s)
        </el-tag>
      </div>
      <el-alert v-if="error" type="error" :title="error" :closable="false" show-icon class="mt-8" />
    </el-card>

    <el-card v-if="!session" shadow="never" class="tag-editor__placeholder">
      <el-empty description="Open a dataset folder above to start editing its tags." :image-size="64" />
    </el-card>

    <div v-if="session" class="tag-editor__body">
      <el-card shadow="never" class="tag-editor__tags">
        <template #header>Tag frequencies</template>
        <TagFrequencyTable
          :tags="stats?.tags ?? []"
          :scope="statsScope"
          @scope-change="onStatsScope"
          @select-tag="addTagToFilter"
          @remove-tag="removeTagEverywhere"
          @rename-tag="renameTag"
          @prune="pruneTags"
        />
      </el-card>

      <div class="tag-editor__main">
        <el-card shadow="never">
          <template #header>
            <div class="tag-editor__filter-header">
              <span>Find images by tags</span>
              <el-select v-model="opScope" size="small" class="tag-editor__scope">
                <el-option label="Apply to: line 1" value="line1" />
                <el-option label="Apply to: tag lines" value="tag_lines" />
                <el-option label="Apply to: all lines" value="all_lines" />
              </el-select>
            </div>
          </template>
          <TagFilterBuilder v-model="filter" :tag-options="tagOptions" />
          <div class="tag-editor__actions">
            <el-button size="small" type="primary" :loading="loading" @click="runQuery">
              Find matching
            </el-button>
            <el-divider direction="vertical" />
            <el-select
              v-model="editTags"
              multiple
              filterable
              allow-create
              clearable
              default-first-option
              size="small"
              class="tag-editor__edit-tags"
              placeholder="tags to add / remove"
            >
              <el-option v-for="t in tagOptions" :key="t" :label="t" :value="t" />
            </el-select>
            <el-button size="small" :disabled="!editTags.length" @click="stageAdd('start')">
              Add at start
            </el-button>
            <el-button size="small" :disabled="!editTags.length" @click="stageAdd('end')">
              Add at end
            </el-button>
            <el-button size="small" type="danger" plain :disabled="!editTags.length" @click="stageRemove">
              Remove
            </el-button>
            <el-button
              size="small"
              type="danger"
              :disabled="filterEmpty"
              @click="stageQuarantine"
            >
              Quarantine matching
            </el-button>
          </div>
          <div class="tag-editor__size-row">
            <span class="tag-editor__size-label">Size filter (px):</span>
            <el-input-number
              v-model="sizeBelow"
              :min="0"
              :step="64"
              size="small"
              controls-position="right"
            />
            <el-button size="small" :loading="loading" @click="findBySize">
              Find images with short side &lt; {{ sizeBelow }}
            </el-button>
            <el-button
              size="small"
              type="danger"
              plain
              :disabled="!sizeResultKeys.length"
              @click="quarantineFound"
            >
              Quarantine found ({{ sizeResultKeys.length }})
            </el-button>
            <el-text size="small" type="info">
              Finds images whose short side is below the threshold. Use it to quarantine thumbnails and web-scrape artifacts before tagging or captioning produces garbled results on them.
            </el-text>
          </div>
        </el-card>

        <el-card shadow="never" class="tag-editor__results">
          <template #header>
            <span>
              Matching images
              <el-tag v-if="query" size="small" effect="plain">{{ query.total }}</el-tag>
            </span>
          </template>
          <el-empty
            v-if="!query"
            description="Build a filter and press 'Find matching'"
            :image-size="60"
          />
          <el-empty
            v-else-if="!query.keys.length"
            description="No images match the filter"
            :image-size="60"
          />
          <div v-else class="tag-editor__grid">
            <div
              v-for="(key, i) in query.keys"
              :key="key"
              class="tag-editor__cell"
            >
              <PreviewImage
                :src="api.datasetPreviewImageUrl(query.previews[key])"
                class="tag-editor__thumb"
                @click="openViewer(i)"
              />
              <div class="tag-editor__cell-name" :title="key">
                {{ key }}
                <span v-if="query.sizes?.[key]" class="tag-editor__cell-size">
                  {{ query.sizes[key][0] }}×{{ query.sizes[key][1] }}
                </span>
              </div>
              <div
                class="tag-editor__cell-caption"
                :title="(query.captions[key] ?? []).join('\n')"
              >
                {{ (query.captions[key] ?? [])[0] || "(no caption)" }}
              </div>
            </div>
          </div>
          <el-pagination
            v-if="query && query.total > query.limit"
            :current-page="Math.floor(query.offset / query.limit) + 1"
            :page-size="query.limit"
            :total="query.total"
            layout="prev, pager, next, total"
            class="tag-editor__pagination"
            @current-change="goToQueryPage"
          />
        </el-card>

        <el-card shadow="never">
          <TagOpsStagedList :ops="stagedOps" @undo="undo" />
          <div class="tag-editor__commit">
            <el-button
              type="primary"
              :disabled="!hasStaged"
              :loading="loading"
              @click="openDiff"
            >
              Review &amp; commit…
            </el-button>
          </div>
        </el-card>
      </div>
    </div>

    <TagDiffDialog
      v-model="diffOpen"
      :diff="diff"
      :loading="diffLoading"
      :committing="committing"
      @commit="doCommit"
    />

    <el-dialog v-model="backupsOpen" title="Caption backups &amp; quarantine" width="640px">
      <h4 class="tag-editor__dlg-h">Backups</h4>
      <el-empty v-if="!backups.length" description="No backups yet" :image-size="48" />
      <el-table v-else :data="backups" size="small">
        <el-table-column prop="name" label="Backup" min-width="170" />
        <el-table-column prop="file_count" label="Files" width="70" align="right" />
        <el-table-column width="110" align="right">
          <template #default="{ row }">
            <el-popconfirm
              title="Restore captions from this backup? Current caption files are replaced."
              @confirm="restoreBackup(row.name)"
            >
              <template #reference>
                <el-button size="small" type="warning" plain>Restore</el-button>
              </template>
            </el-popconfirm>
          </template>
        </el-table-column>
      </el-table>
      <h4 class="tag-editor__dlg-h">Quarantined images</h4>
      <el-empty v-if="!quarantine.length" description="Quarantine is empty" :image-size="48" />
      <el-table v-else :data="quarantine" size="small">
        <el-table-column prop="name" label="Batch" min-width="150" />
        <el-table-column label="Images" min-width="180">
          <template #default="{ row }">{{ row.images.join(", ") }}</template>
        </el-table-column>
        <el-table-column width="110" align="right">
          <template #default="{ row }">
            <el-button size="small" plain @click="restoreQuarantine(row.name)">
              Restore
            </el-button>
          </template>
        </el-table-column>
      </el-table>
    </el-dialog>
  </div>
</template>

<script setup lang="ts">
import { computed, ref } from "vue";
import { ElMessage } from "element-plus";
import { ArrowLeft } from "@element-plus/icons-vue";
import { api } from "../api";
import { useTagSession } from "../composables/useTagSession";
import { useDatasetImageViewerStore } from "../stores/datasetImageViewer";
import PathFieldControl from "../components/PathFieldControl.vue";
import PreviewImage from "../components/PreviewImage.vue";
import TagFilterBuilder from "../components/TagFilterBuilder.vue";
import TagFrequencyTable from "../components/TagFrequencyTable.vue";
import TagOpsStagedList from "../components/TagOpsStagedList.vue";
import TagDiffDialog from "../components/TagDiffDialog.vue";
import type { QuarantineBatchInfo, TagBackupInfo, TagDiffResult, TagEditOpDto } from "../types/api";

const {
  session,
  sessionId,
  stats,
  statsScope,
  query,
  loading,
  error,
  stagedOps,
  hasStaged,
  open,
  runQuery: runSessionQuery,
  runSizeQuery,
  stageOps,
  undo,
  setStatsScope,
  commit,
  queryOffset,
  QUERY_PAGE_SIZE,
  goToQueryPage,
} = useTagSession();

const path = ref("");
const format = ref<"sidecar" | "json">("sidecar");
const ext = ref(".txt");
const filter = ref<TagEditOpDto["filter"]>({ all: [], any: [], none: [] });
const editTags = ref<string[]>([]);
const opScope = ref<"line1" | "tag_lines" | "all_lines">("tag_lines");
const sizeBelow = ref(512);

const diffOpen = ref(false);
const diff = ref<TagDiffResult | null>(null);
const diffLoading = ref(false);
const committing = ref(false);

const backupsOpen = ref(false);
const backups = ref<TagBackupInfo[]>([]);
const quarantine = ref<QuarantineBatchInfo[]>([]);

const { openDatasetImageViewer } = useDatasetImageViewerStore();

const tagOptions = computed(() => (stats.value?.tags ?? []).map((t) => t.tag));
const filterEmpty = computed(
  () =>
    !(filter.value?.all?.length || filter.value?.any?.length || filter.value?.none?.length)
);

async function openSession(): Promise<void> {
  if (!path.value.trim()) return;
  const ok = await open(path.value.trim(), format.value, ext.value.trim() || ".txt");
  if (ok) ElMessage.success(`Opened ${session.value?.image_count} images`);
}

function onStatsScope(scope: string): void {
  void setStatsScope(scope as typeof statsScope.value);
}

function addTagToFilter(tag: string): void {
  const all = [...(filter.value?.all ?? [])];
  if (!all.includes(tag)) all.push(tag);
  filter.value = { ...filter.value, all };
}

async function runQuery(): Promise<void> {
  await runSessionQuery(filter.value, "tag_lines");
}

const sizeResultKeys = computed(() =>
  query.value?.sizes ? query.value.keys : []
);

async function findBySize(): Promise<void> {
  if (!sizeBelow.value) return;
  await runSizeQuery({ below: sizeBelow.value });
}

function quarantineFound(): void {
  void stage(
    { op: "quarantine", keys: [...sizeResultKeys.value] },
    `Staged: quarantine ${sizeResultKeys.value.length} small image(s)`
  );
}

function currentFilter(): TagEditOpDto["filter"] | undefined {
  return filterEmpty.value ? undefined : filter.value;
}

async function stage(op: TagEditOpDto, message: string): Promise<void> {
  if (await stageOps([op])) ElMessage.success(message);
}

function stageAdd(position: "start" | "end"): void {
  void stage(
    {
      op: "add",
      tags: [...editTags.value],
      filter: currentFilter(),
      scope: opScope.value,
      position,
    },
    "Staged: add tags"
  );
}

function stageRemove(): void {
  void stage(
    { op: "remove", tags: [...editTags.value], filter: currentFilter(), scope: opScope.value },
    "Staged: remove tags"
  );
}

function stageQuarantine(): void {
  void stage(
    { op: "quarantine", filter: filter.value, scope: opScope.value },
    "Staged: quarantine matching images"
  );
}

function removeTagEverywhere(tag: string): void {
  void stage({ op: "remove", tags: [tag], scope: opScope.value }, `Staged: remove '${tag}'`);
}

function renameTag(from: string, to: string): void {
  void stage(
    { op: "rename", tags: [from], rename_to: to, scope: opScope.value },
    `Staged: rename '${from}' → '${to}'`
  );
}

function pruneTags(minCount: number): void {
  void stage(
    { op: "prune", min_count: minCount, scope: opScope.value },
    `Staged: prune tags seen < ${minCount} times`
  );
}

function openViewer(index: number): void {
  if (!query.value) return;
  const urls = query.value.keys.map((key) =>
    api.datasetPreviewImageUrl(query.value!.previews[key])
  );
  openDatasetImageViewer(urls, index);
}

async function openDiff(): Promise<void> {
  diffOpen.value = true;
  diffLoading.value = true;
  try {
    diff.value = await api.prepTagDiff(sessionId.value, 200);
  } catch (e) {
    ElMessage.error(e instanceof Error ? e.message : String(e));
    diffOpen.value = false;
  } finally {
    diffLoading.value = false;
  }
}

async function doCommit(): Promise<void> {
  committing.value = true;
  try {
    const result = await commit();
    if (result) {
      diffOpen.value = false;
      ElMessage.success(
        `Committed ${result.files_written.length} file(s) — backup ${result.backup}`
      );
    }
  } finally {
    committing.value = false;
  }
}

async function openBackups(): Promise<void> {
  const target = (session.value?.path ?? path.value).trim();
  if (!target) return;
  try {
    const [b, q] = await Promise.all([
      api.prepTagBackups(target),
      api.prepQuarantineBatches(target),
    ]);
    backups.value = b.backups;
    quarantine.value = q.batches;
    backupsOpen.value = true;
  } catch (e) {
    ElMessage.error(e instanceof Error ? e.message : String(e));
  }
}

async function restoreBackup(name: string): Promise<void> {
  const target = (session.value?.path ?? path.value).trim();
  await api.prepRestoreTagBackup(target, name);
  ElMessage.success("Backup restored");
  backupsOpen.value = false;
  if (session.value) await openSession();
}

async function restoreQuarantine(name: string): Promise<void> {
  const target = (session.value?.path ?? path.value).trim();
  await api.prepRestoreQuarantine(target, name);
  ElMessage.success("Quarantine batch restored");
  backupsOpen.value = false;
  if (session.value) await openSession();
}
</script>

<style scoped>
.tag-editor {
  display: flex;
  flex-direction: column;
  gap: 12px;
}
.tag-editor__head {
  justify-content: flex-start;
  align-items: center;
  gap: var(--rf-space-sm);
  margin-bottom: 0;
}
.tag-editor__title {
  font-size: 16px;
  font-weight: 600;
}
.tag-editor__open-row {
  display: flex;
  gap: var(--rf-space-xs);
  align-items: flex-end;
  flex-wrap: wrap;
}
.tag-editor__field {
  display: flex;
  flex-direction: column;
  gap: 4px;
}
.tag-editor__label {
  font-size: 12px;
  color: var(--el-text-color-secondary);
}
.tag-editor__path {
  flex: 1;
  min-width: 240px;
}
.tag-editor__format {
  width: 150px;
}
.tag-editor__ext {
  width: 90px;
}
.tag-editor__placeholder :deep(.el-card__body) {
  padding: var(--rf-space-md);
}
.w-full {
  width: 100%;
}
.mt-8 {
  margin-top: 8px;
}
.tag-editor__session-info {
  margin-top: 8px;
  display: flex;
  gap: 6px;
}
.tag-editor__body {
  display: grid;
  grid-template-columns: 320px 1fr;
  gap: 12px;
  align-items: start;
}
.tag-editor__tags :deep(.el-card__body) {
  height: calc(100vh - 280px);
  min-height: 320px;
}
.tag-editor__main {
  display: flex;
  flex-direction: column;
  gap: 12px;
  min-width: 0;
}
.tag-editor__filter-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
}
.tag-editor__scope {
  width: 170px;
}
.tag-editor__actions {
  margin-top: 12px;
  display: flex;
  gap: 8px;
  align-items: center;
  flex-wrap: wrap;
}
.tag-editor__edit-tags {
  min-width: 220px;
  flex: 1;
}
.tag-editor__size-row {
  margin-top: 10px;
  display: flex;
  gap: 8px;
  align-items: center;
  flex-wrap: wrap;
}
.tag-editor__size-label {
  font-size: 12px;
  color: var(--el-text-color-secondary);
}
.tag-editor__cell-size {
  font-weight: 400;
  color: var(--el-text-color-secondary);
}
.tag-editor__grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(140px, 1fr));
  gap: 10px;
  max-height: 48vh;
  overflow: auto;
}
.tag-editor__cell {
  display: flex;
  flex-direction: column;
  gap: 4px;
  min-width: 0;
}
.tag-editor__thumb {
  aspect-ratio: 1;
  border-radius: 6px;
  overflow: hidden;
  cursor: zoom-in;
}
.tag-editor__cell-name {
  font-size: 11px;
  font-weight: 600;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
.tag-editor__cell-caption {
  font-size: 11px;
  color: var(--el-text-color-secondary);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
.tag-editor__commit {
  margin-top: 10px;
  display: flex;
  justify-content: flex-end;
}
.tag-editor__dlg-h {
  margin: 8px 0;
}

@media (max-width: 1100px) {
  .tag-editor__body {
    grid-template-columns: 1fr;
  }
  .tag-editor__tags :deep(.el-card__body) {
    height: 360px;
  }
}
</style>
