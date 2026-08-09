<template>
  <div class="workflows-page page-shell">
    <div class="page-head">
      <div class="page-head-text">
        <p class="page-subtitle">Chain folders, prep steps, tools and a training run into one pass</p>
      </div>
      <div class="page-head-actions">
        <el-button type="primary" :icon="Plus" :loading="creating" @click="createWorkflow">
          New workflow
        </el-button>
      </div>
    </div>

    <LibraryListPage
      :loading="loading"
      :error="error"
      :items="items"
      :view-mode="viewMode"
      table-subtitle-label="Chain"
      :table-actions-column-width="220"
      empty-description="No workflows yet"
      @item-click="openItem"
    >
      <template #empty-action>
        <el-button type="primary" :icon="Plus" :loading="creating" @click="createWorkflow">
          New workflow
        </el-button>
      </template>

      <template #toolbar>
        <el-input
          v-model="query"
          clearable
          placeholder="Search by name or ID…"
          class="page-toolbar-search"
          :prefix-icon="Search"
        />
        <LibrarySortControls
          v-model:sort-field="sortField"
          :sort-order="sortOrder"
          :field-options="workflowFieldOptions"
          :order-button-label="orderButtonLabel"
          @toggle-order="toggleSortOrder"
        />
        <LibraryViewModeToggle v-model="viewMode" />
      </template>

      <template #actions="{ item }">
        <LibraryItemOverflowMenu
          :loading="crudBusy"
          @duplicate="duplicateSelected(item.id ?? null)"
          @delete="deleteSelected(item.id ?? null)"
        >
          <el-dropdown-item @click.stop="openItem(item)">
            <span class="rf-dropdown-item-label">
              <el-icon><View /></el-icon><span>Open</span>
            </span>
          </el-dropdown-item>
          <el-dropdown-item @click.stop="renameItem(item)">
            <span class="rf-dropdown-item-label">
              <el-icon><EditPen /></el-icon><span>Rename</span>
            </span>
          </el-dropdown-item>
        </LibraryItemOverflowMenu>
      </template>
    </LibraryListPage>
  </div>
</template>

<script setup lang="ts">
import { computed, onMounted, ref } from "vue";
import { useRouter } from "vue-router";
import { ElMessage, ElMessageBox } from "element-plus";
import { EditPen, Plus, Search, View } from "@element-plus/icons-vue";
import { api } from "../api";
import LibraryItemOverflowMenu from "../components/LibraryItemOverflowMenu.vue";
import LibraryListPage from "../components/LibraryListPage.vue";
import LibrarySortControls from "../components/LibrarySortControls.vue";
import LibraryViewModeToggle from "../components/LibraryViewModeToggle.vue";
import { useDatasetViewMode } from "../composables/useDatasetViewMode";
import { useLibraryCrudActions } from "../composables/useLibraryCrudActions";
import { useLibraryListSort } from "../composables/useLibraryListSort";
import { formatError } from "../lib/formatError";
import { chainSummary, relativeTime } from "../lib/workflowCard";
import type { DatasetPreviewItem } from "../components/DatasetPreviewCollection.vue";
import type { WorkflowStatus, WorkflowSummary } from "../types/workflow";

/**
 * The workflow library.
 *
 * `GET /workflows` takes no query parameters and returns every row, so search and sort are done
 * here rather than on the server — which also means `load()` runs on mount and after a write, not
 * on every keystroke.
 *
 * `_workflow_summary` rides `chain` (every node's type, in order) along with `steps`, so the chain
 * summary reads straight off the row. There is no per-row detail hydration: that used to cost one
 * `GET /workflows/{id}` per row (capped, best-effort) just to learn the node types.
 */

const router = useRouter();
const { viewMode } = useDatasetViewMode("rengu-flow-workflow-list-view");
const { sortField, sortOrder, fieldOptions, toggleSortOrder, orderButtonLabel } =
  useLibraryListSort("rengu-flow-workflow-list-sort", {
    kind: "dataset",
    defaultField: "updated_at",
  });

// `created_at` is not in the list payload, so it is not offered as a sort key.
const workflowFieldOptions = computed(() =>
  fieldOptions.filter((option) => option.value !== "created_at")
);

const rows = ref<WorkflowSummary[]>([]);
const loading = ref(false);
const creating = ref(false);
const error = ref("");
const query = ref("");

const STATUS_LABELS: Record<WorkflowStatus, string> = {
  idle: "Idle",
  running: "Running",
  cancelling: "Stopping",
  done: "Done",
  failed: "Failed",
  stopped: "Stopped",
};

async function load(): Promise<void> {
  loading.value = true;
  error.value = "";
  try {
    const result = await api.listWorkflows();
    rows.value = result.workflows ?? [];
  } catch (e) {
    error.value = formatError(e);
  } finally {
    loading.value = false;
  }
}

function subtitleFor(row: WorkflowSummary): string {
  const parts = [`#${row.id}`, `${row.steps} ${row.steps === 1 ? "step" : "steps"}`];
  const chain = chainSummary(row.chain ?? []);
  if (chain && chain !== "No steps yet") parts.push(chain);
  parts.push(STATUS_LABELS[row.status] ?? row.status);
  const ago = relativeTime(row.updated_at);
  if (ago) parts.push(ago);
  return parts.join(" · ");
}

const filtered = computed(() => {
  const needle = query.value.trim().toLowerCase();
  if (!needle) return rows.value;
  return rows.value.filter(
    (row) =>
      String(row.id).toLowerCase().includes(needle) ||
      (row.name || "").toLowerCase().includes(needle)
  );
});

const sorted = computed(() => {
  const direction = sortOrder.value === "asc" ? 1 : -1;
  const field = sortField.value;
  return [...filtered.value].sort((a, b) => {
    let cmp = 0;
    if (field === "name") cmp = (a.name || "").localeCompare(b.name || "");
    else if (field === "updated_at") cmp = (a.updated_at || "").localeCompare(b.updated_at || "");
    else cmp = Number(a.id) - Number(b.id);
    // A stable tiebreak keeps rows from swapping places when two share a timestamp.
    return (cmp || Number(a.id) - Number(b.id)) * direction;
  });
});

const items = computed((): DatasetPreviewItem[] =>
  sorted.value.map((row) => ({
    key: String(row.id),
    id: row.id,
    title: row.name || `Workflow #${row.id}`,
    subtitle: subtitleFor(row),
    fallbackText: "WF",
    warning: row.status === "failed",
  }))
);

const {
  busy: crudBusy,
  duplicateSelected,
  deleteSelected,
} = useLibraryCrudActions({
  label: "workflow",
  duplicate: async (id) => ({ id: (await api.cloneWorkflow(id)).id }),
  remove: (id) => api.deleteWorkflow(id),
  onDeleted: () => load(),
  onDuplicated: () => load(),
});

function openItem(item: DatasetPreviewItem): void {
  if (item?.id == null) return;
  void router.push(`/workflows/${item.id}`);
}

/**
 * Rename through the graph, which is the only writable copy of the name: there is no rename route,
 * and `PUT /workflows/{id}` carries the whole graph, so the current one is read first.
 */
async function renameItem(item: DatasetPreviewItem): Promise<void> {
  if (item?.id == null) return;
  let next: string;
  try {
    const result = await ElMessageBox.prompt("New name", "Rename workflow", {
      inputValue: item.title ?? "",
      inputPattern: /\S/,
      inputErrorMessage: "A workflow needs a name",
    });
    next = String(result.value ?? "").trim();
  } catch {
    return;
  }
  try {
    const detail = await api.getWorkflow(item.id);
    await api.updateWorkflow(item.id, {
      graph: { ...detail.graph, name: next },
      version: detail.version,
    });
    ElMessage.success("Renamed");
    await load();
  } catch (e) {
    ElMessage.error(formatError(e));
  }
}

async function createWorkflow(): Promise<void> {
  if (creating.value) return;
  creating.value = true;
  try {
    const detail = await api.createWorkflow("New workflow");
    void router.push(`/workflows/${detail.id}`);
  } catch (e) {
    ElMessage.error(formatError(e));
  } finally {
    creating.value = false;
  }
}

onMounted(() => void load());
</script>
