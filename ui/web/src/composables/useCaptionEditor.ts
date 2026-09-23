import { computed, ref } from "vue";
import { api } from "../api";
import { formatError } from "../lib/formatError";
import {
  isDraftDirty,
  linesFromText,
  stepTarget,
  textFromLines,
  type CaptionEditorTarget,
} from "../lib/captionEditor";
import type { CaptionListFilter, CaptionListResult } from "../types/api";

export const CAPTION_PAGE_SIZE = 60;

/**
 * Caption-editor state: one page of a folder's captions, the selected image and its draft.
 *
 * Every way off the current image — selecting another, next/previous (across pages), a new search
 * or filter, another folder — autosaves a dirty draft first, and stays put when that save fails
 * (a 409 conflict or a running prep job), so an edit is never silently dropped. The first save
 * after opening a folder asks the server for a caption backup (a tag-editor snapshot).
 */
export function useCaptionEditor(pageSize = CAPTION_PAGE_SIZE) {
  const target = ref<CaptionEditorTarget | null>(null);
  const page = ref<CaptionListResult | null>(null);
  const selected = ref(-1);
  const draft = ref("");
  const search = ref("");
  const filter = ref<CaptionListFilter>("all");
  const loading = ref(false);
  const saving = ref(false);
  const error = ref("");
  const backupName = ref<string | null>(null);

  const items = computed(() => page.value?.items ?? []);
  const current = computed(() => items.value[selected.value] ?? null);
  const readOnly = computed(() => page.value?.read_only ?? false);
  const dirty = computed(
    () => current.value !== null && isDraftDirty(current.value.lines, draft.value)
  );

  function resetDraft(): void {
    draft.value = current.value ? textFromLines(current.value.lines) : "";
  }

  async function fetchPage(offset: number, at: "first" | "last" | number = "first"): Promise<boolean> {
    if (!target.value) return false;
    loading.value = true;
    error.value = "";
    try {
      const t = target.value;
      page.value = await api.prepCaptions({
        path: t.path,
        control_path: t.control_path || undefined,
        format: t.format,
        ext: t.ext,
        q: search.value.trim() || undefined,
        filter: filter.value,
        limit: pageSize,
        offset,
      });
      const count = page.value.items.length;
      const index = at === "first" ? 0 : at === "last" ? count - 1 : at;
      selected.value = count ? Math.min(Math.max(index, 0), count - 1) : -1;
      resetDraft();
      return true;
    } catch (e) {
      error.value = formatError(e);
      return false;
    } finally {
      loading.value = false;
    }
  }

  /** Save the draft of the selected image. True when there was nothing to save or it saved. */
  async function save(): Promise<boolean> {
    const item = current.value;
    const t = target.value;
    if (!item || !t || !dirty.value) return true;
    if (readOnly.value) {
      error.value = "A prep run is writing this folder; it is read-only until it finishes.";
      return false;
    }
    saving.value = true;
    error.value = "";
    try {
      const result = await api.prepSaveCaption({
        path: page.value?.path ?? t.path,
        key: item.key,
        lines: linesFromText(draft.value),
        format: page.value?.format ?? t.format,
        ext: page.value?.ext ?? t.ext,
        expected: item.lines,
        backup: backupName.value === null,
      });
      if (result.backup) backupName.value = result.backup;
      item.lines = result.lines;
      if (current.value === item) resetDraft();
      return true;
    } catch (e) {
      error.value = formatError(e);
      return false;
    } finally {
      saving.value = false;
    }
  }

  /** Autosave before leaving the current image; false (and stay) if that save failed. */
  async function flush(): Promise<boolean> {
    return dirty.value ? save() : true;
  }

  async function open(next: CaptionEditorTarget): Promise<boolean> {
    if (!(await flush())) return false;
    target.value = { ...next };
    backupName.value = null;
    return fetchPage(0);
  }

  async function select(index: number): Promise<boolean> {
    if (index === selected.value || index < 0 || index >= items.value.length) return false;
    if (!(await flush())) return false;
    selected.value = index;
    resetDraft();
    return true;
  }

  async function move(delta: number): Promise<boolean> {
    if (!page.value) return false;
    const step = stepTarget(selected.value, delta, {
      offset: page.value.offset,
      limit: page.value.limit,
      total: page.value.total,
      count: items.value.length,
    });
    if (!step) return false;
    if (step.kind === "local") return select(step.index);
    if (!(await flush())) return false;
    return fetchPage(step.offset, step.at);
  }

  async function goToPage(pageNumber: number): Promise<boolean> {
    if (!(await flush())) return false;
    return fetchPage((pageNumber - 1) * pageSize);
  }

  /** Re-run the listing with the current search/filter, from the first page. */
  async function applyQuery(): Promise<boolean> {
    if (!(await flush())) return false;
    return fetchPage(0);
  }

  /** Reload the current page from disk, discarding the draft (after a conflict, or a job ends). */
  async function reload(): Promise<boolean> {
    return fetchPage(page.value?.offset ?? 0, selected.value);
  }

  return {
    target,
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
  };
}
