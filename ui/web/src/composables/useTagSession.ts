import { computed, ref } from "vue";
import { api } from "../api";
import type {
  TagEditOpDto,
  TagQueryResult,
  TagSessionSummary,
  TagStatsResult,
} from "../types/api";

/**
 * Tag-editor session state: open a folder, stage ops (server keeps everything in
 * memory until commit), refresh stats/query views after each mutation.
 */
export function useTagSession() {
  const session = ref<TagSessionSummary | null>(null);
  const stats = ref<TagStatsResult | null>(null);
  const statsScope = ref<"line1" | "tag_lines" | "all_lines">("line1");
  const query = ref<TagQueryResult | null>(null);
  const lastFilter = ref<TagEditOpDto["filter"] | null>(null);
  const lastFilterScope = ref("tag_lines");
  const loading = ref(false);
  const error = ref("");

  const sessionId = computed(() => session.value?.session_id ?? "");
  const stagedOps = computed(() => session.value?.staged_ops ?? []);
  const hasStaged = computed(() => stagedOps.value.length > 0);

  async function withLoading<T>(fn: () => Promise<T>): Promise<T | null> {
    loading.value = true;
    error.value = "";
    try {
      return await fn();
    } catch (e) {
      error.value = e instanceof Error ? e.message : String(e);
      return null;
    } finally {
      loading.value = false;
    }
  }

  async function refreshStats(): Promise<void> {
    if (!sessionId.value) return;
    stats.value = await api.prepTagStats(sessionId.value, statsScope.value);
  }

  async function refreshQuery(): Promise<void> {
    if (!sessionId.value || !lastFilter.value) return;
    query.value = await api.prepTagQuery(
      sessionId.value,
      lastFilter.value,
      lastFilterScope.value
    );
  }

  async function open(path: string, format: string, ext: string): Promise<boolean> {
    const result = await withLoading(async () => {
      if (sessionId.value) {
        void api.prepCloseTagSession(sessionId.value).catch(() => {});
      }
      session.value = await api.prepOpenTagSession(path, format, ext);
      query.value = null;
      lastFilter.value = null;
      await refreshStats();
      return true;
    });
    return result === true;
  }

  async function runQuery(
    filter: TagEditOpDto["filter"],
    scope: string
  ): Promise<void> {
    await withLoading(async () => {
      lastFilter.value = filter;
      lastFilterScope.value = scope;
      await refreshQuery();
    });
  }

  async function stageOps(ops: TagEditOpDto[]): Promise<boolean> {
    const result = await withLoading(async () => {
      session.value = await api.prepStageTagOps(sessionId.value, ops);
      await Promise.all([refreshStats(), refreshQuery()]);
      return true;
    });
    return result === true;
  }

  async function undo(): Promise<void> {
    await withLoading(async () => {
      session.value = await api.prepUndoTagOp(sessionId.value);
      await Promise.all([refreshStats(), refreshQuery()]);
    });
  }

  async function setStatsScope(scope: typeof statsScope.value): Promise<void> {
    statsScope.value = scope;
    await withLoading(refreshStats);
  }

  async function commit() {
    return withLoading(async () => {
      const result = await api.prepCommitTagSession(sessionId.value);
      session.value = await api.prepTagSessionSummary(sessionId.value);
      await Promise.all([refreshStats(), refreshQuery()]);
      return result;
    });
  }

  return {
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
    runQuery,
    stageOps,
    undo,
    setStatsScope,
    commit,
  };
}
