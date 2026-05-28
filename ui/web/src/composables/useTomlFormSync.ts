import type { Ref } from "vue";

const DEFAULT_DEBOUNCE_MS = 280;

export interface TomlParseResult<TForm> {
  ok: boolean;
  form?: TForm;
  content?: string;
  error?: unknown;
  extras?: Record<string, unknown>;
}

export interface UseTomlFormSyncOptions<TForm> {
  content: Ref<string>;
  form: Ref<TForm | null>;
  syncing: Ref<boolean>;
  parseError: Ref<string>;
  debounceMs?: number;
  requireNonemptyContent?: boolean;
  sanitize: (raw: TForm) => TForm | null;
  parseToml: (content: string) => Promise<TomlParseResult<TForm>>;
  renderToml: (form: TForm) => Promise<TomlParseResult<TForm>>;
  formatError: (err: unknown) => string;
  onParsed?: (extras: Record<string, unknown>) => void;
  transformParsed?: (form: TForm) => TForm;
  onFormVersionBump?: () => void;
}

/** Debounced TOML ↔ form sync shared by config and dataset editor stores. */
export function useTomlFormSync<TForm extends object>(opts: UseTomlFormSyncOptions<TForm>) {
  const debounceMs = opts.debounceMs ?? DEFAULT_DEBOUNCE_MS;
  let parseTimer: ReturnType<typeof setTimeout> | null = null;
  let renderTimer: ReturnType<typeof setTimeout> | null = null;
  let syncLock: "toml-to-form" | "form-to-toml" | null = null;
  let lastEditSource: "toml" | "form" = "toml";

  function clearSyncTimers() {
    if (parseTimer) clearTimeout(parseTimer);
    if (renderTimer) clearTimeout(renderTimer);
    parseTimer = null;
    renderTimer = null;
  }

  async function parseFromToml() {
    if (opts.requireNonemptyContent && !(opts.content.value || "").trim()) return;
    opts.syncing.value = true;
    opts.parseError.value = "";
    try {
      const r = await opts.parseToml(opts.content.value);
      if (!r.ok) {
        opts.parseError.value =
          opts.formatError({ detail: r.error }) || "Could not parse TOML for the form";
        return;
      }
      syncLock = "toml-to-form";
      let next = opts.sanitize(r.form as TForm);
      if (!next) return;
      if (opts.transformParsed) next = opts.transformParsed(next);
      opts.form.value = next;
      opts.onFormVersionBump?.();
      opts.onParsed?.(r.extras ?? {});
    } catch (e) {
      opts.parseError.value = opts.formatError(e);
    } finally {
      syncLock = null;
      opts.syncing.value = false;
    }
  }

  async function renderFromForm() {
    if (!opts.form.value) return;
    opts.syncing.value = true;
    opts.parseError.value = "";
    try {
      const payload = opts.sanitize(opts.form.value);
      if (!payload) {
        opts.parseError.value = "Could not sync form to TOML (invalid form state).";
        return;
      }
      const r = await opts.renderToml(payload);
      if (!r.ok) {
        opts.parseError.value =
          opts.formatError({ detail: r.error }) || "Could not render TOML from form";
        return;
      }
      const nextToml = r.content ?? "";
      if (nextToml === opts.content.value) return;
      syncLock = "form-to-toml";
      opts.content.value = nextToml;
    } catch (e) {
      opts.parseError.value = opts.formatError(e);
    } finally {
      syncLock = null;
      opts.syncing.value = false;
    }
  }

  function scheduleParseFromToml() {
    if (syncLock === "form-to-toml") return;
    if (parseTimer) clearTimeout(parseTimer);
    parseTimer = setTimeout(() => parseFromToml(), debounceMs);
  }

  function scheduleRenderFromForm() {
    if (syncLock === "toml-to-form") return;
    if (renderTimer) clearTimeout(renderTimer);
    renderTimer = setTimeout(() => renderFromForm(), debounceMs);
  }

  function setContent(toml: string) {
    lastEditSource = "toml";
    if (renderTimer) clearTimeout(renderTimer);
    renderTimer = null;
    opts.content.value = toml;
    scheduleParseFromToml();
  }

  function setForm(nextForm: TForm) {
    lastEditSource = "form";
    if (parseTimer) clearTimeout(parseTimer);
    parseTimer = null;
    const clean = opts.sanitize(nextForm);
    if (!clean) return;
    opts.form.value = clean;
    scheduleRenderFromForm();
  }

  async function applyToml(toml: string) {
    clearSyncTimers();
    syncLock = null;
    lastEditSource = "toml";
    opts.parseError.value = "";
    opts.content.value = toml;
    await parseFromToml();
  }

  async function flushSync() {
    clearSyncTimers();
    if (lastEditSource === "form") {
      await renderFromForm();
      await parseFromToml();
    } else {
      await parseFromToml();
    }
  }

  function resetSyncState() {
    clearSyncTimers();
    syncLock = null;
    lastEditSource = "toml";
  }

  return {
    setContent,
    setForm,
    applyToml,
    flushSync,
    resetSyncState,
  };
}
