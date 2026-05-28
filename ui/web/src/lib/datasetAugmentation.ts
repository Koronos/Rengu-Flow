import type { FormValues, SchemaField } from "../types/forms";

export interface AugParamField {
  path: string;
  label: string;
  type: "number" | "integer" | "select" | "boolean";
  default?: number | string | boolean;
  min?: number;
  max?: number;
  step?: number;
  options?: string[];
  help?: string;
}

export interface AugStrategyCatalogEntry {
  name: string;
  label: string;
  category: "geometric" | "photometric" | string;
  implemented: boolean;
  enumerable: boolean;
  help?: string;
  parameters: AugParamField[];
}

/** Map dataset schema ``augmentation_directory_fields`` by path for FieldHelpIcon. */
export function schemaFieldsByPath(fields: SchemaField[] | undefined): Map<string, SchemaField> {
  const map = new Map<string, SchemaField>();
  for (const field of fields ?? []) {
    if (field.path) map.set(field.path, field);
  }
  return map;
}

export function lookupSchemaField(
  fieldsByPath: Map<string, SchemaField>,
  path: string,
  label: string
): SchemaField {
  return (
    fieldsByPath.get(path) ?? {
      path,
      label,
      type: "string",
      help: "",
    }
  );
}

export interface AugPresetCatalogEntry {
  name: string;
  label: string;
  available: boolean;
  deferred?: boolean;
  strategies: string[];
  /** Full preset strategy map (same as training ``get_preset_strategies``). */
  strategy_defaults?: Record<string, Record<string, unknown>>;
}

export interface AugmentationCatalog {
  version?: string;
  seed_modes?: string[];
  variant_sampling_modes?: string[];
  presets?: AugPresetCatalogEntry[];
  strategies?: AugStrategyCatalogEntry[];
}

export interface AugmentationConfig {
  enabled?: boolean;
  preset?: string;
  seed_mode?: string;
  variant_sampling?: string;
  max_branches_per_image?: number;
  enable_strategies?: string[];
  strategies?: Record<string, Record<string, unknown>>;
}

export const DEFAULT_AUGMENTATION: AugmentationConfig = {
  enabled: false,
  preset: "none",
  seed_mode: "deterministic_per_image",
  variant_sampling: "probability",
};

function parseJsonObject(raw: unknown): Record<string, unknown> | null {
  if (!raw) return null;
  if (typeof raw === "object" && !Array.isArray(raw)) {
    return raw as Record<string, unknown>;
  }
  if (typeof raw === "string" && raw.trim()) {
    try {
      const parsed = JSON.parse(raw) as unknown;
      if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) {
        return parsed as Record<string, unknown>;
      }
    } catch {
      return null;
    }
  }
  return null;
}

function parseStrategies(raw: unknown): Record<string, Record<string, unknown>> | undefined {
  const obj = parseJsonObject(raw);
  if (!obj) return undefined;
  const out: Record<string, Record<string, unknown>> = {};
  for (const [name, value] of Object.entries(obj)) {
    if (value && typeof value === "object" && !Array.isArray(value)) {
      out[name] = { ...(value as Record<string, unknown>) };
    }
  }
  return Object.keys(out).length ? out : undefined;
}

export function parseAugmentationConfig(raw: unknown): AugmentationConfig | null {
  const obj = parseJsonObject(raw);
  if (!obj) return null;
  const config: AugmentationConfig = {};
  if ("enabled" in obj) config.enabled = Boolean(obj.enabled);
  if (typeof obj.preset === "string") config.preset = obj.preset;
  if (typeof obj.seed_mode === "string") config.seed_mode = obj.seed_mode;
  if (typeof obj.variant_sampling === "string") config.variant_sampling = obj.variant_sampling;
  if (obj.max_branches_per_image != null && obj.max_branches_per_image !== "") {
    config.max_branches_per_image = Number(obj.max_branches_per_image);
  }
  if (Array.isArray(obj.enable_strategies)) {
    config.enable_strategies = obj.enable_strategies.map(String);
  }
  const strategies = parseStrategies(obj.strategies);
  if (strategies) config.strategies = strategies;
  return config;
}

export function parseGlobalAugmentation(form: FormValues | null): AugmentationConfig | null {
  if (!form) return null;
  return parseAugmentationConfig(form._dataset_augmentation);
}

export function parseDirectoryAugmentation(entry: FormValues | null | undefined): AugmentationConfig | null {
  if (!entry) return null;
  return parseAugmentationConfig(entry.augmentation);
}

export function serializeAugmentationConfig(
  config: AugmentationConfig | null | undefined,
  options: { forForm?: boolean } = {}
): Record<string, unknown> | string | undefined {
  const { forForm = false } = options;
  if (!config) return undefined;
  const out: Record<string, unknown> = {};
  if (config.enabled != null) out.enabled = config.enabled;
  if (config.preset) out.preset = config.preset;
  if (config.seed_mode) out.seed_mode = config.seed_mode;
  if (config.variant_sampling) out.variant_sampling = config.variant_sampling;
  if (config.max_branches_per_image != null && Number.isFinite(config.max_branches_per_image)) {
    out.max_branches_per_image = config.max_branches_per_image;
  }
  if (config.enable_strategies?.length) {
    out.enable_strategies = [...config.enable_strategies];
  }
  if (config.strategies && Object.keys(config.strategies).length) {
    out.strategies = forForm
      ? JSON.stringify(config.strategies, null, 2)
      : { ...config.strategies };
  }
  return Object.keys(out).length ? out : undefined;
}

export function serializeGlobalAugmentation(
  config: AugmentationConfig | null | undefined
): string | undefined {
  if (!config) return undefined;
  const payload: Record<string, unknown> = {};
  if (config.enabled != null) payload.enabled = config.enabled;
  if (config.preset) payload.preset = config.preset;
  if (config.seed_mode && config.enabled) payload.seed_mode = config.seed_mode;
  if (config.variant_sampling && config.enabled) payload.variant_sampling = config.variant_sampling;
  if (config.max_branches_per_image != null && config.enabled) {
    payload.max_branches_per_image = config.max_branches_per_image;
  }
  if (config.enable_strategies?.length && config.enabled) {
    payload.enable_strategies = config.enable_strategies;
  }
  if (config.strategies && Object.keys(config.strategies).length && config.enabled) {
    payload.strategies = config.strategies;
  }
  if (!Object.keys(payload).length) return undefined;
  return JSON.stringify(payload, null, 2);
}

export function serializeDirectoryAugmentation(
  config: AugmentationConfig | null | undefined,
  options: { global?: AugmentationConfig } = {}
): Record<string, unknown> | undefined {
  const global = options.global;
  if (!config) return undefined;
  if (global && !shouldWriteDirectoryAugmentation(config, global)) {
    return undefined;
  }
  const raw = serializeAugmentationConfig(config, { forForm: true });
  if (!raw || typeof raw !== "object") return undefined;
  const out = { ...(raw as Record<string, unknown>) };
  if (global) {
    if (out.enabled === global.enabled) delete out.enabled;
    if (out.preset === global.preset || (out.preset === "none" && !global.preset)) {
      delete out.preset;
    }
    if (out.seed_mode === global.seed_mode) delete out.seed_mode;
    if (out.variant_sampling === global.variant_sampling) delete out.variant_sampling;
    if (out.max_branches_per_image === global.max_branches_per_image) {
      delete out.max_branches_per_image;
    }
    if (
      Array.isArray(out.enable_strategies) &&
      JSON.stringify(out.enable_strategies) === JSON.stringify(global.enable_strategies ?? [])
    ) {
      delete out.enable_strategies;
    }
  }
  return Object.keys(out).length ? out : undefined;
}

export function isAugmentationEnabled(config: AugmentationConfig | null | undefined): boolean {
  return Boolean(config?.enabled);
}

export function hasDirectoryAugmentationOverride(entry: FormValues | null | undefined): boolean {
  return entry != null && Object.prototype.hasOwnProperty.call(entry, "augmentation");
}

/** Write `[dataset.augmentation]` only when there is meaningful configuration. */
export function shouldWriteGlobalAugmentation(config: AugmentationConfig): boolean {
  if (config.enabled) return true;
  if (config.preset && config.preset !== "none") return true;
  if (config.enable_strategies?.length) return true;
  if (config.strategies && Object.keys(config.strategies).length > 0) return true;
  if (config.seed_mode && config.seed_mode !== DEFAULT_AUGMENTATION.seed_mode) return true;
  if (config.variant_sampling && config.variant_sampling !== DEFAULT_AUGMENTATION.variant_sampling) {
    return true;
  }
  if (config.max_branches_per_image != null && Number.isFinite(config.max_branches_per_image)) {
    return true;
  }
  return false;
}

/** Write per-folder `augmentation` when it differs from global or carries strategy overrides. */
export function shouldWriteDirectoryAugmentation(
  config: AugmentationConfig | null | undefined,
  globalConfig: AugmentationConfig
): boolean {
  if (!config) return false;
  if (config.strategies && Object.keys(config.strategies).length > 0) return true;
  if (config.enable_strategies?.length) return true;
  if (config.enabled !== globalConfig.enabled) return true;
  if ((config.preset || "none") !== (globalConfig.preset || "none")) return true;
  if (config.seed_mode && config.seed_mode !== (globalConfig.seed_mode ?? DEFAULT_AUGMENTATION.seed_mode)) {
    return true;
  }
  if (
    config.variant_sampling &&
    config.variant_sampling !== (globalConfig.variant_sampling ?? DEFAULT_AUGMENTATION.variant_sampling)
  ) {
    return true;
  }
  if (config.max_branches_per_image != null && Number.isFinite(config.max_branches_per_image)) {
    return true;
  }
  return false;
}

export function directoryAugmentationNeedsFullEditor(
  config: AugmentationConfig | null | undefined,
  globalConfig: AugmentationConfig
): boolean {
  if (!config) return false;
  if (config.enabled !== globalConfig.enabled) return true;
  if ((config.preset || "none") !== (globalConfig.preset || "none")) return true;
  if (config.seed_mode && config.seed_mode !== (globalConfig.seed_mode ?? DEFAULT_AUGMENTATION.seed_mode)) {
    return true;
  }
  if (
    config.variant_sampling &&
    config.variant_sampling !== (globalConfig.variant_sampling ?? DEFAULT_AUGMENTATION.variant_sampling)
  ) {
    return true;
  }
  if (config.max_branches_per_image != null && Number.isFinite(config.max_branches_per_image)) {
    return true;
  }
  return false;
}

function cloneStrategyMap(
  map: Record<string, Record<string, unknown>> | undefined
): Record<string, Record<string, unknown>> {
  if (!map) return {};
  const out: Record<string, Record<string, unknown>> = {};
  for (const [name, params] of Object.entries(map)) {
    out[name] = { ...params };
  }
  return out;
}

export function presetStrategyDefaults(
  catalog: AugmentationCatalog | null,
  preset: string | undefined
): Record<string, Record<string, unknown>> {
  const name = preset || "none";
  if (name === "none" || name === "custom") return {};
  const entry = catalog?.presets?.find((p) => p.name === name);
  return cloneStrategyMap(entry?.strategy_defaults);
}

export function mergeStrategyParams(
  base: Record<string, unknown> | undefined,
  override: Record<string, unknown> | undefined
): Record<string, unknown> {
  const out = { ...(base ?? {}) };
  if (override) {
    for (const [key, value] of Object.entries(override)) {
      out[key] = value;
    }
  }
  if (!("enabled" in out)) out.enabled = true;
  return out;
}

/** Resolved strategy name → params (preset defaults merged with stored overrides). */
export function effectiveStrategyMap(
  config: AugmentationConfig,
  catalog: AugmentationCatalog | null
): Record<string, Record<string, unknown>> {
  const preset = config.preset || "none";
  const base = presetStrategyDefaults(catalog, preset);
  const overrides = config.strategies ?? {};
  const names = new Set([...Object.keys(base), ...Object.keys(overrides)]);
  const merged: Record<string, Record<string, unknown>> = {};
  for (const name of names) {
    merged[name] = mergeStrategyParams(base[name], overrides[name]);
  }
  if (config.enable_strategies?.length) {
    const allow = new Set(config.enable_strategies);
    for (const name of Object.keys(merged)) {
      if (!allow.has(name)) delete merged[name];
    }
  }
  return merged;
}

export function effectiveStrategyNames(
  config: AugmentationConfig,
  catalog: AugmentationCatalog | null
): string[] {
  return Object.keys(effectiveStrategyMap(config, catalog)).sort();
}

/** Drop override entries that match preset defaults (compact TOML). */
export function strategyParamsDiff(
  presetParams: Record<string, unknown> | undefined,
  edited: Record<string, unknown>
): Record<string, unknown> | null {
  const baseline = mergeStrategyParams(presetParams, undefined);
  const resolved = mergeStrategyParams(presetParams, edited);
  const diff: Record<string, unknown> = {};
  for (const [key, value] of Object.entries(resolved)) {
    if (baseline[key] !== value) diff[key] = value;
  }
  return Object.keys(diff).length ? diff : null;
}

export function compactStrategiesForStorage(
  config: AugmentationConfig,
  catalog: AugmentationCatalog | null
): AugmentationConfig {
  if (!config.strategies || !Object.keys(config.strategies).length) {
    return config;
  }
  const preset = config.preset || "none";
  const defaults = presetStrategyDefaults(catalog, preset);
  const compact: Record<string, Record<string, unknown>> = {};
  for (const [name, params] of Object.entries(config.strategies)) {
    const diff = strategyParamsDiff(defaults[name], params);
    if (diff) compact[name] = diff;
  }
  return {
    ...config,
    strategies: Object.keys(compact).length ? compact : undefined,
  };
}

/** Reset per-strategy overrides when the preset bundle changes. */
export function applyPresetChange(
  config: AugmentationConfig,
  preset: string
): AugmentationConfig {
  return {
    ...config,
    preset,
    enable_strategies: undefined,
    strategies: undefined,
  };
}

export function catalogDefaultsForStrategy(
  catalog: AugmentationCatalog | null,
  name: string
): Record<string, unknown> {
  const meta = catalog?.strategies?.find((s) => s.name === name);
  const defaults: Record<string, unknown> = { enabled: true };
  for (const field of meta?.parameters ?? []) {
    if (field.default !== undefined) defaults[field.path] = field.default;
  }
  return defaults;
}

export function summarizeAugmentation(
  config: AugmentationConfig | null | undefined,
  catalog: AugmentationCatalog | null
): string {
  if (!config) return "Not set";
  const preset = config.preset || "none";
  const presetLabel =
    catalog?.presets?.find((p) => p.name === preset)?.label || preset;
  if (!config.enabled) return `Off · preset ${presetLabel}`;
  const strategyCount = effectiveStrategyNames(config, catalog).length;
  const extra = strategyCount ? ` · ${strategyCount} strateg${strategyCount === 1 ? "y" : "ies"}` : "";
  return `On · ${presetLabel}${extra}`;
}

export function emptyAugmentationConfig(): AugmentationConfig {
  return { ...DEFAULT_AUGMENTATION };
}

export function mergeAugmentationPatch(
  base: AugmentationConfig | null,
  patch: Partial<AugmentationConfig>
): AugmentationConfig {
  const next = { ...(base ?? emptyAugmentationConfig()), ...patch };
  if (patch.strategies === undefined && base?.strategies) {
    next.strategies = { ...base.strategies };
  } else if (patch.strategies) {
    next.strategies = { ...patch.strategies };
  }
  return next;
}

export function setStrategyOverride(
  config: AugmentationConfig,
  name: string,
  params: Record<string, unknown> | null
): AugmentationConfig {
  const strategies = { ...(config.strategies ?? {}) };
  if (!params || !Object.keys(params).length) {
    delete strategies[name];
  } else {
    strategies[name] = { ...params };
  }
  return { ...config, strategies: Object.keys(strategies).length ? strategies : undefined };
}

export function availablePresets(catalog: AugmentationCatalog | null): AugPresetCatalogEntry[] {
  return (catalog?.presets ?? []).filter((p) => p.available);
}

export function implementedStrategies(
  catalog: AugmentationCatalog | null
): AugStrategyCatalogEntry[] {
  return (catalog?.strategies ?? []).filter((s) => s.implemented);
}