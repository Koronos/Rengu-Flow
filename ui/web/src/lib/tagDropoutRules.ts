/** Dataset TOML `tag_dropout_rules` — per-tag or file-based drop probabilities. */

export type TagDropoutRuleSource = "tags" | "file";

export interface TagDropoutRuleToml {
  tags?: string[];
  tags_file?: string;
  drop_probability: number;
}

export interface TagDropoutRuleUi {
  source: TagDropoutRuleSource;
  tags: string[];
  tags_file: string;
  drop_probability: number;
}

function clampProbability(value: number): number {
  if (!Number.isFinite(value)) return 0;
  return Math.min(1, Math.max(0, value));
}

function normalizeTagList(raw: unknown): string[] {
  if (!Array.isArray(raw)) return [];
  const seen = new Set<string>();
  const out: string[] = [];
  for (const item of raw) {
    const tag = String(item).trim();
    if (!tag || seen.has(tag)) continue;
    seen.add(tag);
    out.push(tag);
  }
  return out;
}

export function ruleFromToml(entry: unknown): TagDropoutRuleUi | null {
  if (!entry || typeof entry !== "object" || Array.isArray(entry)) return null;
  const row = entry as Record<string, unknown>;
  const tags = normalizeTagList(row.tags);
  const hasTagsKey = Array.isArray(row.tags);
  const tagsFileRaw = row.tags_file;
  const hasTagsFileKey = typeof tagsFileRaw === "string";
  const tagsFile = hasTagsFileKey ? tagsFileRaw.trim() : "";
  const dropProbability = clampProbability(Number(row.drop_probability ?? 0));

  if (tags.length > 0 || (hasTagsKey && !hasTagsFileKey && !tagsFile)) {
    return {
      source: "tags",
      tags,
      tags_file: "",
      drop_probability: dropProbability,
    };
  }
  if (tagsFile || (hasTagsFileKey && !hasTagsKey)) {
    return {
      source: "file",
      tags: [],
      tags_file: tagsFile,
      drop_probability: dropProbability,
    };
  }
  return null;
}

export function parseTagDropoutRules(raw: unknown): TagDropoutRuleUi[] {
  if (raw === undefined || raw === null || raw === "") return [];
  let parsed: unknown = raw;
  if (typeof raw === "string") {
    const text = raw.trim();
    if (!text) return [];
    try {
      parsed = JSON.parse(text);
    } catch {
      return [];
    }
  }
  if (!Array.isArray(parsed)) return [];
  return parsed
    .map((entry) => ruleFromToml(entry))
    .filter((row): row is TagDropoutRuleUi => row !== null);
}

export function ruleToToml(rule: TagDropoutRuleUi): TagDropoutRuleToml {
  const drop_probability = clampProbability(rule.drop_probability);
  if (rule.source === "file") {
    const tags_file = rule.tags_file.trim();
    if (!tags_file) return { tags_file: "", drop_probability };
    return { tags_file, drop_probability };
  }
  return { tags: normalizeTagList(rule.tags), drop_probability };
}

function isCompleteTagDropoutRuleToml(row: TagDropoutRuleToml): boolean {
  if (row.tags_file?.trim()) return true;
  return Boolean(row.tags?.length);
}

/** Rules ready for dataset TOML (incomplete drafts omitted). */
export function tagDropoutRulesTomlValue(raw: unknown): TagDropoutRuleToml[] | "" {
  const out = parseTagDropoutRules(raw)
    .map((rule) => ruleToToml(rule))
    .filter(isCompleteTagDropoutRuleToml);
  if (!out.length) return "";
  return out;
}

/** Form state for the rule editor (keeps in-progress rows). */
export function tagDropoutRulesFormValue(rules: TagDropoutRuleUi[]): TagDropoutRuleToml[] | "" {
  if (!rules.length) return "";
  return rules.map((rule) => ruleToToml(rule));
}

export function emptyTagDropoutRule(): TagDropoutRuleUi {
  return {
    source: "tags",
    tags: [],
    tags_file: "",
    drop_probability: 0.1,
  };
}

export function validateTagDropoutRulesJson(text: string): string | null {
  const trimmed = text.trim();
  if (!trimmed) return null;
  let parsed: unknown;
  try {
    parsed = JSON.parse(trimmed);
  } catch {
    return "Invalid JSON";
  }
  if (!Array.isArray(parsed)) {
    return "Expected an array of rule objects";
  }
  for (let i = 0; i < parsed.length; i += 1) {
    const entry = parsed[i];
    if (!entry || typeof entry !== "object" || Array.isArray(entry)) {
      return `Rule ${i + 1}: expected an object`;
    }
    const row = entry as Record<string, unknown>;
    const tags = normalizeTagList(row.tags);
    const tagsFile = typeof row.tags_file === "string" ? row.tags_file.trim() : "";
    if (!tags.length && !tagsFile) {
      return `Rule ${i + 1}: set tags or tags_file`;
    }
    if ("drop_probability" in row) {
      const p = Number(row.drop_probability);
      if (!Number.isFinite(p) || p < 0 || p > 1) {
        return `Rule ${i + 1}: drop_probability must be between 0 and 1`;
      }
    }
  }
  return null;
}

export function tagDropoutRulesNeedJsonEditor(raw: unknown): boolean {
  if (raw === undefined || raw === null || raw === "") return false;
  if (typeof raw === "string") {
    const trimmed = raw.trim();
    if (!trimmed) return false;
    try {
      const parsed = JSON.parse(trimmed);
      if (!Array.isArray(parsed)) return true;
      return parsed.some((entry) => ruleFromToml(entry) === null && entry != null);
    } catch {
      return true;
    }
  }
  if (!Array.isArray(raw)) return true;
  return raw.some((entry) => ruleFromToml(entry) === null);
}

export function formatTagDropoutRuleSummary(rule: TagDropoutRuleUi): string {
  const p = Math.round(rule.drop_probability * 100);
  if (rule.source === "file") {
    const name = rule.tags_file.trim() || "…";
    return `${name} @ ${p}%`;
  }
  const tags = rule.tags;
  if (!tags.length) return `… @ ${p}%`;
  const label = tags.length <= 2 ? tags.join(", ") : `${tags.slice(0, 2).join(", ")} +${tags.length - 2}`;
  return `${label} @ ${p}%`;
}
