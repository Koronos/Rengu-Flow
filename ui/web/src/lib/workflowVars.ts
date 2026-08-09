/**
 * `${name}` substitution for the editor's inline preview.
 *
 * This is the **one duplication of server logic accepted in the whole design**, and only because
 * a round-trip per keystroke costs more than ~25 lines. The server
 * (`rengu_flow_ui/workflow_graph.py::resolve_text`) remains the source of truth: it resolves at
 * execution, per node, immediately before launch, and pre-flight — not this file — is what
 * decides whether a workflow may run. Both sides are tested against the same edge cases.
 *
 * The rules, all three of them:
 *
 * * `${name}` with `name` matching `[A-Za-z_][A-Za-z0-9_]*`; `$$` is a literal `$`.
 * * **One pass, no recursion** — a value that itself contains `${other}` is left as-is.
 * * A missing variable keeps its literal `${name}`. It is never emptied: substituting `""` would
 *   silently run a stage against `/`.
 */

import type { WorkflowGraph, WorkflowVariable } from "./workflowGraph";

/** `$$` (escaped literal `$`) or `${name}`. One regex for substitution *and* ref collection, so
 * both agree on what the escape hides. */
const TOKEN_RE = /\$\$|\$\{([A-Za-z_][A-Za-z0-9_]*)\}/g;

export type VariableSource =
  | Record<string, unknown>
  | readonly WorkflowVariable[]
  | null
  | undefined;

/** Accept either a plain object or the graph's `variables` list. */
export function variableMap(variables: VariableSource): Record<string, string> {
  if (!variables) return {};
  const entries: [string, unknown][] = Array.isArray(variables)
    ? (variables as readonly WorkflowVariable[]).map((v) => [v.name, v.value])
    : Object.entries(variables as Record<string, unknown>);
  const out: Record<string, string> = {};
  for (const [name, value] of entries) {
    out[String(name)] = typeof value === "string" ? value : String(value);
  }
  return out;
}

/**
 * Substitute `${name}` in *text*. A single pass: the replacement is never re-scanned, so a
 * variable whose value contains `${other}` comes out literal.
 */
export function resolveText(text: string, variables: VariableSource): string {
  if (typeof text !== "string") return text;
  const mapping = variableMap(variables);
  // A function replacement is what makes this safe: the returned string is inserted verbatim,
  // so a value containing `$&` or `$1` is not re-interpreted.
  return text.replace(TOKEN_RE, (match, name?: string) => {
    if (name === undefined) return "$"; // the `$$` branch
    return name in mapping ? mapping[name] : match;
  });
}

/** Resolve strings only, recursively. Numbers and booleans are never touched. */
export function resolveValue<T>(value: T, variables: VariableSource): T {
  const mapping = variableMap(variables);
  const walk = (item: unknown): unknown => {
    if (typeof item === "string") return resolveText(item, mapping);
    if (Array.isArray(item)) return item.map(walk);
    if (item && typeof item === "object") {
      const out: Record<string, unknown> = {};
      for (const [key, child] of Object.entries(item as Record<string, unknown>)) {
        out[key] = walk(child);
      }
      return out;
    }
    return item;
  };
  return walk(value) as T;
}

/** `prep.quality` -> `quality`; every other type is its own label. */
function configLabel(type: string): string {
  return type.startsWith("prep.") ? type.slice("prep.".length) : type;
}

function* walkStrings(value: unknown, path: string): Generator<[string, string]> {
  if (typeof value === "string") {
    yield [path, value];
  } else if (Array.isArray(value)) {
    for (let i = 0; i < value.length; i += 1) yield* walkStrings(value[i], `${path}[${i}]`);
  } else if (value && typeof value === "object") {
    for (const [key, child] of Object.entries(value as Record<string, unknown>)) {
      yield* walkStrings(child, `${path}.${key}`);
    }
  }
}

/**
 * Variable name -> the places it is used, for the editor's "used by" column.
 *
 * Every referenced name is reported, **including ones the workflow does not define** — the UI
 * looks up by name, and an undefined reference is exactly what the user needs to find.
 */
export function collectRefs(graph: Pick<WorkflowGraph, "nodes">): Record<string, string[]> {
  const refs: Record<string, string[]> = {};
  for (const node of graph.nodes) {
    for (const [path, text] of walkStrings(node.config, configLabel(node.type))) {
      for (const match of text.matchAll(TOKEN_RE)) {
        const name = match[1];
        if (name === undefined) continue; // `$$` is an escape, not a reference
        const where = `${node.id} · ${path}`;
        const locations = (refs[name] ??= []);
        if (!locations.includes(where)) locations.push(where);
      }
    }
  }
  return refs;
}

/** The referenced names the workflow does not define — what pre-flight will refuse to run on. */
export function unknownRefs(graph: Pick<WorkflowGraph, "nodes" | "variables">): string[] {
  const defined = new Set(graph.variables.map((variable) => variable.name));
  return Object.keys(collectRefs(graph)).filter((name) => !defined.has(name));
}
