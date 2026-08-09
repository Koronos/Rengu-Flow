import { describe, expect, it } from "vitest";
import { gutterRows } from "./workflowGutter";
import { LANE_COUNT, edgeKey } from "./workflowLayout";

const chain = (...ids: string[]) =>
  ids.map((id, index) => ({ id, from: index ? ids[index - 1] : null }));

describe("gutterRows", () => {
  it("returns one (possibly empty) row per node", () => {
    const rows = gutterRows(chain("a", "b", "c"));
    expect(rows).toHaveLength(3);
    expect(rows[0].every((s) => s.span === "bottom")).toBe(true);
  });

  it("gives a source node nothing when it has no from", () => {
    expect(gutterRows([{ id: "a", from: null }])).toEqual([[]]);
  });

  it("splits a consecutive link into a bottom and a top half, capless", () => {
    const rows = gutterRows(chain("a", "b"));
    const key = edgeKey("a", "b");
    expect(rows[0]).toEqual([
      { key, lane: 0, from: "a", to: "b", span: "bottom", cap: null },
    ]);
    expect(rows[1]).toEqual([{ key, lane: 0, from: "a", to: "b", span: "top", cap: null }]);
  });

  it("runs a jump through every skipped row and caps both ends", () => {
    // a -> b -> c plus the jump a -> d, which reads past b and c.
    const nodes = [
      { id: "a", from: null },
      { id: "b", from: "a" },
      { id: "c", from: "b" },
      { id: "d", from: "a" },
    ];
    const rows = gutterRows(nodes);
    const jump = edgeKey("a", "d");

    const pieces = rows.map((row) => row.find((s) => s.key === jump));
    expect(pieces.map((p) => p?.span)).toEqual(["bottom", "full", "full", "top"]);
    expect(pieces.map((p) => p?.cap)).toEqual(["start", null, null, "end"]);
    expect(pieces.every((p) => (p?.lane ?? 0) > 0)).toBe(true);
  });

  it("keeps a row's consecutive segment alongside a jump crossing it", () => {
    const nodes = [
      { id: "a", from: null },
      { id: "b", from: "a" },
      { id: "c", from: "a" },
    ];
    const rows = gutterRows(nodes);
    // Row b carries its own arrival (lane 0, top) and the a->c jump passing through.
    expect(rows[1].map((s) => `${s.key}:${s.span}`).sort()).toEqual(
      [`${edgeKey("a", "b")}:top`, `${edgeKey("a", "c")}:full`].sort()
    );
  });

  it("draws nothing for an edge the packer could not fit", () => {
    // Four mutually overlapping jumps out of one source: only LANE_COUNT-1 offset lanes exist.
    const nodes = [
      { id: "src", from: null },
      { id: "p1", from: "src" },
      { id: "p2", from: "src" },
      { id: "p3", from: "src" },
      { id: "p4", from: "src" },
      { id: "p5", from: "src" },
    ];
    const rows = gutterRows(nodes);
    const drawnJumps = new Set(
      rows.flat().filter((s) => s.lane > 0).map((s) => s.key)
    );
    expect(drawnJumps.size).toBe(LANE_COUNT - 1);
    // The overflow edge is absent everywhere, not drawn at lane -1.
    expect(rows.flat().some((s) => s.lane < 0)).toBe(false);
  });

  it("ignores a from that points forward or at a missing node", () => {
    const rows = gutterRows([
      { id: "a", from: "b" },
      { id: "b", from: "ghost" },
    ]);
    expect(rows).toEqual([[], []]);
  });
});
