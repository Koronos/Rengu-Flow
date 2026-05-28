import { describe, expect, it } from "vitest";
import { parseIntegerList } from "./integerList";
import { parseNumberList } from "./numberList";
import { parseStringList, stringListNeedsJsonEditor } from "./stringList";

describe("list parsers", () => {
  it("parseIntegerList accepts JSON and comma-separated strings", () => {
    expect(parseIntegerList("[512, 1024]")).toEqual([512, 1024]);
    expect(parseIntegerList("512, 1024")).toEqual([512, 1024]);
    expect(parseIntegerList(768)).toEqual([768]);
  });

  it("parseNumberList deduplicates and sorts", () => {
    expect(parseNumberList("0.5, 0.5, 1")).toEqual([0.5, 1]);
  });

  it("parseNumberList respects maxLength in input order", () => {
    expect(parseNumberList([0.9, 0.999, 0.95], 2)).toEqual([0.9, 0.999]);
    expect(parseNumberList("0.9, 0.95, 0.99", 2)).toEqual([0.9, 0.95]);
  });

  it("parseStringList splits lines", () => {
    expect(parseStringList("a\nb\n")).toEqual(["a", "b"]);
  });

  it("stringListNeedsJsonEditor detects object rows", () => {
    expect(stringListNeedsJsonEditor('[{"name":"x"}]')).toBe(true);
    expect(stringListNeedsJsonEditor(["plain"])).toBe(false);
  });
});
