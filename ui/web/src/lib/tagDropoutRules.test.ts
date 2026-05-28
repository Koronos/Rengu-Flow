import { describe, expect, it } from "vitest";
import {
  emptyTagDropoutRule,
  parseTagDropoutRules,
  ruleFromToml,
  ruleToToml,
  tagDropoutRulesFormValue,
  tagDropoutRulesNeedJsonEditor,
  tagDropoutRulesTomlValue,
  validateTagDropoutRulesJson,
} from "./tagDropoutRules";

describe("tagDropoutRules", () => {
  it("parseTagDropoutRules accepts arrays and JSON strings", () => {
    expect(
      parseTagDropoutRules([
        { tags: ["hero"], drop_probability: 0.1 },
        { tags_file: "drop.txt", drop_probability: 0.5 },
      ])
    ).toEqual([
      { source: "tags", tags: ["hero"], tags_file: "", drop_probability: 0.1 },
      { source: "file", tags: [], tags_file: "drop.txt", drop_probability: 0.5 },
    ]);
    expect(
      parseTagDropoutRules(
        '[{"tags":["a"],"drop_probability":0.2},{"tags_file":"f.txt","drop_probability":1}]'
      )
    ).toHaveLength(2);
    expect(parseTagDropoutRules("")).toEqual([]);
  });

  it("ruleToToml emits tags or tags_file only", () => {
    expect(
      ruleToToml({
        source: "tags",
        tags: ["char", "style"],
        tags_file: "",
        drop_probability: 0.08,
      })
    ).toEqual({ tags: ["char", "style"], drop_probability: 0.08 });
    expect(
      ruleToToml({
        source: "file",
        tags: [],
        tags_file: "extras.txt",
        drop_probability: 0.5,
      })
    ).toEqual({ tags_file: "extras.txt", drop_probability: 0.5 });
  });

  it("tagDropoutRulesFormValue keeps in-progress rows for the editor", () => {
    expect(tagDropoutRulesFormValue([])).toBe("");
    expect(tagDropoutRulesFormValue([emptyTagDropoutRule()])).toEqual([
      { tags: [], drop_probability: 0.1 },
    ]);
    expect(
      tagDropoutRulesFormValue([
        { source: "tags", tags: ["x"], tags_file: "", drop_probability: 0.1 },
      ])
    ).toEqual([{ tags: ["x"], drop_probability: 0.1 }]);
  });

  it("tagDropoutRulesTomlValue omits incomplete rules for save", () => {
    expect(tagDropoutRulesTomlValue([{ tags: [], drop_probability: 0.1 }])).toBe("");
    expect(
      tagDropoutRulesTomlValue([{ tags: ["x"], drop_probability: 0.1 }])
    ).toEqual([{ tags: ["x"], drop_probability: 0.1 }]);
  });

  it("parseTagDropoutRules round-trips draft tag-list rows", () => {
    expect(parseTagDropoutRules([{ tags: [], drop_probability: 0.2 }])).toEqual([
      { source: "tags", tags: [], tags_file: "", drop_probability: 0.2 },
    ]);
  });

  it("ruleFromToml prefers inline tags when both are present", () => {
    const row = ruleFromToml({
      tags: ["a"],
      tags_file: "b.txt",
      drop_probability: 0.2,
    });
    expect(row?.source).toBe("tags");
    expect(row?.tags).toEqual(["a"]);
  });

  it("validateTagDropoutRulesJson reports errors", () => {
    expect(validateTagDropoutRulesJson("")).toBeNull();
    expect(validateTagDropoutRulesJson("not-json")).toBe("Invalid JSON");
    expect(validateTagDropoutRulesJson('[{"drop_probability":0.5}]')).toBe(
      "Rule 1: set tags or tags_file"
    );
    expect(validateTagDropoutRulesJson('[{"tags":["a"],"drop_probability":2}]')).toBe(
      "Rule 1: drop_probability must be between 0 and 1"
    );
  });

  it("tagDropoutRulesNeedJsonEditor detects uneditable shapes", () => {
    expect(tagDropoutRulesNeedJsonEditor("")).toBe(false);
    expect(tagDropoutRulesNeedJsonEditor([{ tags: ["a"], drop_probability: 0.1 }])).toBe(false);
    expect(tagDropoutRulesNeedJsonEditor([{ tags: [], drop_probability: 0.1 }])).toBe(false);
    expect(tagDropoutRulesNeedJsonEditor([{ drop_probability: 0.1 }])).toBe(true);
    expect(tagDropoutRulesNeedJsonEditor("not json")).toBe(true);
  });
});
