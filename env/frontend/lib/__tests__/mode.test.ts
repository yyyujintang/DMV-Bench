import { describe, expect, it, vi, beforeEach } from "vitest";

// Mutable mock store — the mock factory returns a function that reads it.
// Each test resets `headerStore` to drive the next call.
let headerStore: Map<string, string> = new Map();

vi.mock("next/headers", () => ({
  headers: () => ({
    get: (key: string) => headerStore.get(key.toLowerCase()) ?? null,
  }),
}));

import { filterAnchorVariants, getMode, isAnchor, emptyMode } from "@/lib/mode";

beforeEach(() => {
  headerStore = new Map();
});

describe("getMode", () => {
  it("returns encoding defaults when no x-mode header is set", async () => {
    const m = await getMode();
    expect(m.mode).toBe("encoding");
    expect(m.anchorVariantIds.size).toBe(0);
    expect(m.taskId).toBeNull();
    expect(m.source).toBe("default");
  });

  it("parses x-mode=recall and a comma-separated anchor list", async () => {
    headerStore.set("x-mode", "recall");
    headerStore.set("x-anchors", "var1, var2 ,var3");
    const m = await getMode();
    expect(m.mode).toBe("recall");
    expect(Array.from(m.anchorVariantIds).sort()).toEqual(["var1", "var2", "var3"]);
    expect(m.taskId).toBeNull();
    expect(m.source).toBe("url");
  });

  it("treats an unknown x-mode value as encoding (anchors still parse)", async () => {
    headerStore.set("x-mode", "bogus");
    headerStore.set("x-anchors", "v");
    const m = await getMode();
    expect(m.mode).toBe("encoding");
    expect(m.anchorVariantIds.has("v")).toBe(true);
    expect(m.source).toBe("url");
  });
});

describe("isAnchor", () => {
  it("is true exactly when the id is in the anchor set", () => {
    const m = {
      mode: "recall" as const,
      anchorVariantIds: new Set(["a", "b"]),
      taskId: null,
      source: "url" as const,
    };
    expect(isAnchor("a", m)).toBe(true);
    expect(isAnchor("b", m)).toBe(true);
    expect(isAnchor("c", m)).toBe(false);
  });
});

describe("emptyMode", () => {
  it("is the encoding zero value", () => {
    const m = emptyMode();
    expect(m.mode).toBe("encoding");
    expect(m.anchorVariantIds.size).toBe(0);
    expect(m.taskId).toBeNull();
  });
});

describe("filterAnchorVariants", () => {
  const list = [{ id: "a" }, { id: "b" }, { id: "c" }];

  it("passes through unchanged in encoding mode", () => {
    const m = {
      mode: "encoding" as const,
      anchorVariantIds: new Set(["a"]),
      taskId: null,
    };
    expect(filterAnchorVariants(list, m)).toEqual(list);
  });

  it("drops anchor IDs from the list in recall mode", () => {
    const m = {
      mode: "recall" as const,
      anchorVariantIds: new Set(["a", "c"]),
      taskId: null,
    };
    expect(filterAnchorVariants(list, m).map((x) => x.id)).toEqual(["b"]);
  });

  it("returns a fresh array (does not mutate the input)", () => {
    const m = emptyMode();
    const out = filterAnchorVariants(list, m);
    expect(out).not.toBe(list);
    expect(out).toEqual(list);
  });
});
