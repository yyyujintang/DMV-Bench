import { describe, expect, it } from "vitest";
import { scoreAttempt } from "@/lib/annotate/scoring";
import type { TaskInstance } from "@/lib/annotate/task-instance";

function makeTask(overrides: Partial<TaskInstance> = {}): TaskInstance {
  return {
    task_id: "nc_test",
    sub_task: "NC",
    grain_tier: 2,
    category_ids: ["sofas"],
    variants_used: ["A", "B"],
    turns: [],
    ncp_metadata: {
      anchor_variant_ids: ["A"],
      recall_turn_indices: [5],
      memory_gated_branch_turn: 5,
      cross_turn_predicates: [],
    },
    ground_truth: {
      final_action: "navigate",
      target_url: "/product/B",
      target_variant_id: "B",
      accepted_alternatives: ["B", "C", "D"],
    },
    success_criteria: {
      type: "variant_match",
      evaluator_fn: "match_final_variant_with_alternatives",
      tolerance: null,
    },
    ...overrides,
  };
}

describe("scoreAttempt", () => {
  it("gaveUp short-circuits to incorrect", () => {
    const r = scoreAttempt(makeTask(), { finalUrl: "/product/B", gaveUp: true });
    expect(r.correct).toBe(false);
    expect(r.matchedAlternative).toBe(false);
  });

  it("matches the exact target_url (W7 url_match)", () => {
    const r = scoreAttempt(makeTask(), { finalUrl: "/product/B", gaveUp: false });
    expect(r.correct).toBe(true);
  });

  it("rejects a different /product/ URL — no alternatives in W7", () => {
    const r = scoreAttempt(makeTask(), { finalUrl: "/product/C", gaveUp: false });
    expect(r.correct).toBe(false);
  });

  it("rejects an unrelated variant", () => {
    const r = scoreAttempt(makeTask(), { finalUrl: "/product/Z", gaveUp: false });
    expect(r.correct).toBe(false);
  });

  it("rejects a non-target URL", () => {
    const r = scoreAttempt(makeTask(), { finalUrl: "/category/sofas", gaveUp: false });
    expect(r.correct).toBe(false);
  });

  it("url_match compares full target_url", () => {
    const t = makeTask({
      success_criteria: { type: "url_match", evaluator_fn: "match_final_url", tolerance: null },
      ground_truth: {
        final_action: "navigate",
        target_url: "/category/sofas/modern",
        target_variant_id: "B",
        accepted_alternatives: [],
      },
    });
    expect(scoreAttempt(t, { finalUrl: "/category/sofas/modern", gaveUp: false }).correct).toBe(true);
    expect(scoreAttempt(t, { finalUrl: "/category/sofas/modern/", gaveUp: false }).correct).toBe(true);
    expect(scoreAttempt(t, { finalUrl: "/category/sofas/vintage", gaveUp: false }).correct).toBe(false);
  });

  it("null finalUrl is incorrect", () => {
    const r = scoreAttempt(makeTask(), { finalUrl: null, gaveUp: false });
    expect(r.correct).toBe(false);
  });
});
