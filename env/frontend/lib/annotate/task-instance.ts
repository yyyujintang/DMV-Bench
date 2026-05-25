/**
 * Reader for the Python-generated TaskInstance JSON files synced into
 * `public/tasks/validated/<sub-task>/<task_id>.json` by
 * `scripts/sync-task-pool.mjs`.
 *
 * The W4 `parseTaskSpec()` in `lib/session.ts` accepts a permissive shape
 * meant for the agent-eval surface. Annotation needs more: ground_truth +
 * accepted_alternatives for scoring, and the Python schema's `turns[].role`
 * + `content` so the AnnotateOverlay can render chat bubbles and decide
 * when to show "Continue" vs "Submit answer".
 *
 * This module owns the richer reader. It also adapts a TaskInstance to the
 * W4 TaskSpec via `toTaskSpec()` so we can hand the result to
 * `parseTaskSpec()` and stay on a single session pipeline.
 */
import { promises as fs } from "node:fs";
import path from "node:path";
import type { TaskSpec } from "@/lib/session";

export type SubTask = "NC" | "SA" | "IC" | "VL" | "PD";

export const SUB_TASKS: readonly SubTask[] = ["NC", "SA", "IC", "VL", "PD"] as const;

export type TaskInstanceTurn = {
  turn_index: number;
  role: "system" | "user" | "agent";
  mode: "encoding" | "recall";
  content: string | null;
  is_rejection: boolean;
  references_variant: string | null;
  expected_url: string | null;
  cue_injections: Array<{ page_id: string; cue_key: string; position_override?: string | null }>;
  recall_turn_metadata: { anchor_invisibility_targets: string[]; expected_memory_usage: string } | null;
};

export type TaskInstance = {
  task_id: string;
  sub_task: SubTask;
  grain_tier: 1 | 2 | 3;
  category_ids: string[];
  variants_used: string[];
  turns: TaskInstanceTurn[];
  ncp_metadata: {
    anchor_variant_ids: string[];
    recall_turn_indices: number[];
    memory_gated_branch_turn: number;
    cross_turn_predicates: string[];
  };
  ground_truth: {
    final_action: string;
    target_url: string | null;
    target_variant_id: string | null;
    accepted_alternatives: string[];
  };
  success_criteria: {
    type: "url_match" | "variant_match" | "predicate_match";
    evaluator_fn: string;
    tolerance: number | null;
  };
};

const SUB_TASK_SET = new Set<string>(SUB_TASKS);

export function isSubTask(s: string): s is SubTask {
  return SUB_TASK_SET.has(s);
}

const POOL_ROOT = path.resolve(process.cwd(), "public", "tasks", "validated");

export function poolPathFor(sub: SubTask, taskId: string): string {
  return path.join(POOL_ROOT, sub, `${taskId}.json`);
}

export async function loadTaskInstance(sub: SubTask, taskId: string): Promise<TaskInstance> {
  const raw = await fs.readFile(poolPathFor(sub, taskId), "utf8");
  const parsed = JSON.parse(raw) as TaskInstance;
  if (parsed.sub_task !== sub) {
    throw new Error(`task ${taskId} has sub_task=${parsed.sub_task}, expected ${sub}`);
  }
  return parsed;
}

/** Convert a TaskInstance to the W4 TaskSpec shape consumed by /api/session. */
export function toTaskSpec(t: TaskInstance): TaskSpec {
  return {
    taskId: t.task_id,
    turns: t.turns.map((turn) => ({
      turnIndex: turn.turn_index,
      mode: turn.mode,
      anchorVariantIds: turn.mode === "recall" ? t.ncp_metadata.anchor_variant_ids : [],
      goal: turn.content ?? "",
      cueInjections: turn.cue_injections.map((c) => ({
        pageId: c.page_id,
        cueKey: c.cue_key,
        ...(c.position_override ? { positionOverride: c.position_override } : {}),
      })),
    })),
  };
}

/**
 * Where should the overlay direct the worker for `turnIndex`?
 *  - agent turn with `expected_url`: that URL
 *  - user turn / system turn: stay on current page (the previous agent turn's URL)
 *  - first turn: redirect to the next agent turn's expected_url, or "/" as fallback
 */
export function urlForTurn(t: TaskInstance, turnIndex: number): string | null {
  const turn = t.turns.find((x) => x.turn_index === turnIndex);
  if (turn?.expected_url) return turn.expected_url;
  return null;
}

/**
 * The first non-null expected_url in the turn list. Used to redirect the
 * worker into the shop after `/api/annotate/attempt/start` creates the
 * session.
 */
export function firstExpectedUrl(t: TaskInstance): string {
  for (const turn of t.turns) {
    if (turn.expected_url) return turn.expected_url;
  }
  return "/";
}
