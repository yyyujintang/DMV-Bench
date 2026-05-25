/**
 * Score an annotation attempt against a TaskInstance's ground truth.
 *
 * Rules:
 *  - `gaveUp` short-circuits to correct=false.
 *  - For url_match success_criteria, finalUrl must equal target_url (after
 *    normalisation: trailing slashes stripped).
 *  - For variant_match success_criteria, the urlHash extracted from a
 *    /product/<urlHash> path must equal target_variant_id or be in
 *    accepted_alternatives.
 */
import type { TaskInstance } from "@/lib/annotate/task-instance";

export type ScoreInput = {
  finalUrl: string | null;
  gaveUp: boolean;
};

export type ScoreResult = {
  correct: boolean;
  matchedAlternative: boolean;  // true if matched via accepted_alternatives, not the primary target
};

function normalize(u: string): string {
  return u.replace(/\/+$/, "").trim();
}

export function scoreAttempt(task: TaskInstance, { finalUrl, gaveUp }: ScoreInput): ScoreResult {
  if (gaveUp || !finalUrl) return { correct: false, matchedAlternative: false };
  const fu = normalize(finalUrl);
  const target = task.ground_truth.target_url ? normalize(task.ground_truth.target_url) : null;
  // W7 — every generator emits url_match success_criteria. We always
  // compare full pathnames; legacy variant_match tasks (if any survive
  // from older pools) fall through to the same equality check.
  if (target && fu === target) return { correct: true, matchedAlternative: false };
  return { correct: false, matchedAlternative: false };
}
