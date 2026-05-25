/**
 * Shop session library — Phase W4 of proposal_website.md.
 *
 * Server-side session state replaces URL-param mode resolution as the
 * primary source of `mode` + `anchorVariantIds`. URL params remain as a
 * fallback (for W1/W2 testing surfaces).
 *
 * `TaskSpec` is intentionally permissive in W4: the full task content
 * schema is owned by `proposal_tasks.md` (TBD). What W4 cares about per
 * turn is: which mode the request is in, which variants are anchors at
 * recall time, and which peripheral cues should be injected on which
 * pages.
 */

import { prisma } from "@/lib/prisma";

export const SHOP_SESSION_COOKIE = "dmv_shop_session_id";

export type SessionMode = "encoding" | "recall";

export type TaskSpecTurn = {
  turnIndex: number;
  mode: SessionMode;
  anchorVariantIds: string[];        // empty in encoding
  goal?: string;                      // human description, for debug
  cueInjections?: Array<{
    pageId: string;                   // arbitrary id within a task, often a variant urlHash or a category slug
    cueKey: string;
    positionOverride?: string;
  }>;
};

export type TaskSpec = {
  taskId: string;
  turns: TaskSpecTurn[];
};

export type ResolvedSessionMode = {
  source: "session" | "url" | "default";
  mode: SessionMode;
  anchorVariantIds: ReadonlySet<string>;
  sessionId: string | null;
  taskId: string | null;
  currentTurn: number | null;
};

const EMPTY_ANCHORS: ReadonlySet<string> = new Set();

export function decodeAnchorList(raw: string | null | undefined): ReadonlySet<string> {
  if (!raw) return EMPTY_ANCHORS;
  return new Set(
    raw.split(",").map((s) => s.trim()).filter((s) => s.length > 0),
  );
}

export function encodeAnchorList(set: ReadonlySet<string>): string {
  return Array.from(set).join(",");
}

/**
 * Look up a session row by cookie value. Returns null if cookie missing
 * or session is unknown / ended.
 */
export async function lookupSession(sessionId: string | null) {
  if (!sessionId) return null;
  const s = await prisma.shopSession.findUnique({
    where: { id: sessionId },
    select: {
      id: true,
      taskId: true,
      currentTurn: true,
      mode: true,
      anchorVariantIds: true,
      endedAt: true,
    },
  });
  if (!s || s.endedAt) return null;
  return s;
}

/**
 * Validate a TaskSpec JSON blob. Throws with a descriptive message on
 * malformed input; returns the typed value on success.
 */
export function parseTaskSpec(raw: unknown): TaskSpec {
  if (!raw || typeof raw !== "object") {
    throw new Error("taskSpec must be an object");
  }
  const r = raw as Record<string, unknown>;
  if (typeof r.taskId !== "string" || r.taskId.length === 0) {
    throw new Error("taskSpec.taskId required");
  }
  if (!Array.isArray(r.turns) || r.turns.length === 0) {
    throw new Error("taskSpec.turns must be a non-empty array");
  }
  const turns: TaskSpecTurn[] = r.turns.map((t, i) => {
    if (!t || typeof t !== "object") {
      throw new Error(`turns[${i}] not an object`);
    }
    const tt = t as Record<string, unknown>;
    const idx = typeof tt.turnIndex === "number" ? tt.turnIndex : i;
    const mode = tt.mode === "recall" ? "recall" : "encoding";
    const anchors = Array.isArray(tt.anchorVariantIds)
      ? tt.anchorVariantIds.filter((x): x is string => typeof x === "string")
      : [];
    const goal = typeof tt.goal === "string" ? tt.goal : undefined;
    const cueInjections = Array.isArray(tt.cueInjections)
      ? tt.cueInjections.flatMap((c) => {
          if (!c || typeof c !== "object") return [];
          const cc = c as Record<string, unknown>;
          if (typeof cc.pageId !== "string" || typeof cc.cueKey !== "string") return [];
          return [{
            pageId: cc.pageId,
            cueKey: cc.cueKey,
            positionOverride: typeof cc.positionOverride === "string" ? cc.positionOverride : undefined,
          }];
        })
      : undefined;
    return { turnIndex: idx, mode, anchorVariantIds: anchors, goal, cueInjections };
  });
  return { taskId: r.taskId, turns };
}

/** Read the per-turn directives for the session's current turn. */
export function turnAt(spec: TaskSpec, turnIndex: number): TaskSpecTurn | null {
  return spec.turns.find((t) => t.turnIndex === turnIndex) ?? null;
}

/**
 * Compute the (mode, anchors) snapshot to write onto the session row when
 * the current turn advances. Pure function — easy to unit-test.
 */
export function snapshotForTurn(spec: TaskSpec, turnIndex: number): {
  mode: SessionMode;
  anchorVariantIds: string;
} {
  const t = turnAt(spec, turnIndex);
  if (!t) return { mode: "encoding", anchorVariantIds: "" };
  return {
    mode: t.mode,
    anchorVariantIds: t.anchorVariantIds.join(","),
  };
}
