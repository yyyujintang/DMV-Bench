/**
 * NCP rendering mode — server-side resolver.
 *
 * Phase W1 introduced URL-param-driven mode resolution (`?mode=recall&anchors=…`).
 * Phase W4 layers session-driven resolution on top:
 *
 *   priority: session cookie → URL params → encoding default
 *
 * Middleware (`middleware.ts`) reads the `dmv_shop_session_id` cookie and
 * forwards it as `x-shop-session-id`. `getMode()` looks the session up
 * from Prisma and returns its current mode + anchors. If no session, the
 * URL-param path applies. If neither, encoding is the zero value.
 *
 * The function is `async` because the session lookup hits the DB; existing
 * server-component callers already run in async contexts.
 */

import { headers } from "next/headers";
import { prisma } from "@/lib/prisma";

export type Mode = "encoding" | "recall";

export type ResolvedMode = {
  mode: Mode;
  anchorVariantIds: ReadonlySet<string>;
  taskId: string | null;
  /** Debug-only: which path the resolver took. Optional for ergonomic test fixtures. */
  source?: "session" | "url" | "default";
};

const EMPTY: ReadonlySet<string> = new Set();

export function emptyMode(): ResolvedMode {
  return { mode: "encoding", anchorVariantIds: EMPTY, taskId: null, source: "default" };
}

function parseAnchors(raw: string | null): ReadonlySet<string> {
  if (!raw || raw.length === 0) return EMPTY;
  return new Set(raw.split(",").map((s) => s.trim()).filter((s) => s.length > 0));
}

export async function getMode(): Promise<ResolvedMode> {
  const h = headers();

  // W4 — session-driven (highest priority)
  const sessionId = h.get("x-shop-session-id");
  if (sessionId) {
    try {
      const session = await prisma.shopSession.findUnique({
        where: { id: sessionId },
        select: { taskId: true, mode: true, anchorVariantIds: true, endedAt: true },
      });
      if (session && !session.endedAt) {
        return {
          mode: session.mode === "recall" ? "recall" : "encoding",
          anchorVariantIds: parseAnchors(session.anchorVariantIds),
          taskId: session.taskId,
          source: "session",
        };
      }
    } catch {
      /* DB unreachable mid-request — fall through to URL path */
    }
  }

  // W1 — URL-param fallback (kept so test surfaces still work)
  const rawMode = h.get("x-mode");
  if (rawMode === "recall" || rawMode === "encoding") {
    return {
      mode: rawMode,
      anchorVariantIds: parseAnchors(h.get("x-anchors")),
      taskId: null,
      source: "url",
    };
  }
  const rawAnchors = h.get("x-anchors");
  if (rawAnchors) {
    return {
      mode: "encoding",
      anchorVariantIds: parseAnchors(rawAnchors),
      taskId: null,
      source: "url",
    };
  }

  return emptyMode();
}

export function isAnchor(variantId: string, m: ResolvedMode): boolean {
  return m.anchorVariantIds.has(variantId);
}

/**
 * Strip anchor variants from a list when the resolved mode is `recall`.
 * Encoding mode is a passthrough. Used by search, recommendation
 * carousels, and any other surface where anchors must be invisible.
 *
 * The set may carry either Prisma cuids (`id`) or urlHashes — task-pipeline
 * tasks (proposal_tasks.md) populate it with urlHashes; URL-param tests
 * may use cuids. We accept both.
 */
export function filterAnchorVariants<T extends { id: string; urlHash?: string }>(
  list: readonly T[],
  m: ResolvedMode,
): T[] {
  if (m.mode !== "recall") return [...list];
  return list.filter(
    (v) =>
      !m.anchorVariantIds.has(v.id) &&
      !(v.urlHash !== undefined && m.anchorVariantIds.has(v.urlHash)),
  );
}
