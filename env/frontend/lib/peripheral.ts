/**
 * Peripheral cue resolver — Phase W3 of proposal_website.md §6.
 *
 * Given a request's session + the current page id, returns the cues to
 * inject. Two paths:
 *
 *   · Session-driven (W4 wired): looks up `TaskCueInjection` rows for
 *     `(session.taskId, session.currentTurn, pageId)` and returns the
 *     attached `PeripheralCue` rows.
 *   · URL-param fallback (testing): `?cues=carpet_checker,sticker_red_dot`
 *     in the request URL is parsed from the `x-cue-keys` header (set by
 *     middleware). Used before W4 / for standalone tests.
 *
 * Cues exist only as visual evidence; the agent can recall them only by
 * having encoded them at observation time. L2 contract: no cue key,
 * descriptor, or asset URL ever appears in user-facing text on the page
 * — the CueOverlay component renders the image as an absolutely-positioned
 * `<img>` with no text label.
 */

import { headers } from "next/headers";
import { prisma } from "@/lib/prisma";

export type CueRender = {
  cueKey: string;
  cueType: string;
  assetUrl: string;
  position: string;
  salience: number;
};

const CUE_KEYS_RE = /^[A-Za-z0-9_,-]{1,400}$/;

export async function resolveCuesForPage(pageId: string): Promise<CueRender[]> {
  const h = headers();
  const sessionId = h.get("x-shop-session-id");

  // Session-driven
  if (sessionId) {
    try {
      const session = await prisma.shopSession.findUnique({
        where: { id: sessionId },
        select: { taskId: true, currentTurn: true, endedAt: true },
      });
      if (session && !session.endedAt) {
        const injections = await prisma.taskCueInjection.findMany({
          where: {
            taskId: session.taskId,
            turnIndex: session.currentTurn,
            pageId,
          },
          include: { cue: true },
        });
        if (injections.length > 0) {
          return injections.map((i) => ({
            cueKey: i.cue.cueKey,
            cueType: i.cue.cueType,
            assetUrl: i.cue.assetUrl,
            position: i.positionOverride ?? i.cue.defaultPosition,
            salience: i.cue.salienceScore,
          }));
        }
      }
    } catch {
      /* DB hiccup — fall through to URL fallback */
    }
  }

  // URL-param fallback (`?cues=k1,k2` → middleware sets x-cue-keys)
  const rawKeys = h.get("x-cue-keys");
  if (!rawKeys || !CUE_KEYS_RE.test(rawKeys)) return [];
  const keys = rawKeys
    .split(",")
    .map((s) => s.trim())
    .filter((s) => s.length > 0);
  if (keys.length === 0) return [];
  const cues = await prisma.peripheralCue.findMany({
    where: { cueKey: { in: keys } },
  });
  return cues.map((c) => ({
    cueKey: c.cueKey,
    cueType: c.cueType,
    assetUrl: c.assetUrl,
    position: c.defaultPosition,
    salience: c.salienceScore,
  }));
}
