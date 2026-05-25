/**
 * POST /api/session/turn
 *   body: { advance?: 1 }   |   { setTurn: number }
 *
 * Advances the session's current turn (or jumps to a specific one), then
 * refreshes the cached `mode` + `anchorVariantIds` from the stored task
 * spec. Returns the new snapshot so the caller knows what NCP mode the
 * next request will see.
 *
 * Idempotent on resubmit with the same target turn.
 */

import { NextResponse } from "next/server";
import { cookies } from "next/headers";
import { prisma } from "@/lib/prisma";
import {
  SHOP_SESSION_COOKIE,
  parseTaskSpec,
  snapshotForTurn,
  decodeAnchorList,
} from "@/lib/session";

export async function POST(req: Request) {
  const id = cookies().get(SHOP_SESSION_COOKIE)?.value;
  if (!id) return NextResponse.json({ error: "no_session" }, { status: 403 });

  let body: { advance?: unknown; setTurn?: unknown };
  try {
    body = await req.json();
  } catch {
    return NextResponse.json({ error: "bad_json" }, { status: 400 });
  }

  const s = await prisma.shopSession.findUnique({
    where: { id },
    select: { currentTurn: true, taskSpec: true, endedAt: true },
  });
  if (!s) return NextResponse.json({ error: "not_found" }, { status: 404 });
  if (s.endedAt) return NextResponse.json({ error: "session_ended" }, { status: 409 });
  if (!s.taskSpec) return NextResponse.json({ error: "no_task_spec" }, { status: 409 });

  let spec;
  try {
    spec = parseTaskSpec(JSON.parse(s.taskSpec));
  } catch {
    return NextResponse.json({ error: "stored_task_spec_invalid" }, { status: 500 });
  }

  let targetTurn: number;
  if (typeof body.setTurn === "number" && Number.isInteger(body.setTurn) && body.setTurn >= 0) {
    targetTurn = body.setTurn;
  } else {
    targetTurn = s.currentTurn + 1;
  }
  if (targetTurn >= spec.turns.length) {
    return NextResponse.json(
      { error: "turn_out_of_range", maxTurn: spec.turns.length - 1 },
      { status: 409 },
    );
  }

  const snap = snapshotForTurn(spec, targetTurn);
  const updated = await prisma.shopSession.update({
    where: { id },
    data: {
      currentTurn: targetTurn,
      mode: snap.mode,
      anchorVariantIds: snap.anchorVariantIds,
    },
    select: { id: true, currentTurn: true, mode: true, anchorVariantIds: true },
  });

  return NextResponse.json({
    sessionId: updated.id,
    currentTurn: updated.currentTurn,
    mode: updated.mode,
    anchorVariantIds: Array.from(decodeAnchorList(updated.anchorVariantIds)),
  });
}
