/**
 * POST /api/session
 *   body: { taskSpec: TaskSpec }
 *   → 200 { sessionId, currentTurn: 0, mode, anchorVariantIds: string[] }
 *   Sets dmv_shop_session_id cookie.
 *
 * GET /api/session
 *   → 200 { sessionId, taskId, currentTurn, mode, anchorVariantIds, ... } | 404
 *   Reads the cookie; returns the current session state.
 */

import { NextResponse } from "next/server";
import { cookies } from "next/headers";
import { prisma } from "@/lib/prisma";
import {
  SHOP_SESSION_COOKIE,
  decodeAnchorList,
  parseTaskSpec,
  snapshotForTurn,
} from "@/lib/session";

const SESSION_MAX_AGE_SEC = 60 * 60 * 8;

export async function POST(req: Request) {
  let body: unknown;
  try {
    body = await req.json();
  } catch {
    return NextResponse.json({ error: "bad_json" }, { status: 400 });
  }
  const taskSpecRaw = (body as { taskSpec?: unknown })?.taskSpec;
  let taskSpec;
  try {
    taskSpec = parseTaskSpec(taskSpecRaw);
  } catch (e) {
    return NextResponse.json(
      { error: "invalid_task_spec", detail: e instanceof Error ? e.message : String(e) },
      { status: 400 },
    );
  }
  const snap = snapshotForTurn(taskSpec, 0);
  const session = await prisma.shopSession.create({
    data: {
      taskId: taskSpec.taskId,
      currentTurn: 0,
      mode: snap.mode,
      anchorVariantIds: snap.anchorVariantIds,
      taskSpec: JSON.stringify(taskSpec),
    },
  });
  cookies().set(SHOP_SESSION_COOKIE, session.id, {
    httpOnly: true,
    sameSite: "lax",
    secure: process.env.NODE_ENV === "production",
    maxAge: SESSION_MAX_AGE_SEC,
    path: "/",
  });
  return NextResponse.json({
    sessionId: session.id,
    taskId: session.taskId,
    currentTurn: session.currentTurn,
    mode: session.mode,
    anchorVariantIds: Array.from(decodeAnchorList(session.anchorVariantIds)),
  });
}

export async function GET() {
  const id = cookies().get(SHOP_SESSION_COOKIE)?.value;
  if (!id) {
    return NextResponse.json({ error: "no_session" }, { status: 404 });
  }
  const s = await prisma.shopSession.findUnique({
    where: { id },
    select: {
      id: true, taskId: true, currentTurn: true, mode: true,
      anchorVariantIds: true, viewHistory: true, rejections: true,
      wishlist: true, endedAt: true,
    },
  });
  if (!s) {
    return NextResponse.json({ error: "not_found" }, { status: 404 });
  }
  return NextResponse.json({
    sessionId: s.id,
    taskId: s.taskId,
    currentTurn: s.currentTurn,
    mode: s.mode,
    anchorVariantIds: Array.from(decodeAnchorList(s.anchorVariantIds)),
    viewHistory: safeJson(s.viewHistory),
    rejections: safeJson(s.rejections),
    wishlist: safeJson(s.wishlist),
    ended: Boolean(s.endedAt),
  });
}

function safeJson(s: string) {
  try { return JSON.parse(s); } catch { return null; }
}
