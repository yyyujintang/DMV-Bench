/**
 * POST /api/annotate/attempt/visit
 *   body: { pathname: string }
 *
 * Appends a visit log entry to the worker's open attempt. The overlay
 * client calls this on every page mount during the session so we capture
 * the worker's navigation trail (especially during recall turns where
 * they freely click around).
 *
 * Idempotent on duplicates within a 500 ms window — Next.js double-renders
 * fire two visits otherwise.
 */
import { NextResponse } from "next/server";
import { cookies } from "next/headers";
import { prisma } from "@/lib/prisma";
import { getCurrentWorker } from "@/lib/annotate/session";
import { SHOP_SESSION_COOKIE } from "@/lib/session";

type LogEntry = { turnIndex: number; pathname: string; atMs: number };

const DUP_WINDOW_MS = 500;

export async function POST(req: Request) {
  const worker = await getCurrentWorker();
  if (!worker) return NextResponse.json({ error: "no_worker" }, { status: 401 });

  const sessionId = cookies().get(SHOP_SESSION_COOKIE)?.value;
  if (!sessionId) return NextResponse.json({ error: "no_session" }, { status: 403 });

  let body: { pathname?: unknown };
  try {
    body = await req.json();
  } catch {
    return NextResponse.json({ error: "bad_json" }, { status: 400 });
  }
  if (typeof body.pathname !== "string" || body.pathname.length === 0) {
    return NextResponse.json({ error: "invalid_pathname" }, { status: 400 });
  }
  const pathname = body.pathname;

  const attempt = await prisma.annotationSubTaskAttempt.findFirst({
    where: { workerId: worker.id, shopSessionId: sessionId, finishedAt: null },
    select: { id: true, navigationLog: true },
  });
  if (!attempt) return NextResponse.json({ error: "no_open_attempt" }, { status: 404 });

  const session = await prisma.shopSession.findUnique({
    where: { id: sessionId },
    select: { currentTurn: true },
  });
  const turnIndex = session?.currentTurn ?? 0;

  let log: LogEntry[];
  try {
    log = JSON.parse(attempt.navigationLog);
    if (!Array.isArray(log)) log = [];
  } catch {
    log = [];
  }

  const now = Date.now();
  const last = log[log.length - 1];
  if (last && last.pathname === pathname && now - last.atMs < DUP_WINDOW_MS) {
    return NextResponse.json({ ok: true, deduped: true });
  }
  log.push({ turnIndex, pathname, atMs: now });

  await prisma.annotationSubTaskAttempt.update({
    where: { id: attempt.id },
    data: { navigationLog: JSON.stringify(log) },
  });
  return NextResponse.json({ ok: true, logged: log.length });
}
