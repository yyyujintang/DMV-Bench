/**
 * POST /api/annotate/attempt/finish
 *   body: { gaveUp?: boolean, finalPathname?: string }
 *
 *  - Loads the worker's open attempt + the task it points at
 *  - Picks finalUrl: explicit `finalPathname` from the client (the URL
 *    the worker was on when they hit Submit), or the last entry of
 *    navigationLog, or null
 *  - Calls scoreAttempt() to set `correct`
 *  - Sets finishedAt, ends the ShopSession (sets endedAt), clears the
 *    dmv_shop_session_id cookie
 */
import { NextResponse } from "next/server";
import { cookies } from "next/headers";
import { revalidatePath } from "next/cache";
import { prisma } from "@/lib/prisma";
import { getCurrentWorker } from "@/lib/annotate/session";
import {
  isSubTask,
  loadTaskInstance,
} from "@/lib/annotate/task-instance";
import { scoreAttempt } from "@/lib/annotate/scoring";
import { SHOP_SESSION_COOKIE } from "@/lib/session";

type LogEntry = { turnIndex: number; pathname: string; atMs: number };

export async function POST(req: Request) {
  const worker = await getCurrentWorker();
  if (!worker) return NextResponse.json({ error: "no_worker" }, { status: 401 });

  const sessionId = cookies().get(SHOP_SESSION_COOKIE)?.value;
  if (!sessionId) return NextResponse.json({ error: "no_session" }, { status: 403 });

  let body: { gaveUp?: unknown; finalPathname?: unknown };
  try {
    body = await req.json();
  } catch {
    body = {};
  }
  const gaveUp = body.gaveUp === true;
  const explicitFinal =
    typeof body.finalPathname === "string" && body.finalPathname.length > 0
      ? body.finalPathname
      : null;

  const attempt = await prisma.annotationSubTaskAttempt.findFirst({
    where: { workerId: worker.id, shopSessionId: sessionId, finishedAt: null },
    select: {
      id: true,
      subTask: true,
      taskId: true,
      navigationLog: true,
      startedAt: true,
    },
  });
  if (!attempt) return NextResponse.json({ error: "no_open_attempt" }, { status: 404 });

  if (!isSubTask(attempt.subTask)) {
    return NextResponse.json({ error: "corrupt_attempt" }, { status: 500 });
  }

  let task;
  try {
    task = await loadTaskInstance(attempt.subTask, attempt.taskId);
  } catch (e) {
    return NextResponse.json(
      { error: "task_load_failed", detail: e instanceof Error ? e.message : String(e) },
      { status: 500 },
    );
  }

  let log: LogEntry[];
  try {
    log = JSON.parse(attempt.navigationLog);
    if (!Array.isArray(log)) log = [];
  } catch {
    log = [];
  }
  const finalUrl = explicitFinal ?? log[log.length - 1]?.pathname ?? null;
  const score = scoreAttempt(task, { finalUrl, gaveUp });

  const finishedAt = new Date();
  const totalDurationMs = finishedAt.getTime() - attempt.startedAt.getTime();

  await prisma.$transaction([
    prisma.annotationSubTaskAttempt.update({
      where: { id: attempt.id },
      data: {
        finishedAt,
        finalUrl,
        correct: score.correct,
        gaveUp,
        totalDurationMs,
      },
    }),
    prisma.shopSession.update({
      where: { id: sessionId },
      data: { endedAt: finishedAt },
    }),
  ]);

  cookies().set(SHOP_SESSION_COOKIE, "", {
    httpOnly: true,
    sameSite: "lax",
    secure: process.env.NODE_ENV === "production",
    maxAge: 0,
    path: "/",
  });

  revalidatePath("/", "layout");

  return NextResponse.json({
    ok: true,
    next: "/annotate",
    correct: score.correct,
    matchedAlternative: score.matchedAlternative,
    gaveUp,
  });
}
