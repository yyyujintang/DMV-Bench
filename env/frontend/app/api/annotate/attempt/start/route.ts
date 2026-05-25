/**
 * POST /api/annotate/attempt/start
 *   body: { subTask: "NC" | "SA" | "IC" | "VL" }
 *
 *  1. Pick a validated TaskInstance for this worker (unattempted first,
 *     then unsucceeded). Reject 409 if exhausted.
 *  2. Create a ShopSession (mirrors POST /api/session): writes the W4
 *     session row, sets the `dmv_shop_session_id` cookie.
 *  3. Create an AnnotationSubTaskAttempt linking worker ↔ session ↔ task.
 *  4. Return `{ next: "/annotate/play" }` so the client can router.push().
 *
 * If the worker already has an open attempt (their previous attempt
 * never finished), refuse and instruct them to finish that one first —
 * we don't want overlapping sessions on the same worker cookie.
 */
import { NextResponse } from "next/server";
import { cookies } from "next/headers";
import { revalidatePath } from "next/cache";
import { prisma } from "@/lib/prisma";
import { getCurrentWorker } from "@/lib/annotate/session";
import {
  isSubTask,
  loadTaskInstance,
  toTaskSpec,
} from "@/lib/annotate/task-instance";
import { pickTaskForWorker } from "@/lib/annotate/task-pool";
import {
  SHOP_SESSION_COOKIE,
  parseTaskSpec,
  snapshotForTurn,
} from "@/lib/session";

const SESSION_MAX_AGE_SEC = 60 * 60 * 8;

export async function POST(req: Request) {
  const worker = await getCurrentWorker();
  if (!worker) {
    return NextResponse.json({ error: "no_worker" }, { status: 401 });
  }
  if (!worker.consentGiven) {
    return NextResponse.json({ error: "consent_required" }, { status: 403 });
  }

  let body: { subTask?: unknown };
  try {
    body = await req.json();
  } catch {
    return NextResponse.json({ error: "bad_json" }, { status: 400 });
  }
  const sub = body.subTask;
  if (typeof sub !== "string" || !isSubTask(sub)) {
    return NextResponse.json({ error: "invalid_sub_task" }, { status: 400 });
  }

  // Refuse if there's an unfinished attempt — the worker should finish
  // (or give up) before starting another.
  const open = await prisma.annotationSubTaskAttempt.findFirst({
    where: { workerId: worker.id, finishedAt: null },
    select: { id: true, subTask: true },
  });
  if (open) {
    return NextResponse.json(
      { error: "open_attempt_exists", openSubTask: open.subTask },
      { status: 409 },
    );
  }

  const taskId = await pickTaskForWorker(sub, worker.id);
  if (!taskId) {
    return NextResponse.json({ error: "pool_exhausted" }, { status: 409 });
  }

  let taskInstance;
  try {
    taskInstance = await loadTaskInstance(sub, taskId);
  } catch (e) {
    return NextResponse.json(
      { error: "task_load_failed", detail: e instanceof Error ? e.message : String(e) },
      { status: 500 },
    );
  }

  const taskSpec = parseTaskSpec(toTaskSpec(taskInstance));
  const snap = snapshotForTurn(taskSpec, 0);

  // Create ShopSession + Attempt atomically — neither half should land
  // without the other, because the cookie is the bridge.
  const result = await prisma.$transaction(async (tx) => {
    const session = await tx.shopSession.create({
      data: {
        taskId: taskSpec.taskId,
        currentTurn: 0,
        mode: snap.mode,
        anchorVariantIds: snap.anchorVariantIds,
        taskSpec: JSON.stringify(taskSpec),
      },
    });
    const attempt = await tx.annotationSubTaskAttempt.create({
      data: {
        workerId: worker.id,
        subTask: sub,
        taskId: taskInstance.task_id,
        shopSessionId: session.id,
        targetUrl: taskInstance.ground_truth.target_url ?? "",
        acceptedAlternatives: taskInstance.ground_truth.accepted_alternatives.join(","),
      },
      select: { id: true },
    });
    return { sessionId: session.id, attemptId: attempt.id };
  });

  cookies().set(SHOP_SESSION_COOKIE, result.sessionId, {
    httpOnly: true,
    sameSite: "lax",
    secure: process.env.NODE_ENV === "production",
    maxAge: SESSION_MAX_AGE_SEC,
    path: "/",
  });

  // Invalidate the root layout's cached cookie read so the overlay mounts
  // on the very next navigation, and /annotate's progress query refreshes.
  revalidatePath("/", "layout");

  return NextResponse.json({
    ok: true,
    next: "/annotate/play",
    attemptId: result.attemptId,
    sessionId: result.sessionId,
  });
}
