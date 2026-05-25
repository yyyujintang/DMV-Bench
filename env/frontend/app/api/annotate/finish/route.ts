/**
 * POST /api/annotate/finish
 *
 * Worker-initiated global completion from the task selector. Marks the
 * AnnotationWorker as completed and issues a Prolific completion code,
 * provided the worker has at least one finished sub-task attempt
 * (correct or not). Idempotent — already-completed workers get the same
 * code back.
 */

import { NextResponse } from "next/server";
import { cookies } from "next/headers";
import { randomBytes } from "node:crypto";
import { prisma } from "@/lib/prisma";
import { WORKER_COOKIE } from "@/lib/annotate/session";

function generateCompletionCode(): string {
  const buf = randomBytes(6).toString("base64").replace(/[+/=]/g, "");
  return `DMV-${buf.slice(0, 4).toUpperCase()}-${buf.slice(4, 8).toUpperCase()}`;
}

export async function POST() {
  const cookieWorkerId = cookies().get(WORKER_COOKIE)?.value;
  if (!cookieWorkerId) return NextResponse.json({ error: "no_session" }, { status: 403 });

  const worker = await prisma.annotationWorker.findUnique({
    where: { id: cookieWorkerId },
    select: { id: true, status: true, completionCode: true, consentGiven: true, practiceCompleted: true },
  });
  if (!worker) return NextResponse.json({ error: "not_found" }, { status: 404 });
  if (!worker.consentGiven || !worker.practiceCompleted) {
    return NextResponse.json({ error: "phase0_incomplete" }, { status: 403 });
  }
  if (worker.status === "completed") {
    return NextResponse.json({
      ok: true,
      next: "/annotate/done",
      completionCode: worker.completionCode,
    });
  }
  if (worker.status !== "in_progress") {
    return NextResponse.json({ error: "session_closed" }, { status: 409 });
  }

  const finishedAttempts = await prisma.annotationSubTaskAttempt.count({
    where: { workerId: worker.id, finishedAt: { not: null } },
  });
  if (finishedAttempts === 0) {
    return NextResponse.json(
      { error: "no_task_completed", finishedAttempts: 0 },
      { status: 409 },
    );
  }

  const code = generateCompletionCode();
  await prisma.annotationWorker.update({
    where: { id: worker.id },
    data: {
      status: "completed",
      completionCode: code,
      sessionEndedAt: new Date(),
    },
  });
  return NextResponse.json({
    ok: true,
    next: "/annotate/done",
    completionCode: code,
    finishedAttempts,
  });
}
