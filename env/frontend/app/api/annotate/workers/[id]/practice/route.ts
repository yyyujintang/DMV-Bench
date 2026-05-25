/**
 * POST /api/annotate/workers/[id]/practice
 *   body: {} (no payload — practice answers are not stored)
 *
 * Marks practice as complete. Required gate before /annotate/phase1.
 * Requires consent to already be true; otherwise we reject so a worker
 * can't skip the consent form by hitting this endpoint directly.
 */

import { NextResponse } from "next/server";
import { cookies } from "next/headers";
import { revalidatePath } from "next/cache";
import { prisma } from "@/lib/prisma";
import { WORKER_COOKIE } from "@/lib/annotate/session";

export async function POST(
  _req: Request,
  { params }: { params: { id: string } },
) {
  const cookieWorkerId = cookies().get(WORKER_COOKIE)?.value;
  if (!cookieWorkerId || cookieWorkerId !== params.id) {
    return NextResponse.json({ error: "no_session" }, { status: 403 });
  }
  const worker = await prisma.annotationWorker.findUnique({
    where: { id: params.id },
    select: { consentGiven: true, status: true },
  });
  if (!worker) return NextResponse.json({ error: "not_found" }, { status: 404 });
  if (!worker.consentGiven) {
    return NextResponse.json({ error: "consent_required" }, { status: 403 });
  }
  if (worker.status !== "in_progress") {
    return NextResponse.json({ error: "session_closed" }, { status: 409 });
  }
  await prisma.annotationWorker.update({
    where: { id: params.id },
    data: { practiceCompleted: true },
  });
  // Invalidate /annotate so the next navigation sees practiceCompleted=true
  // and renders the task selector instead of redirecting back to intro.
  revalidatePath("/annotate");
  revalidatePath("/annotate/intro");
  return NextResponse.json({ ok: true, next: "/annotate" });
}
