/**
 * GET /api/annotate/attempt/current?pathname=<current-path>
 *
 * Returns the AttemptInfo snapshot the AnnotateOverlay needs to render the
 * pinned chat strip. The `pathname` query param is REQUIRED — the overlay
 * passes it explicitly because Referer is unreliable on cross-route fetch.
 *
 * 404 if the worker has no open attempt — overlay self-hides.
 *
 * Submit-button policy (W7):
 *  - All sub-tasks emit single-URL ground truths (url_match).
 *  - `isAtTarget` is true when the worker's pathname matches target_url
 *    exactly OR (for the URL families NC/SA/IC) the worker is on
 *    any /product/<...> page (we still accept landing there — server-side
 *    scoring is the final judge via scoreAttempt).
 *  - `canSubmit` is true on any /product/* or /collection/* page during
 *    a recall turn, so the worker is never stranded on a 'wrong' page
 *    with no Submit affordance. Server still scores against target_url.
 */
import { NextResponse } from "next/server";
import { cookies } from "next/headers";
import { prisma } from "@/lib/prisma";
import { getCurrentWorker } from "@/lib/annotate/session";
import { isSubTask, loadTaskInstance } from "@/lib/annotate/task-instance";
import { SHOP_SESSION_COOKIE } from "@/lib/session";

export const dynamic = "force-dynamic";

function isSubmittablePath(pathname: string): boolean {
  // Any product detail page or collection landing page counts. Worker
  // can submit; server-side scoring decides correctness.
  return pathname.startsWith("/product/") || pathname.startsWith("/collection/");
}

export async function GET(req: Request) {
  const worker = await getCurrentWorker();
  if (!worker) return NextResponse.json({ error: "no_worker" }, { status: 401 });

  const sessionId = cookies().get(SHOP_SESSION_COOKIE)?.value;
  if (!sessionId) return NextResponse.json({ error: "no_session" }, { status: 403 });

  const attempt = await prisma.annotationSubTaskAttempt.findFirst({
    where: { workerId: worker.id, shopSessionId: sessionId, finishedAt: null },
    select: {
      id: true,
      subTask: true,
      taskId: true,
      targetUrl: true,
      acceptedAlternatives: true,
    },
  });
  if (!attempt || !isSubTask(attempt.subTask)) {
    return NextResponse.json({ error: "no_open_attempt" }, { status: 404 });
  }

  const session = await prisma.shopSession.findUnique({
    where: { id: sessionId },
    select: { currentTurn: true, endedAt: true },
  });
  if (!session || session.endedAt) {
    return NextResponse.json({ error: "session_ended" }, { status: 404 });
  }

  let task;
  try {
    task = await loadTaskInstance(attempt.subTask, attempt.taskId);
  } catch {
    return NextResponse.json({ error: "task_load_failed" }, { status: 500 });
  }
  const t = task.turns.find((x) => x.turn_index === session.currentTurn);
  if (!t) return NextResponse.json({ error: "turn_not_found" }, { status: 500 });

  // Client passes the worker's current pathname as a query param. Falls
  // back to Referer if absent (older clients), then to empty.
  const url = new URL(req.url);
  let pathname = url.searchParams.get("pathname") ?? "";
  if (!pathname) {
    const ref = req.headers.get("referer");
    if (ref) {
      try { pathname = new URL(ref).pathname; } catch { /* ignore */ }
    }
  }

  const isRecall = t.mode === "recall";
  const targetPath = (attempt.targetUrl ?? "").replace(/\/+$/, "");
  const here = pathname.replace(/\/+$/, "");
  const isAtTarget = isRecall && targetPath !== "" && here === targetPath;
  const canSubmit = isRecall && isSubmittablePath(here);

  // If the current turn references a variant ("Here's one I like"), hydrate
  // its image + display name for the overlay's anchor card.
  let refImage: string | null = null;
  let refTitle: string | null = null;
  if (t.references_variant) {
    const refVariant = await prisma.productVariant.findUnique({
      where: { urlHash: t.references_variant },
      select: { primaryImage: true, displayName: true },
    });
    if (refVariant) {
      refImage = refVariant.primaryImage;
      refTitle = refVariant.displayName;
    }
  }

  return NextResponse.json({
    attemptId: attempt.id,
    subTask: attempt.subTask,
    taskId: attempt.taskId,
    targetUrl: attempt.targetUrl,
    acceptedAlternatives: (attempt.acceptedAlternatives ?? "")
      .split(",").filter(Boolean),
    totalTurns: task.turns.length,
    currentTurn: session.currentTurn,
    currentTurnData: {
      turn_index: t.turn_index,
      role: t.role,
      mode: t.mode,
      content: t.content,
      is_rejection: t.is_rejection,
      references_variant: t.references_variant,
      expected_url: t.expected_url,
      referenced_variant_image: refImage,
      referenced_variant_title: refTitle,
    },
    isRecall,
    isAtTarget,
    canSubmit,
  });
}
