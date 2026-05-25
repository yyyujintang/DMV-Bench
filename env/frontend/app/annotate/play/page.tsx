/**
 * /annotate/play — redirect controller.
 *
 * After /api/annotate/attempt/start creates the ShopSession + Attempt, the
 * client sends the worker here. This page reads the session state and
 * redirects them to the current turn's expected URL (or the first
 * non-null expected URL if the current turn is a system/user-content
 * turn with no URL).
 *
 * The overlay is mounted in app/layout.tsx based on cookies, so once
 * we redirect, the worker sees the shop with the chat strip on top.
 */
import { cookies } from "next/headers";
import { redirect } from "next/navigation";
import { prisma } from "@/lib/prisma";
import { getCurrentWorker } from "@/lib/annotate/session";
import {
  isSubTask,
  loadTaskInstance,
  firstExpectedUrl,
  urlForTurn,
} from "@/lib/annotate/task-instance";
import { SHOP_SESSION_COOKIE } from "@/lib/session";

export const dynamic = "force-dynamic";

export default async function AnnotatePlayPage() {
  const worker = await getCurrentWorker();
  if (!worker) redirect("/annotate");

  const sessionId = cookies().get(SHOP_SESSION_COOKIE)?.value;
  if (!sessionId) redirect("/annotate");

  const attempt = await prisma.annotationSubTaskAttempt.findFirst({
    where: { workerId: worker.id, shopSessionId: sessionId, finishedAt: null },
    select: { subTask: true, taskId: true },
  });
  if (!attempt || !isSubTask(attempt.subTask)) redirect("/annotate");

  const session = await prisma.shopSession.findUnique({
    where: { id: sessionId },
    select: { currentTurn: true, endedAt: true },
  });
  if (!session || session.endedAt) redirect("/annotate");

  const task = await loadTaskInstance(attempt.subTask, attempt.taskId);
  // Prefer the current turn's URL; if it's a content-only turn (no URL),
  // fall back to the first expected_url so the worker lands somewhere.
  const target = urlForTurn(task, session.currentTurn) ?? firstExpectedUrl(task);
  redirect(target);
}
