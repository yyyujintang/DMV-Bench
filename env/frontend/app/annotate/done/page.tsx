/**
 * Terminal page shown after a worker clicks "Finish & get my completion
 * code" with ≥ 1 sub-task attempted.
 *
 * Displays the Prolific completion code stored on the worker row.
 * Returning here after status='completed' just re-renders the code,
 * so a worker who closes the tab mid-payout can come back to fetch it.
 */

import { redirect } from "next/navigation";
import { cookies } from "next/headers";
import { prisma } from "@/lib/prisma";
import { WORKER_COOKIE } from "@/lib/annotate/session";

export default async function DonePage() {
  const id = cookies().get(WORKER_COOKIE)?.value;
  if (!id) return redirect("/annotate");
  const w = await prisma.annotationWorker.findUnique({
    where: { id },
    select: { status: true, completionCode: true },
  });
  if (!w) return redirect("/annotate");
  if (w.status === "abandoned") return redirect("/annotate/declined");
  if (w.status === "rejected") return redirect("/annotate/rejected");
  if (w.status !== "completed" || !w.completionCode) {
    // Worker landed here without finishing — bounce back into the funnel.
    return redirect("/annotate");
  }

  // How many of the 5 sub-tasks did this worker actually finish?
  const finishedCount = await prisma.annotationSubTaskAttempt.count({
    where: { workerId: id, finishedAt: { not: null } },
  });
  const total = 5;
  const doneSummary =
    finishedCount === total
      ? `You completed all ${total} memory tasks.`
      : `You completed ${finishedCount} of ${total} memory tasks.`;

  return (
    <div className="space-y-6">
      <header className="space-y-2">
        <h1 className="text-2xl font-light tracking-tight">
          Thank you — your session is complete
        </h1>
        <p className="text-stone-700">
          {doneSummary} Your responses have been saved.
        </p>
      </header>

      <div className="rounded-md border border-stone-300 bg-white p-5 space-y-2">
        <p className="text-xs uppercase tracking-widest text-stone-500">
          Your Prolific completion code
        </p>
        <p className="font-mono text-2xl tracking-wider text-stone-900 select-all">
          {w.completionCode}
        </p>
        <p className="text-xs text-stone-500">
          Copy this code and paste it on the Prolific completion page to
          confirm your submission. You may also screenshot it for your
          records.
        </p>
      </div>

      <p className="text-sm text-stone-500">
        You may now close this tab.
      </p>
    </div>
  );
}
