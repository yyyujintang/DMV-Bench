/**
 * Terminal page after auto-rejection (failed attention checks or too-fast
 * mean RT in Phase 1, or whichever later-phase gate trips). The worker
 * row carries the rejection_reason; we surface a friendly version here.
 */

import { redirect } from "next/navigation";
import { prisma } from "@/lib/prisma";
import { cookies } from "next/headers";
import { WORKER_COOKIE } from "@/lib/annotate/session";

const REASON_COPY: Record<string, string> = {
  failed_attention_check:
    "Several of the attention-check trials weren't answered correctly, which usually indicates the trials weren't being viewed carefully.",
  too_fast:
    "The average response time was below the threshold for considered responses, which usually indicates random or accidental clicking.",
};

export default async function RejectedPage() {
  const id = cookies().get(WORKER_COOKIE)?.value;
  if (!id) return redirect("/annotate");
  const w = await prisma.annotationWorker.findUnique({
    where: { id },
    select: { status: true, rejectionReason: true },
  });
  if (!w || w.status !== "rejected") return redirect("/annotate");

  const reasonCopy =
    (w.rejectionReason && REASON_COPY[w.rejectionReason]) ??
    "Your session didn't meet the quality criteria for this study.";

  return (
    <div className="space-y-4">
      <h1 className="text-2xl font-light tracking-tight">
        Thanks — your session has ended
      </h1>
      <p className="text-stone-700">{reasonCopy}</p>
      <p className="text-stone-700">
        No further phases will be offered for this session. Please return
        to Prolific. Your responses up to this point have been saved for
        our records; partial completion may be paid pro-rata per the
        study&apos;s listing.
      </p>
      <p className="text-sm text-stone-500">You may close this tab.</p>
    </div>
  );
}
