/**
 * /annotate — entry route + 5-card task selector (v2 IDs).
 *
 * Flow:
 *   1. Prolific link → middleware redirects to /api/annotate/entry,
 *      which upserts the worker row and writes the cookie.
 *   2. Unconsented → ConsentForm. Otherwise → static intro → selector.
 *   3. Selector shows 5 cards (NC/SA/IC/VL/PD). Clicking a card
 *      POSTs /api/annotate/attempt/start, which creates the W4
 *      ShopSession + AnnotationSubTaskAttempt and redirects to
 *      /annotate/play (which routes the worker into the shop).
 */
import { redirect } from "next/navigation";
import { prisma } from "@/lib/prisma";
import { getCurrentWorker } from "@/lib/annotate/session";
import { ConsentForm } from "@/components/annotate/consent-form";
import {
  TaskSelector,
  type SubTaskCode,
  type SubTaskProgress,
} from "@/components/annotate/task-selector";

const SUB_TASKS: SubTaskCode[] = ["NC", "SA", "IC", "VL", "PD"];

export default async function AnnotateEntryPage() {
  const worker = await getCurrentWorker();
  if (!worker) return <MissingProlificParams />;

  if (worker.status === "abandoned") return redirect("/annotate/declined");
  if (worker.status === "rejected") return redirect("/annotate/rejected");
  if (worker.status === "completed") return redirect("/annotate/done");
  if (!worker.consentGiven) return <ConsentForm workerId={worker.id} />;
  if (!worker.practiceCompleted) return redirect("/annotate/intro");

  // Aggregate one row per sub-task — worker either has a finished attempt
  // for it or doesn't. We only flag "attempted" once the attempt is
  // closed (finishedAt != null); an in-flight attempt blocks /attempt/start
  // anyway via the open_attempt_exists guard, so a separate state isn't
  // needed here.
  const finished = await prisma.annotationSubTaskAttempt.findMany({
    where: { workerId: worker.id, finishedAt: { not: null } },
    select: { subTask: true, correct: true },
  });
  const byCode = new Map<string, { attempted: boolean; correct: boolean }>();
  for (const a of finished) {
    const prior = byCode.get(a.subTask);
    // If multiple attempts exist (shouldn't, but defend), surface the
    // most-positive outcome.
    if (!prior || (a.correct && !prior.correct)) {
      byCode.set(a.subTask, { attempted: true, correct: a.correct });
    }
  }
  const progress: SubTaskProgress[] = SUB_TASKS.map((code) => ({
    subTask: code,
    attempted: byCode.get(code)?.attempted ?? false,
    correct: byCode.get(code)?.correct ?? false,
  }));
  return <TaskSelector progress={progress} />;
}

function MissingProlificParams() {
  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-light tracking-tight">
        Access via Prolific only
      </h1>
      <p className="text-stone-600">
        This study is hosted on{" "}
        <a
          href="https://prolific.com"
          className="underline hover:no-underline"
          rel="noopener"
        >
          Prolific
        </a>
        . Please open the study from your Prolific dashboard so we can pay
        you and link your responses to your worker account.
      </p>
      <p className="text-sm text-stone-500">
        If you reached this page by accident, you can safely close the
        browser tab — no data has been recorded.
      </p>
    </div>
  );
}
