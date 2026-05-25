/**
 * Task selector — the worker's landing page after consent.
 *
 * Five independent sub-tasks (NC/SA/IC/VL/PD). The worker picks any in
 * any order. Clicking a card POSTs /api/annotate/attempt/start which
 * creates a ShopSession + Attempt and returns the redirect target
 * (/annotate/play). The "Finish" button calls /api/annotate/finish to
 * issue a Prolific completion code once at least one attempt has been
 * recorded.
 */
"use client";

import { useState } from "react";
import { Button } from "@/components/ui/button";

export type SubTaskCode = "NC" | "SA" | "IC" | "VL" | "PD";

export type SubTaskProgress = {
  subTask: SubTaskCode;
  attempted: boolean;
  correct: boolean;
};

const SUB_TASK_META: Record<
  SubTaskCode,
  { title: string; intent: string; estMinutes: string; longHorizon?: boolean }
> = {
  NC: {
    title: "Negative constraint",
    intent:
      "Find a product that resembles one you saw — but not in the style the customer rejected.",
    estMinutes: "3–5 min",
  },
  SA: {
    title: "Style abstraction",
    intent:
      "Three items hint at a shared vibe. Find a fourth item in a new category that fits the vibe.",
    estMinutes: "5–7 min",
  },
  IC: {
    title: "Incidental cue",
    intent:
      "You'll briefly see four pages — one has a small visual mark. Find your way back to it.",
    estMinutes: "4–6 min",
  },
  VL: {
    title: "Visual landmark",
    intent:
      "You'll briefly see four collections. Find your way back to the one matching a visual description.",
    estMinutes: "4–6 min",
  },
  PD: {
    title: "Preference drift",
    intent:
      "The customer changes their mind several times across a long conversation. Pick the product that matches what they currently want.",
    estMinutes: "10–15 min",
    longHorizon: true,
  },
};

const SUB_TASK_ORDER: SubTaskCode[] = ["NC", "SA", "IC", "VL", "PD"];

export function TaskSelector({ progress }: { progress: SubTaskProgress[] }) {
  const [starting, setStarting] = useState<SubTaskCode | null>(null);
  const [finishing, setFinishing] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const progressBy = new Map(progress.map((p) => [p.subTask, p]));
  const anyDone = progress.some((p) => p.attempted);

  async function start(subTask: SubTaskCode) {
    setStarting(subTask);
    setError(null);
    try {
      const res = await fetch("/api/annotate/attempt/start", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ subTask }),
        credentials: "include",
      });
      const j = await res.json();
      if (!res.ok) {
        throw new Error(
          j.error === "open_attempt_exists"
            ? `You already have an open ${j.openSubTask} attempt — please finish it first.`
            : j.error ?? "request_failed",
        );
      }
      // Full reload — the layout's cookie-driven AnnotateOverlay mount
      // needs the fresh shop-session cookie, which a client navigation
      // can miss if the RSC payload was prefetched.
      window.location.href = (j.next as string) ?? "/annotate/play";
    } catch (e) {
      setError(e instanceof Error ? e.message : "request_failed");
      setStarting(null);
    }
  }

  async function finish() {
    setFinishing(true);
    setError(null);
    try {
      const res = await fetch("/api/annotate/finish", { method: "POST" });
      const j = await res.json();
      if (!res.ok) throw new Error(j.error ?? "request_failed");
      window.location.href = j.next as string;
    } catch (e) {
      setError(e instanceof Error ? e.message : "request_failed");
      setFinishing(false);
    }
  }

  return (
    <div className="space-y-8">
      <header className="space-y-2">
        <p className="text-xs uppercase tracking-widest text-stone-500">
          Choose a task
        </p>
        <h1 className="text-2xl font-light tracking-tight">
          Five independent memory tasks
        </h1>
        <p className="text-sm text-stone-600">
          Each task is a short shopping conversation. You'll briefly see some
          products, then answer a question that depends on what you remember.
          Pick any task to start — you can come back for the others later, but
          one task is enough to receive your completion code.
        </p>
      </header>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
        {SUB_TASK_ORDER.map((code) => {
          const p = progressBy.get(code) ?? { subTask: code, attempted: false, correct: false };
          return (
            <TaskCard
              key={code}
              code={code}
              progress={p}
              starting={starting === code}
              disabled={starting !== null || finishing}
              onStart={() => start(code)}
            />
          );
        })}
      </div>

      {error && (
        <p className="text-sm text-destructive">{error}</p>
      )}

      <div className="border-t border-stone-200 pt-6 space-y-3">
        <p className="text-sm text-stone-700">
          Done for today? Click below to receive your completion code. You
          can also leave and come back — your progress is saved.
        </p>
        <Button onClick={finish} disabled={!anyDone || finishing || starting !== null}>
          {finishing
            ? "Finalising…"
            : anyDone
              ? "Finish & get my completion code"
              : "Finish at least one task to unlock"}
        </Button>
      </div>
    </div>
  );
}

function TaskCard({
  code,
  progress,
  starting,
  disabled,
  onStart,
}: {
  code: SubTaskCode;
  progress: SubTaskProgress;
  starting: boolean;
  disabled: boolean;
  onStart: () => void;
}) {
  const meta = SUB_TASK_META[code];
  const isDone = progress.attempted;
  const label = isDone
    ? progress.correct
      ? "Completed ✓"
      : "Completed"
    : starting
      ? "Starting…"
      : "Start this task";
  const statusClass = isDone ? "text-green-700" : "text-stone-500";

  return (
    <article className="rounded-md border border-stone-200 bg-white p-5 space-y-3">
      <div className="flex items-start justify-between gap-4">
        <div className="space-y-1">
          <p className="text-xs uppercase tracking-widest text-stone-500">
            {code} · {meta.estMinutes}
            {meta.longHorizon && (
              <span className="ml-2 inline-block rounded-sm bg-amber-100 text-amber-900 px-1.5 py-0.5 text-[10px] font-medium">
                Set aside 10–15 min
              </span>
            )}
          </p>
          <h2 className="text-lg font-medium text-stone-900">{meta.title}</h2>
        </div>
        <div className={`text-xs uppercase tracking-widest ${statusClass}`}>
          {isDone ? "Done" : "Open"}
        </div>
      </div>
      <p className="text-sm text-stone-600">{meta.intent}</p>
      <button
        type="button"
        onClick={onStart}
        disabled={isDone || disabled}
        data-testid={`start-${code}`}
        className="inline-flex items-center text-sm font-medium text-stone-900 border-b border-stone-900 hover:text-stone-600 hover:border-stone-600 transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
      >
        {label}
      </button>
    </article>
  );
}
