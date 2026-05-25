"use client";

/**
 * AnnotateOverlay — the chat-strip pinned to the top of the shop site
 * during an annotation attempt. Mounted from `app/layout.tsx` only when
 * both `dmv_worker_id` and `dmv_shop_session_id` cookies are present,
 * so non-annotator visitors never see it.
 *
 * Responsibilities:
 *  · Reads `GET /api/session` + `GET /api/annotate/attempt/current` to
 *    discover the active task + current turn.
 *  · Renders user/agent/system content as chat bubbles.
 *  · Encoding turn with `expected_url` → "Continue" advances via
 *    `POST /api/session/turn`, then routes to the next expected_url.
 *  · Encoding turn without `expected_url` (content-only) → "Continue"
 *    just advances the turn.
 *  · Recall turn → no "Continue"; worker browses freely. When the path
 *    matches the task's target_url or accepted_alternatives we enable
 *    "Submit answer".
 *  · "Give up" always available on recall turns.
 *  · On every mount / route change, fires `POST /api/annotate/attempt/visit`
 *    so the navigation log captures the worker's path.
 */
import { useEffect, useRef, useState } from "react";
import { useRouter, usePathname } from "next/navigation";

type Turn = {
  turn_index: number;
  role: "system" | "user" | "agent";
  mode: "encoding" | "recall";
  content: string | null;
  is_rejection: boolean;
  references_variant: string | null;
  expected_url: string | null;
  referenced_variant_image?: string | null;
  referenced_variant_title?: string | null;
};

type AttemptInfo = {
  attemptId: string;
  subTask: string;
  taskId: string;
  targetUrl: string;
  acceptedAlternatives: string[];
  totalTurns: number;
  currentTurn: number;
  currentTurnData: Turn;
  isRecall: boolean;
  isAtTarget: boolean;
  canSubmit: boolean;
};

function describeMode(mode: "encoding" | "recall"): string {
  return mode === "recall" ? "Memory check" : "Setup";
}

export function AnnotateOverlay() {
  const router = useRouter();
  const pathname = usePathname();
  const [state, setState] = useState<AttemptInfo | null>(null);
  const [busy, setBusy] = useState<"continue" | "submit" | "give-up" | null>(null);
  const [error, setError] = useState<string | null>(null);
  const lastVisitedPath = useRef<string | null>(null);

  // Pull the current attempt + turn snapshot. Refreshes on every pathname
  // change so the strip stays in sync with the page the worker is on. We
  // pass `pathname` explicitly because Referer is unreliable on fetch.
  useEffect(() => {
    let cancelled = false;
    const url = `/api/annotate/attempt/current?pathname=${encodeURIComponent(pathname ?? "/")}`;
    fetch(url, { credentials: "include" })
      .then(async (r) => {
        if (!r.ok) {
          if (r.status === 404 || r.status === 401 || r.status === 403) {
            if (!cancelled) setState(null);
            return null;
          }
          throw new Error(`fetch_failed_${r.status}`);
        }
        return r.json();
      })
      .then((j) => {
        if (cancelled || !j) return;
        setState(j);
      })
      .catch((e) => {
        if (!cancelled) setError(e instanceof Error ? e.message : String(e));
      });
    return () => {
      cancelled = true;
    };
  }, [pathname]);

  // Log every distinct path the worker visits. Visit endpoint dedupes
  // re-renders within 500ms server-side.
  useEffect(() => {
    if (!state) return;
    if (lastVisitedPath.current === pathname) return;
    lastVisitedPath.current = pathname;
    fetch("/api/annotate/attempt/visit", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ pathname }),
      credentials: "include",
    }).catch(() => { /* best-effort logging */ });
  }, [pathname, state]);

  if (!state) return null;
  const { currentTurnData: t, isRecall, isAtTarget, canSubmit } = state;

  async function onContinue() {
    if (!state) return;
    setBusy("continue");
    setError(null);
    try {
      const r = await fetch("/api/session/turn", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ advance: 1 }),
        credentials: "include",
      });
      const j = await r.json();
      if (!r.ok) throw new Error(j.error ?? "turn_failed");
      // After advancing, re-fetch the attempt snapshot so we know whether
      // the new turn has an expected_url to route to.
      const next = await fetch("/api/annotate/attempt/current", { credentials: "include" });
      const nextJ: AttemptInfo = await next.json();
      setState(nextJ);
      const url = nextJ.currentTurnData.expected_url;
      if (url) router.push(url);
      else router.refresh();
    } catch (e) {
      setError(e instanceof Error ? e.message : "continue_failed");
    } finally {
      setBusy(null);
    }
  }

  async function onSubmit(gaveUp: boolean) {
    setBusy(gaveUp ? "give-up" : "submit");
    setError(null);
    try {
      const r = await fetch("/api/annotate/attempt/finish", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ gaveUp, finalPathname: pathname }),
        credentials: "include",
      });
      const j = await r.json();
      if (!r.ok) throw new Error(j.error ?? "finish_failed");
      router.push(j.next ?? "/annotate");
    } catch (e) {
      setError(e instanceof Error ? e.message : "finish_failed");
      setBusy(null);
    }
  }

  // Bubble text:
  //   - user/system turns with content → show the content
  //   - agent turns: empty — the page itself reflects the navigation,
  //     and the URL would leak the canonical style word
  //     (e.g. `/collection/sofas-industrial`) per v2 §8.5.
  //   - recall turns with no content (rare) → show a generic hint
  const bubbleText = t.content
    ? t.content
    : "";

  // Anchor card display rule: only on "here's one / here are" reveal
  // turns. References-by-reaction ("I like that olive green one") point
  // at a variant the agent already navigated to last turn, so the card
  // would be visually redundant.
  const REVEAL_RE = /\b(here['']s|here are|show me this|show me these)\b/i;
  const showAnchorCard =
    !!t.referenced_variant_image && !!t.content && REVEAL_RE.test(t.content);

  return (
    <div
      className="fixed top-0 inset-x-0 z-50 bg-amber-50 border-b border-amber-300 shadow-sm"
      data-testid="annotate-overlay"
      data-mode={t.mode}
    >
      <div className="mx-auto max-w-7xl px-4 py-3 flex items-start gap-4">
        <div className="text-xs uppercase tracking-widest text-amber-900">
          {state.subTask} · {describeMode(t.mode)} · Turn {state.currentTurn + 1} of {state.totalTurns}
        </div>
        <div className="flex-1 min-w-0 flex items-center gap-3">
          {showAnchorCard && (
            // Anchor card displayed inline (proposal_tasks_v2.md §3.2)
            // ONLY on reveal turns ("here's one I like"). Reaction turns
            // skip this so the overlay doesn't redundantly re-show a
            // variant the agent just navigated to.
            <a
              href={`/product/${t.references_variant}`}
              target="_blank"
              rel="noopener noreferrer"
              data-testid="anchor-card"
              className="flex items-center gap-2 border border-stone-300 bg-white rounded px-2 py-1 hover:border-stone-900 transition-colors"
              title={t.referenced_variant_title ?? "Anchor product"}
            >
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img
                src={t.referenced_variant_image!}
                alt={t.referenced_variant_title ?? "Anchor"}
                className="w-10 h-10 object-contain"
              />
              <span className="text-xs text-stone-700 truncate max-w-[140px]">
                {t.referenced_variant_title ?? "Anchor"}
              </span>
            </a>
          )}
          {bubbleText ? (
            <div
              className="text-sm text-stone-900 flex-1 whitespace-normal break-words leading-snug"
              data-role={t.role}
            >
              <span className="text-stone-500 mr-2">
                {t.role === "user" ? "Customer:" : t.role === "agent" ? "Assistant:" : ""}
              </span>
              {bubbleText}
            </div>
          ) : (
            <div className="text-sm text-stone-500 italic flex-1">
              {isRecall
                ? "Find the answer using your memory of what was shown earlier."
                : t.role === "agent"
                  ? "(Loading the next page — click Continue.)"
                  : ""}
            </div>
          )}
        </div>
        <div className="flex items-center gap-2">
          {!isRecall && (
            <button
              type="button"
              onClick={onContinue}
              disabled={busy !== null}
              className="px-3 py-1.5 text-sm bg-stone-900 text-white hover:bg-stone-800 disabled:opacity-50 rounded"
            >
              {busy === "continue" ? "…" : "Continue"}
            </button>
          )}
          {isRecall && (
            <>
              <button
                type="button"
                onClick={() => onSubmit(false)}
                disabled={!canSubmit || busy !== null}
                className={
                  "px-3 py-1.5 text-sm text-white hover:bg-emerald-800 disabled:opacity-40 rounded " +
                  (isAtTarget ? "bg-emerald-700" : "bg-emerald-600")
                }
                title={
                  canSubmit
                    ? (isAtTarget ? "This page matches the target — submit as correct" : "Submit your final answer (server scores)")
                    : "Navigate to a product or collection page first"
                }
              >
                {busy === "submit" ? "…" : "Submit answer"}
              </button>
              <button
                type="button"
                onClick={() => onSubmit(true)}
                disabled={busy !== null}
                className="px-3 py-1.5 text-sm bg-white border border-stone-300 text-stone-600 hover:bg-stone-50 disabled:opacity-40 rounded"
              >
                {busy === "give-up" ? "…" : "Give up"}
              </button>
            </>
          )}
        </div>
      </div>
      {error && (
        <div className="mx-auto max-w-7xl px-4 pb-2 text-xs text-rose-700">
          {error}
        </div>
      )}
    </div>
  );
}
