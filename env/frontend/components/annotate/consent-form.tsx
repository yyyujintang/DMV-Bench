/**
 * Consent form. Worker must tick the box AND click "I consent" to advance.
 * "I decline" closes the session (status='abandoned') and routes to
 * /annotate/declined, which surfaces the Prolific return link.
 *
 * The consent copy below is a placeholder — replace with the IRB-approved
 * text before the pilot run (Section 10 of annotation_proposal.md).
 */

"use client";

import { useState } from "react";
import { Button } from "@/components/ui/button";

export function ConsentForm({ workerId }: { workerId: string }) {
  const [agreed, setAgreed] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function submit(consent: boolean) {
    setSubmitting(true);
    setError(null);
    try {
      const res = await fetch(`/api/annotate/workers/${workerId}/consent`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ consent }),
      });
      const j = await res.json();
      if (!res.ok) throw new Error(j.error ?? "request_failed");
      // Full reload — avoids Next 14's RSC navigation cache returning the
      // pre-mutation /annotate render. revalidatePath helps for cross-tab
      // but the same router instance can still serve stale.
      window.location.href = j.next as string;
    } catch (e) {
      setError(e instanceof Error ? e.message : "request_failed");
      setSubmitting(false);
    }
  }

  return (
    <div className="space-y-6">
      <header className="space-y-2">
        <h1 className="text-2xl font-light tracking-tight">
          Welcome — please read before starting
        </h1>
        <p className="text-sm text-stone-500">
          Estimated time: ~5–30 minutes (depends on which tasks you pick)
          {" · "}Payment: per Prolific listing
        </p>
      </header>

      <section className="space-y-3 text-sm leading-relaxed text-stone-700">
        <p>
          You are being invited to take part in a research study run by
          the DMV-Bench team. The goal is to measure how well people
          remember and re-find items on a small shopping site under
          different memory conditions.
        </p>
        <p>
          After a brief overview, you&apos;ll see a menu of{" "}
          <strong>five independent tasks</strong>. Each is a short shopping
          conversation in which a customer asks you to find a product
          using your memory of items you saw earlier. You can do any
          number of them, in any order — finishing one is enough to
          receive your completion code. The tasks are:
        </p>
        <ul className="list-disc pl-5 space-y-1 text-stone-700">
          <li>
            <strong>NC — Negative constraint.</strong> Find a product that
            resembles one you saw, but not in the style the customer
            rejected. ~3–5 min.
          </li>
          <li>
            <strong>SA — Style abstraction.</strong> Three items hint at a
            shared vibe. Find a fourth item in a new category that fits
            the vibe. ~5–7 min.
          </li>
          <li>
            <strong>IC — Incidental cue.</strong> You&apos;ll briefly see
            four pages, one with a small visual mark. Find your way back
            to it. ~4–6 min.
          </li>
          <li>
            <strong>VL — Visual landmark.</strong> You&apos;ll briefly see
            four collections. Find your way back to the one matching a
            visual description. ~4–6 min.
          </li>
          <li>
            <strong>PD — Preference drift.</strong> The customer changes
            their mind several times across a long conversation. Pick the
            product that matches what they currently want. ~10–15 min —
            set aside time before starting.
          </li>
        </ul>
        <p>
          We collect your Prolific ID (hashed before storage), the pages
          you visit during each task, the time you spend on each turn,
          and your final answer. We do not collect names, contact
          details, or other personal information. You can stop between
          tasks; your progress is saved.
        </p>
        <p>
          Aggregated statistics from this study will be released publicly
          alongside the DMV-Bench benchmark. Individual responses are
          released with your Prolific ID replaced by an anonymous code.
        </p>
        <p className="text-stone-500">
          Questions or concerns? Contact the study lead listed on the
          Prolific submission page. This protocol will be reviewed by the
          institutional IRB before the main study launches.
        </p>
      </section>

      <label className="flex items-start gap-3 text-sm cursor-pointer select-none">
        <input
          type="checkbox"
          className="mt-1 size-4 rounded border-stone-300"
          checked={agreed}
          onChange={(e) => setAgreed(e.target.checked)}
          disabled={submitting}
        />
        <span>
          I am 18 years of age or older, I have read the description above,
          and I agree to take part on the terms described.
        </span>
      </label>

      {error && (
        <p className="text-sm text-destructive">
          Something went wrong: {error}. Please retry, or return to Prolific.
        </p>
      )}

      <div className="flex flex-wrap gap-3 pt-2">
        <Button
          onClick={() => submit(true)}
          disabled={!agreed || submitting}
        >
          I consent — continue
        </Button>
        <Button
          variant="outline"
          onClick={() => submit(false)}
          disabled={submitting}
        >
          I decline
        </Button>
      </div>
    </div>
  );
}
