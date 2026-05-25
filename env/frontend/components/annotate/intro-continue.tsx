"use client";

import { useState } from "react";
import { Button } from "@/components/ui/button";

export function IntroContinue({ workerId }: { workerId: string }) {
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function onContinue() {
    setBusy(true);
    setError(null);
    try {
      const r = await fetch(`/api/annotate/workers/${workerId}/practice`, {
        method: "POST",
      });
      const j = await r.json();
      if (!r.ok) throw new Error(j.error ?? "request_failed");
      // Full reload — see consent-form.tsx for rationale.
      window.location.href = (j.next as string) ?? "/annotate";
    } catch (e) {
      setError(e instanceof Error ? e.message : "request_failed");
      setBusy(false);
    }
  }

  return (
    <div className="space-y-2">
      <Button onClick={onContinue} disabled={busy}>
        {busy ? "Loading…" : "I understand — show me the tasks"}
      </Button>
      {error && <p className="text-sm text-destructive">{error}</p>}
    </div>
  );
}
