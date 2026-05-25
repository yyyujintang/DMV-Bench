/**
 * /annotate/intro — static "what you'll do" screen.
 *
 * No practice trial anymore. After the NCP-pivot rebuild, the worker's
 * actual tasks are multi-turn navigation sessions on the real shop site
 * (NC/SA/IC/VL), and a single 4AFC practice doesn't preview that.
 * The Continue button just flips the practiceCompleted flag and routes
 * to the selector.
 */
import { redirect } from "next/navigation";
import { getCurrentWorker } from "@/lib/annotate/session";
import { IntroContinue } from "@/components/annotate/intro-continue";

export default async function AnnotateIntroPage() {
  const worker = await getCurrentWorker();
  if (!worker) return redirect("/annotate");
  if (worker.status !== "in_progress") return redirect("/annotate");
  if (!worker.consentGiven) return redirect("/annotate");
  if (worker.practiceCompleted) return redirect("/annotate");

  return (
    <div className="space-y-6">
      <header className="space-y-2">
        <p className="text-xs uppercase tracking-widest text-stone-500">
          Before you start
        </p>
        <h1 className="text-2xl font-light tracking-tight">
          What you'll do
        </h1>
      </header>

      <div className="space-y-4 text-stone-700">
        <p>
          You'll be the shopping assistant on a small furniture store. A
          customer will tell you what they're looking for, walk you past a
          few items, and then ask you to find the right answer based on what
          you remember.
        </p>
        <ul className="list-disc list-inside space-y-2 text-sm">
          <li>
            Each task takes <strong>3 to 7 minutes</strong>. You'll pick which
            one to do from a list of six.
          </li>
          <li>
            A pinned bar at the top of the page tells you whose turn it is
            (customer or you) and what to do next. Click <em>Continue</em> to
            advance through the setup.
          </li>
          <li>
            When the bar switches to <em>Memory check</em>, you'll see the
            store again but some items will be greyed out. That's on purpose
            — you're being asked to find the answer using your memory. Click
            around and submit when you've found it.
          </li>
        </ul>

        <div className="rounded-md border border-amber-300 bg-amber-50 p-4 text-sm">
          <p className="font-medium text-amber-900 mb-1">Please don't:</p>
          <ul className="list-disc list-inside space-y-1 text-amber-900">
            <li>refresh the page or use the browser's back button</li>
            <li>open the store in a second tab</li>
            <li>look up items by description in a search engine</li>
          </ul>
        </div>
      </div>

      <IntroContinue workerId={worker.id} />
    </div>
  );
}
