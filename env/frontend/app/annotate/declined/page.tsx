/**
 * Terminal page shown after a worker declines consent. No data beyond the
 * AnnotationWorker row (status='abandoned') is recorded.
 */

export default function DeclinedPage() {
  return (
    <div className="space-y-4">
      <h1 className="text-2xl font-light tracking-tight">
        Thanks — your session has ended
      </h1>
      <p className="text-stone-700">
        You declined to take part. No responses have been collected. You
        can return to Prolific and select &quot;Return submission&quot; so
        the study is not charged against your approval rate.
      </p>
      <p className="text-sm text-stone-500">
        You may close this tab.
      </p>
    </div>
  );
}
