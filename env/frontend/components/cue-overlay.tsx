/**
 * Renders a set of peripheral cues as absolutely-positioned overlays.
 *
 * Phase W3 of proposal_website.md §6. The parent server component resolves
 * the cue list via `lib/peripheral.ts:resolveCuesForPage` and passes it
 * down; this component is purely presentational. Cues are rendered with
 * `<img>` (no `next/image` optimisation — these are small transparent
 * PNGs and we want a stable DOM signature that the audit pipeline can
 * spot if a cue ever leaks into recall).
 *
 * Positions:
 *   corner_bl   — bottom-left corner of the page chrome
 *   banner      — top centre, overlapping the header
 *   badge_tl    — top-left of the main product image (caller decides
 *                 which container to anchor against by where they mount
 *                 the overlay)
 *   background  — fills the viewport behind content, low opacity
 *   overlay     — large soft wash across whole page, low opacity
 *
 * Audit contract: cue keys never appear in user-visible text. The data
 * attributes (`data-cue-key`) are kept for the audit pipeline only —
 * recall-mode audit can flag any cue whose key belongs to a previous
 * (anchor) turn.
 */

import type { CueRender } from "@/lib/peripheral";

const POSITION_CLASSES: Record<string, string> = {
  corner_bl:
    "fixed bottom-4 left-4 w-24 h-24 opacity-90 pointer-events-none z-30",
  banner:
    "fixed top-3 left-1/2 -translate-x-1/2 w-14 h-14 opacity-95 pointer-events-none z-30",
  badge_tl:
    "absolute top-3 left-3 w-12 h-12 opacity-95 pointer-events-none z-20",
  background:
    "fixed inset-0 opacity-15 pointer-events-none z-0",
  overlay:
    "fixed inset-0 opacity-25 pointer-events-none z-10 mix-blend-multiply",
};

function classFor(position: string): string {
  return POSITION_CLASSES[position] ?? POSITION_CLASSES.corner_bl;
}

export function CueOverlay({ cues }: { cues: CueRender[] }) {
  if (cues.length === 0) return null;
  return (
    <div data-testid="cue-overlay" aria-hidden="true">
      {cues.map((c) => (
        <img
          key={c.cueKey}
          src={c.assetUrl}
          alt=""
          data-cue-key={c.cueKey}
          data-cue-type={c.cueType}
          data-cue-position={c.position}
          // eslint-disable-next-line @next/next/no-img-element
          className={classFor(c.position) + " select-none"}
          style={c.position === "background" || c.position === "overlay"
            ? { objectFit: "cover", width: "100%", height: "100%" }
            : { objectFit: "contain" }}
        />
      ))}
    </div>
  );
}
