/**
 * Mount inside `/product/[id]/page.tsx`. On mount, prepends the visited
 * variant to `localStorage["dmv_recently_viewed"]` so the RecentlyViewed
 * sidebar can render it on subsequent page loads.
 *
 * No NCP gating happens here — we record the view regardless of mode.
 * The mode-aware filter lives in `<RecentlyViewed>` at render time, so
 * a worker can never *see* an anchor they previously visited even though
 * the history entry exists. (This matches the proposal §5.1 point 2 —
 * "reads the session's view history but filters out anchors before
 * rendering".)
 */

"use client";

import { useEffect } from "react";

const KEY = "dmv_recently_viewed";
const MAX = 12;

export type RecentlyViewedItem = {
  variantId: string;
  urlHash: string;
  title: string;
  altText: string;
  primaryImage: string;
  viewedAt: number;
};

export function TrackView(props: {
  variantId: string;
  urlHash: string;
  title: string;
  altText: string;
  primaryImage: string;
}) {
  useEffect(() => {
    try {
      const raw = localStorage.getItem(KEY) ?? "[]";
      const list: RecentlyViewedItem[] = JSON.parse(raw);
      const filtered = list.filter((x) => x.variantId !== props.variantId);
      const next: RecentlyViewedItem[] = [
        {
          variantId: props.variantId,
          urlHash: props.urlHash,
          title: props.title,
          altText: props.altText,
          primaryImage: props.primaryImage,
          viewedAt: Date.now(),
        },
        ...filtered,
      ].slice(0, MAX);
      localStorage.setItem(KEY, JSON.stringify(next));
    } catch {
      /* swallow — never let a tracker break the page */
    }
  }, [props.variantId, props.urlHash, props.title, props.altText, props.primaryImage]);

  return null;
}
