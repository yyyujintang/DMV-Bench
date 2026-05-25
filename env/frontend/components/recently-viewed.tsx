/**
 * Recently Viewed strip — rendered on every shop page below the main
 * content. Reads `localStorage["dmv_recently_viewed"]` (populated by
 * `<TrackView>` on product detail pages) and renders up to 6 thumbnails.
 *
 * NCP enforcement (proposal_website.md §5.1 point 2):
 *   · Reads the underlying history regardless of mode.
 *   · In recall mode, anchor variants are filtered out of the rendered list.
 *   · If the filtered list is empty, the entire strip is suppressed
 *     ("the sidebar is collapsed").
 *
 * Hydration: the strip renders a stable placeholder during SSR to avoid
 * a mismatch warning, then swaps to the real list after mount.
 */

"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import Image from "next/image";
import { useMode } from "@/components/mode-provider";
import type { RecentlyViewedItem } from "@/components/track-view";

const KEY = "dmv_recently_viewed";
const SHOW = 6;

function readHistory(): RecentlyViewedItem[] {
  try {
    const raw = localStorage.getItem(KEY) ?? "[]";
    const list = JSON.parse(raw);
    return Array.isArray(list) ? list : [];
  } catch {
    return [];
  }
}

export function RecentlyViewed() {
  const mode = useMode();
  const [hydrated, setHydrated] = useState(false);
  const [items, setItems] = useState<RecentlyViewedItem[]>([]);

  useEffect(() => {
    setItems(readHistory());
    setHydrated(true);
    const onStorage = () => setItems(readHistory());
    window.addEventListener("storage", onStorage);
    window.addEventListener("focus", onStorage);
    return () => {
      window.removeEventListener("storage", onStorage);
      window.removeEventListener("focus", onStorage);
    };
  }, []);

  if (!hydrated) return null;

  // MR1 filter — anchor variants never render in recall mode. Match
  // by either cuid (legacy) or urlHash (task-pipeline standard).
  const visible = items
    .filter(
      (x) =>
        !mode.anchorVariantIds.has(x.variantId) &&
        !mode.anchorVariantIds.has(x.urlHash),
    )
    .slice(0, SHOW);

  // §5.1 point 2: empty filtered list → entire panel collapsed.
  if (visible.length === 0) return null;

  return (
    <section
      data-testid="recently-viewed"
      data-mode={mode.mode}
      className="border-t border-stone-200 bg-white"
    >
      <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8 py-6">
        <h2 className="text-xs uppercase tracking-widest text-stone-500 mb-3">
          Recently Viewed
        </h2>
        <div className="flex gap-3 overflow-x-auto">
          {visible.map((v) => (
            <Link
              key={v.variantId}
              href={`/product/${v.urlHash}`}
              className="flex-none w-24 group"
              data-testid="recently-viewed-item"
              data-variant-id={v.variantId}
            >
              <div className="aspect-square bg-stone-100 relative overflow-hidden border border-stone-200 group-hover:border-stone-400 transition-colors">
                <Image
                  src={v.primaryImage}
                  alt={v.altText}
                  fill
                  sizes="96px"
                  className="object-contain"
                />
              </div>
              <p className="text-xs text-stone-700 mt-1 line-clamp-1">{v.title}</p>
            </Link>
          ))}
        </div>
      </div>
    </section>
  );
}
