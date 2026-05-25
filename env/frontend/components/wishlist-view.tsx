"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { useMode } from "@/components/mode-provider";

interface WishlistItem {
  variantId: string;
  urlHash: string;
  title: string;
  primaryImage: string;
  altText: string;
  price: number;
}

export function WishlistView() {
  const mode = useMode();
  const [items, setItems] = useState<WishlistItem[] | null>(null);

  useEffect(() => {
    const hashes: string[] = JSON.parse(localStorage.getItem("dmv_wishlist") ?? "[]");
    if (hashes.length === 0) {
      setItems([]);
      return;
    }
    fetch(`/api/variants?ids=${hashes.join(",")}`)
      .then((r) => r.json())
      .then((d) => setItems(d.variants ?? []))
      .catch(() => setItems([]));
  }, []);

  if (items === null) {
    return <p className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8 py-12 text-stone-500">Loading…</p>;
  }
  if (items.length === 0) {
    return (
      <section className="mx-auto max-w-3xl px-4 sm:px-6 lg:px-8 py-16 text-center">
        <p className="text-stone-500 mb-6">
          No items saved yet. Browse the catalog and use &ldquo;Add to wishlist&rdquo;.
        </p>
        <Link
          href="/category/chairs"
          className="inline-block bg-stone-900 text-white px-8 py-3 text-sm hover:bg-stone-800 transition-colors"
        >
          Browse the catalog
        </Link>
      </section>
    );
  }
  return (
    <section
      className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8 py-8"
      data-testid="wishlist-view"
      data-mode={mode.mode}
    >
      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-x-6 gap-y-10">
        {items.map((it) => {
          const isAnchor =
            mode.anchorVariantIds.has(it.variantId) ||
            mode.anchorVariantIds.has(it.urlHash);
          if (isAnchor) {
            // MR1: anchor's image must NOT render. Keep the L2-safe title
            // (titles are L2-compliant by the leakage contract) and a saved-
            // item placeholder tile. No link out — clicking through would
            // land on the product page where the anchor is fully visible.
            return (
              <div
                key={it.urlHash}
                data-testid="wishlist-anchor-placeholder"
                data-variant-id={it.variantId}
                className="group block"
              >
                <div className="aspect-square bg-stone-50 border border-dashed border-stone-300 mb-3 flex items-center justify-center">
                  <span className="text-xs uppercase tracking-widest text-stone-400">
                    Saved
                  </span>
                </div>
                <h3 className="text-sm font-medium text-stone-900 line-clamp-1">
                  {it.title}
                </h3>
                <p className="text-xs text-stone-400 mt-1">Hidden during this task</p>
              </div>
            );
          }
          return (
            <Link key={it.urlHash} href={`/product/${it.urlHash}`} className="group block">
              <div className="aspect-square bg-white overflow-hidden mb-3 relative border border-stone-200 group-hover:border-stone-400 transition-colors">
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img
                  src={it.primaryImage}
                  alt={it.altText}
                  className="absolute inset-0 w-full h-full object-contain transition-transform duration-500 group-hover:scale-105"
                />
              </div>
              <h3 className="text-sm font-medium text-stone-900 line-clamp-1">{it.title}</h3>
              <p className="text-sm text-stone-600 mt-1">${it.price.toFixed(2)}</p>
            </Link>
          );
        })}
      </div>
    </section>
  );
}
