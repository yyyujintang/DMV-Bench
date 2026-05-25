/**
 * Product grid tile.
 *
 * Phase W1 mode awareness: when `anchorVariantIds` is supplied and the
 * variant id appears in it, the card renders an inert placeholder that
 * preserves the grid slot but exposes neither image nor product URL.
 * This is the MR1 enforcement seam — the parent server component reads
 * mode via `getMode()` and passes the anchor set down.
 *
 * The placeholder occupies the same outer shape as the normal tile so
 * the grid layout doesn't reflow when a single card is suppressed.
 */

import Link from "next/link";
import Image from "next/image";

export interface ProductCardData {
  variantId: string;
  urlHash: string;
  title: string;
  altText: string;
  primaryImage: string;
  price: number;
  badge?: string;
}

export function ProductCard({
  data,
  anchorVariantIds,
}: {
  data: ProductCardData;
  anchorVariantIds?: ReadonlySet<string>;
}) {
  // Anchor identity is stable across environments via urlHash; the task
  // pipeline emits urlHashes while client-side stores (wishlist,
  // recently-viewed) often track urlHash too. Match either id so the
  // contract holds whichever flavour the caller's set carries.
  const isAnchor = !!anchorVariantIds && (
    anchorVariantIds.has(data.variantId) || anchorVariantIds.has(data.urlHash)
  );
  if (isAnchor) {
    return (
      <div
        data-testid="product-card-placeholder"
        data-variant-id={data.variantId}
        role="presentation"
        aria-hidden="true"
        className="group block"
      >
        <div className="aspect-square bg-stone-50 border border-dashed border-stone-200 mb-3" />
        <div className="h-4 bg-stone-100 rounded w-2/3" />
        <div className="h-3 bg-stone-100 rounded w-1/4 mt-2" />
      </div>
    );
  }

  return (
    <Link
      href={`/product/${data.urlHash}`}
      className="group block"
    >
      <div className="aspect-square bg-white overflow-hidden mb-3 relative border border-stone-200 group-hover:border-stone-400 transition-colors">
        <Image
          src={data.primaryImage}
          alt={data.altText}
          fill
          sizes="(min-width: 1024px) 25vw, (min-width: 640px) 33vw, 50vw"
          className="object-contain transition-transform duration-500 group-hover:scale-105"
        />
        {data.badge && (
          <span className="absolute top-3 left-3 bg-white text-xs px-2 py-1 text-stone-700 border border-stone-200">
            {data.badge}
          </span>
        )}
      </div>
      <h3 className="text-sm font-medium text-stone-900 line-clamp-1">{data.title}</h3>
      <p className="text-sm text-stone-600 mt-1">${data.price.toFixed(2)}</p>
    </Link>
  );
}
