/**
 * /search — keyword match against the L2-safe text fields.
 *
 * NCP enforcement (proposal_website.md §5.1 point 5 + §5.2):
 *   · In recall mode, anchor variants are filtered OUT of the result
 *     set entirely. They neither appear in the visible list nor count
 *     toward the result tally.
 *   · TODO(W4): once the session API exposes ground-truth-aware mode,
 *     the position-shuffle hook below will push the ground-truth variant
 *     past position 3 instead of removing it. The hook is documented;
 *     the W2 cut just removes anchors.
 */

import Link from "next/link";
import { prisma } from "@/lib/prisma";
import { ProductCard } from "@/components/product-card";
import { BreadcrumbTrail } from "@/components/breadcrumb-trail";
import { filterAnchorVariants, getMode } from "@/lib/mode";

interface PageProps {
  searchParams: Promise<{ q?: string }>;
}

export default async function SearchPage({ searchParams }: PageProps) {
  const { q } = await searchParams;
  const query = (q ?? "").trim();
  const mode = await getMode();

  // Trivial keyword match against L2 fields. Real search out of Phase 1 scope.
  const rawVariants = query
    ? await prisma.productVariant.findMany({
        where: {
          OR: [
            { displayName: { contains: query } },
            { title: { contains: query } },
            { product: { is: { baseName: { contains: query } } } },
            { product: { is: { category: { is: { display: { contains: query } } } } } },
          ],
        },
        take: 24,
      })
    : [];

  // MR1 / MR2 filter — anchors never appear in recall-mode search.
  const variants = filterAnchorVariants(rawVariants, mode);
  // TODO(W4): if `mode.groundTruthIds` is non-empty in recall mode,
  // shuffle so each id lands at position >=4 instead of position 0-2.

  return (
    <>
      <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8 pt-6 pb-2">
        <BreadcrumbTrail
          mode={mode.mode}
          crumbs={[{ label: "Home", href: "/" }, { label: "Search" }]}
        />
      </div>
      <header className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8 py-6 border-b border-stone-200">
        <h1 className="text-3xl lg:text-4xl font-light tracking-tight text-stone-900">
          {query ? `Results for ${query}` : "Search"}
        </h1>
        <form action="/search" method="get" className="mt-4 max-w-xl">
          <input
            name="q"
            defaultValue={query}
            placeholder="Search products..."
            className="w-full border border-stone-300 px-4 py-2 text-sm focus:outline-none focus:border-stone-900"
          />
        </form>
      </header>
      <section
        className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8 py-8"
        data-testid="search-results"
        data-mode={mode.mode}
      >
        {query && variants.length === 0 && (
          <p className="text-stone-500">
            No results. Try{" "}
            <Link className="underline" href="/search?q=lamp">lamp</Link>,{" "}
            <Link className="underline" href="/search?q=vase">vase</Link>, or{" "}
            <Link className="underline" href="/search?q=chair">chair</Link>.
          </p>
        )}
        {!query && (
          <p className="text-stone-500">Type a query above to search the catalog.</p>
        )}
        {variants.length > 0 && (
          <div
            className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-6"
            data-testid="search-grid"
          >
            {variants.map((v) => (
              <ProductCard
                key={v.id}
                data={{
                  variantId: v.id,
                  urlHash: v.urlHash,
                  title: v.displayName,
                  altText: v.altText,
                  primaryImage: v.primaryImage,
                  price: v.price,
                }}
                anchorVariantIds={mode.anchorVariantIds}
              />
            ))}
          </div>
        )}
      </section>
    </>
  );
}
