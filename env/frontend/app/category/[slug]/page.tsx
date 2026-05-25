/**
 * /category/[slug] — category landing.
 *
 * Layout C (Hybrid): the category landing lists its **collections** (3 per
 * category, one per tier — bold / mid / tonal). Each collection is a
 * stylistic family of 4 products. No variant thumbnails are shown here
 * — that would violate MR2 by leaking comparison surfaces. Each
 * collection card shows: collection name, tier flavour, item count, and
 * the price range of its 4 variants.
 */
import { notFound } from "next/navigation";
import Link from "next/link";
import { prisma } from "@/lib/prisma";
import { BreadcrumbTrail } from "@/components/breadcrumb-trail";
import { getMode } from "@/lib/mode";

interface PageProps {
  params: Promise<{ slug: string }>;
}

// Style-descriptor copy, per proposal_tasks_v2.md §8.5 (style-nameability):
// the canonical style word must NOT appear on the catalog page in a way the
// agent can text-grep. We surface a gestalt sentence instead. Style is still
// the URL slug (sofas-modern → unique key) for routing; descriptor copy is
// what the user sees.

export default async function CategoryLandingPage({ params }: PageProps) {
  const { slug } = await params;
  const category = await prisma.category.findUnique({
    where: { slug },
    include: {
      collections: {
        orderBy: { displayOrder: "asc" },
        include: {
          variants: {
            select: { id: true, urlHash: true, price: true },
            orderBy: { price: "asc" },
          },
        },
      },
    },
  });
  if (!category) notFound();

  const mode = await getMode();
  const homeCrumb = { label: "Home", href: "/" };

  return (
    <>
      <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8 pt-6 pb-2">
        <BreadcrumbTrail
          mode={mode.mode}
          crumbs={[homeCrumb, { label: category.display }]}
        />
      </div>

      <header className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8 py-6 border-b border-stone-200">
        <h1 className="text-3xl lg:text-4xl font-light tracking-tight text-stone-900">
          {category.display}
        </h1>
        <p className="text-stone-600 mt-2 max-w-2xl">{category.description}</p>
        <p className="text-xs text-stone-500 mt-3">
          Ten collections in this room. Each collection has ten pieces with
          their own prices — open one to see the lineup.
        </p>
      </header>

      <section className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8 py-10">
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
          {category.collections.map((coll) => {
            const prices = coll.variants.map((v) => v.price);
            const min = prices.length > 0 ? Math.min(...prices) : 0;
            const max = prices.length > 0 ? Math.max(...prices) : 0;
            return (
              <Link
                key={coll.id}
                href={`/collection/${coll.slug}`}
                data-testid="collection-tile"
                data-collection-slug={coll.slug}
                className="block border border-stone-200 hover:border-stone-900 transition-colors p-6 bg-white"
              >
                <h2 className="text-xl font-light tracking-tight text-stone-900">
                  {coll.displayName}
                </h2>
                <p className="text-sm text-stone-600 mt-3">
                  {coll.variants.length} pieces
                </p>
                <p className="text-sm text-stone-900 mt-1 font-medium">
                  ${min.toFixed(2)} – ${max.toFixed(2)}
                </p>
              </Link>
            );
          })}
        </div>
      </section>
    </>
  );
}
