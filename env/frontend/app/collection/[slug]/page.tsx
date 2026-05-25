/**
 * /collection/[slug] — collection detail.
 *
 * Layout C: the collection page lists its 4 stylistic-sibling products as
 * cards. Each card carries the product's full display name + price.
 * Anchor variants are MR1-hidden via the standard ProductCard placeholder
 * behaviour, so during recall mode the agent (or worker) sees N − 1
 * regular cards plus a placeholder slot — and must use memory to pick.
 *
 * This is the **memory-gated branch** for NC/SA: at recall time the
 * agent lands here and chooses one product card by visually inferring
 * which one satisfies the user's positive + negative + price constraint.
 */
import { notFound } from "next/navigation";
import { prisma } from "@/lib/prisma";
import { ProductCard } from "@/components/product-card";
import { BreadcrumbTrail } from "@/components/breadcrumb-trail";
import { getMode } from "@/lib/mode";

interface PageProps {
  params: Promise<{ slug: string }>;
}

// Style-descriptor copy, per proposal_tasks_v2.md §8.5. No canonical style
// word in the page header. The URL slug retains the style name for routing
// stability; the product detail page's spec sheet still shows the style as
// data (which is fine — the spec calls out the catalog-page surface).

export default async function CollectionPage({ params }: PageProps) {
  const { slug } = await params;
  const collection = await prisma.collection.findUnique({
    where: { slug },
    include: {
      category: true,
      variants: {
        orderBy: { price: "asc" },
      },
    },
  });
  if (!collection) notFound();

  const mode = await getMode();
  const crumbs = [
    { label: "Home", href: "/" },
    { label: collection.category.display, href: `/category/${collection.category.slug}` },
    { label: collection.displayName },
  ];

  return (
    <>
      <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8 pt-6 pb-2">
        <BreadcrumbTrail mode={mode.mode} crumbs={crumbs} />
      </div>

      <header className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8 py-6 border-b border-stone-200">
        <p className="text-xs uppercase tracking-widest text-stone-500">
          {collection.category.display}
        </p>
        <h1 className="text-3xl lg:text-4xl font-light tracking-tight text-stone-900 mt-2">
          {collection.displayName}
        </h1>
        <p className="text-stone-600 mt-2 max-w-2xl">
          Ten pieces in this collection. Browse the lineup below.
        </p>
      </header>

      <section className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8 py-10">
        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-x-6 gap-y-10">
          {collection.variants.map((v) => (
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
      </section>
    </>
  );
}
