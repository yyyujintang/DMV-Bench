import { notFound } from "next/navigation";
import Link from "next/link";
import { prisma } from "@/lib/prisma";
import { ProductGallery } from "@/components/product-gallery";
import { ProductCard } from "@/components/product-card";
import { AddToWishlist } from "@/components/add-to-wishlist";
import { TrackView } from "@/components/track-view";
import { CueOverlay } from "@/components/cue-overlay";
import { resolveCuesForPage } from "@/lib/peripheral";
import { getMode } from "@/lib/mode";

interface PageProps {
  params: Promise<{ id: string }>;
}

export default async function ProductDetailPage({ params }: PageProps) {
  const { id } = await params;
  const variant = await prisma.productVariant.findUnique({
    where: { urlHash: id },
    include: {
      reviews: { orderBy: { createdAt: "desc" } },
      product: { include: { category: true } },
      collection: true,
    },
  });
  if (!variant) notFound();
  const cues = await resolveCuesForPage(variant.urlHash);
  const mode = await getMode();

  // W7v2 MR1 extension (proposal_tasks_v2.md §3.6): the anchor product's
  // own detail page is also hidden during the action phase. The agent
  // cannot navigate to /product/<anchor> as a shortcut — the anchor
  // exists only in the chat-card transcript from setup.
  if (mode.mode === "recall" && mode.anchorVariantIds.has(variant.urlHash)) {
    return (
      <section
        className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8 py-24 text-center"
        data-testid="product-anchor-hidden"
        data-mode="recall"
      >
        <p className="text-xs uppercase tracking-widest text-stone-500 mb-4">
          Item unavailable during this task
        </p>
        <h1 className="text-2xl font-light tracking-tight text-stone-900 mb-2">
          This product has been removed from view.
        </h1>
        <p className="text-sm text-stone-600 max-w-md mx-auto">
          You saw it earlier in the conversation. Return to the catalog
          and use your memory to find what the customer wants.
        </p>
      </section>
    );
  }

  const { product, collection } = variant;
  const gallery: string[] = JSON.parse(variant.galleryImages || "[]");
  const galleryImages = gallery.length > 0 ? gallery : [variant.primaryImage];

  // "More from this collection" — siblings in the same collection (3
  // products: the other stylistic siblings minus this one).
  const siblings = collection
    ? await prisma.productVariant.findMany({
        where: { collectionId: collection.id, NOT: { id: variant.id } },
        orderBy: { price: "asc" },
      })
    : [];

  return (
    <>
      <CueOverlay cues={cues} />
      <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8 pt-6 pb-2 text-xs text-stone-500">
        <Link href="/" className="hover:text-stone-900">Home</Link>
        <span className="mx-2">/</span>
        <Link href={`/category/${product.category.slug}`} className="hover:text-stone-900">
          {product.category.display}
        </Link>
        {collection && (
          <>
            <span className="mx-2">/</span>
            <Link href={`/collection/${collection.slug}`} className="hover:text-stone-900">
              {collection.displayName}
            </Link>
          </>
        )}
        <span className="mx-2">/</span>
        <span className="text-stone-900">{variant.displayName}</span>
      </div>

      <section className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8 py-8">
        <div className="grid lg:grid-cols-2 gap-12 lg:gap-16">
          <ProductGallery images={galleryImages} alt={variant.altText} />

          <div>
            <p className="text-xs uppercase tracking-widest text-stone-500 mb-2">
              {collection?.displayName ?? product.category.display}
            </p>
            <h1 className="text-3xl font-light tracking-tight text-stone-900 mb-3">
              {variant.displayName}
            </h1>
            <p className="text-2xl text-stone-900 mb-6">${variant.price.toFixed(2)}</p>

            <p className="text-stone-600 leading-relaxed mb-8">{product.longDesc}</p>

            <dl className="grid grid-cols-2 gap-4 py-6 border-y border-stone-200 mb-8 text-sm">
              {/* Style word intentionally omitted on detail page (per user
                  feedback during W7v2 testing): the canonical style names
                  must not be text-greppable anywhere on the catalog
                  surface, otherwise SA / VL tasks degrade to text match. */}
              <div>
                <dt className="text-stone-500 mb-1">Colour</dt>
                <dd className="text-stone-900">{variant.colorName || "—"}</dd>
              </div>
              <div>
                <dt className="text-stone-500 mb-1">Material</dt>
                <dd className="text-stone-900">{product.material}</dd>
              </div>
            </dl>

            <div className="space-y-3">
              <AddToWishlist urlHash={variant.urlHash} title={variant.displayName} />
            </div>
            <TrackView
              variantId={variant.id}
              urlHash={variant.urlHash}
              title={variant.displayName}
              altText={variant.altText}
              primaryImage={variant.primaryImage}
            />
          </div>
        </div>
      </section>

      <section className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8 py-12 border-t border-stone-200">
        <h2 className="text-xl font-light tracking-tight text-stone-900 mb-6">
          Customer Reviews
        </h2>
        <ul className="grid md:grid-cols-3 gap-6">
          {variant.reviews.map((r) => (
            <li key={r.id} className="bg-white border border-stone-200 p-5">
              <div className="text-yellow-600 text-sm mb-2">{"★".repeat(r.rating)}{"☆".repeat(5 - r.rating)}</div>
              <p className="text-sm text-stone-700 mb-3">&ldquo;{r.body}&rdquo;</p>
              <p className="text-xs text-stone-500">— {r.authorName}</p>
            </li>
          ))}
        </ul>
      </section>

      {siblings.length > 0 && (
        <section className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8 py-12 border-t border-stone-200">
          <h2 className="text-xl font-light tracking-tight text-stone-900 mb-6">
            More from this collection
          </h2>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
            {siblings.map((v) => (
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
      )}
    </>
  );
}
