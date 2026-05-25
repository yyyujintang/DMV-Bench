import Link from "next/link";
import Image from "next/image";
import { prisma } from "@/lib/prisma";

export default async function HomePage() {
  const categories = await prisma.category.findMany({
    orderBy: { sortOrder: "asc" },
    include: {
      products: {
        take: 1,
        include: { variants: { where: { grainTier: 1 }, take: 1 } },
      },
    },
  });

  // Pull one lifestyle hero image (chair var_b @ tier 1 = warm terracotta — a "warm room" feel).
  const heroVariant = await prisma.productVariant.findFirst({
    where: { product: { category: { slug: "chairs" } }, internalVariantKey: "var_b", grainTier: 1 },
  });

  return (
    <>
      {/* Hero */}
      <section className="bg-stone-100 border-b border-stone-200">
        <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8 py-24 lg:py-32 text-center">
          <p className="text-xs uppercase tracking-widest text-stone-500 mb-4">
            New Collection
          </p>
          <h1 className="text-4xl lg:text-6xl font-light tracking-tight text-stone-900 mb-6">
            Modern goods for<br />everyday rooms.
          </h1>
          <p className="text-base text-stone-600 max-w-xl mx-auto mb-8">
            Studio Living is a small collection of furniture and home objects
            designed for contemporary apartments and reading nooks.
          </p>
          <Link
            href="/category/chairs"
            className="inline-block bg-stone-900 text-white px-8 py-3 text-sm tracking-wide hover:bg-stone-800 transition-colors"
          >
            Shop the Collection
          </Link>
        </div>
      </section>

      {/* Category grid */}
      <section className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8 py-16">
        <h2 className="text-2xl font-light tracking-tight text-stone-900 mb-8">
          Shop by category
        </h2>
        <div className="grid grid-cols-2 md:grid-cols-3 gap-6">
          {categories.map((cat) => {
            const variant = cat.products[0]?.variants[0];
            return (
              <Link
                key={cat.slug}
                href={`/category/${cat.slug}`}
                className="group block"
              >
                <div className="aspect-square bg-stone-100 overflow-hidden mb-3 relative">
                  {variant?.primaryImage && (
                    <Image
                      src={variant.primaryImage}
                      alt={variant.altText}
                      fill
                      sizes="(min-width: 768px) 33vw, 50vw"
                      className="object-contain transition-transform duration-500 group-hover:scale-105"
                    />
                  )}
                </div>
                <h3 className="text-sm font-medium text-stone-900">{cat.display}</h3>
                <p className="text-xs text-stone-500 mt-1 line-clamp-1">{cat.description}</p>
              </Link>
            );
          })}
        </div>
      </section>

      {/* Editorial band */}
      <section className="bg-stone-100 border-t border-stone-200">
        <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8 py-16">
          <div className="grid md:grid-cols-2 gap-12 items-center">
            <div>
              <h2 className="text-3xl font-light tracking-tight text-stone-900 mb-4">
                Designed for the way you live.
              </h2>
              <p className="text-stone-600 mb-6 max-w-md">
                Every Studio Living piece is built for everyday use — durable
                materials, considered silhouettes, and palettes that work in
                rooms big and small.
              </p>
              <Link
                href="/about"
                className="text-sm text-stone-900 border-b border-stone-900 hover:text-stone-600 hover:border-stone-600 transition-colors"
              >
                Read our story
              </Link>
            </div>
            <div className="aspect-[4/3] bg-stone-200 relative overflow-hidden">
              {heroVariant && (
                <Image
                  src={heroVariant.primaryImage}
                  alt={heroVariant.altText}
                  fill
                  sizes="(min-width: 768px) 50vw, 100vw"
                  className="object-contain"
                />
              )}
            </div>
          </div>
        </div>
      </section>
    </>
  );
}
