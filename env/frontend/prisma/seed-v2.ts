// DMV-Bench v2 seeder — populates ANY Prisma datasource (Postgres or
// sqlite) from data/vismem_diag_v2/seed_v2.json (the v2.5 source of
// truth, 10 categories / 100 collections / 100 products / 1000 variants).
//
// This is the Postgres-deploy counterpart of scripts/v2_reseed_prisma.py,
// which writes directly to the local sqlite dev.db. That Python script
// CANNOT target Postgres (it uses the `sqlite3` module). This seeder uses
// the Prisma client, so it follows whatever `DATABASE_URL` points at.
//
// Usage (Vercel/Supabase deploy — run once after `prisma migrate deploy`):
//   export DATABASE_URL='postgresql://...:5432/postgres'   # DIRECT url
//   npm run prisma:postgres        # flip schema.prisma → postgresql
//   npx prisma generate --schema prisma/schema.prisma
//   node --import tsx prisma/seed-v2.ts
//   npm run prisma:sqlite          # flip back so local dev keeps working
//   npx prisma generate --schema prisma/schema.prisma
//
// The seed_v2.json IDs are deterministic (cat_/col_/prod_/var_ prefixes),
// so re-running this is idempotent in effect: it wipes inventory rows
// first, then re-inserts the same rows.

import { PrismaClient } from "@prisma/client";
import { readFileSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = dirname(fileURLToPath(import.meta.url));
const prisma = new PrismaClient();

// frontend/prisma → repo root data/vismem_diag_v2/seed_v2.json
const defaultSeed = join(
  __dirname,
  "..",
  "..",
  "..",
  "data",
  "vismem_diag_v2",
  "seed_v2.json",
);
const seedPath = process.env.SEED_V2_FILE ?? defaultSeed;

type V2Category = {
  id: string;
  slug: string;
  display: string;
  description: string;
  sortOrder: number;
};
type V2Collection = {
  id: string;
  slug: string;
  categoryId: string;
  tier: number;
  styleSlug: string;
  displayName: string;
  displayOrder: number;
};
type V2Product = {
  id: string;
  categoryId: string;
  baseName: string;
  baseDesc: string;
  longDesc: string;
  material: string;
  price: number;
};
type V2Variant = {
  id: string;
  productId: string;
  collectionId: string;
  grainTier: number;
  internalVariantKey: string;
  styleSlug: string;
  colorName: string;
  incidentalDetails: string;
  displayName: string;
  price: number;
  urlHash: string;
  primaryImage: string;
  galleryImages: string;
  title: string;
  altText: string;
};
type V2Seed = {
  version: string;
  categories: V2Category[];
  collections: V2Collection[];
  products: V2Product[];
  variants: V2Variant[];
};

async function main() {
  console.log(`[seed-v2] reading ${seedPath}`);
  const data = JSON.parse(readFileSync(seedPath, "utf8")) as V2Seed;
  console.log(
    `[seed-v2] payload ${data.version}: ` +
      `cats=${data.categories.length} collections=${data.collections.length} ` +
      `products=${data.products.length} variants=${data.variants.length}`,
  );

  // Wipe inventory + dependent rows. Order matters (children first).
  await prisma.taskCueInjection.deleteMany();
  await prisma.peripheralCue.deleteMany();
  await prisma.encodingEvent.deleteMany();
  await prisma.variantSubPage.deleteMany();
  await prisma.categorySubPage.deleteMany();
  await prisma.review.deleteMany();
  await prisma.productVariant.deleteMany();
  await prisma.product.deleteMany();
  await prisma.collection.deleteMany();
  await prisma.category.deleteMany();

  for (const c of data.categories) {
    await prisma.category.create({
      data: {
        id: c.id,
        slug: c.slug,
        display: c.display,
        description: c.description,
        sortOrder: c.sortOrder ?? 0,
      },
    });
  }
  for (const c of data.collections) {
    await prisma.collection.create({
      data: {
        id: c.id,
        slug: c.slug,
        categoryId: c.categoryId,
        tier: c.tier,
        styleSlug: c.styleSlug ?? "",
        tierFlavour: "",
        materialSlug: "",
        materialDescriptor: "",
        displayName: c.displayName,
        displayOrder: c.displayOrder ?? 0,
      },
    });
  }
  for (const p of data.products) {
    await prisma.product.create({
      data: {
        id: p.id,
        categoryId: p.categoryId,
        baseName: p.baseName,
        baseDesc: p.baseDesc,
        longDesc: p.longDesc,
        material: p.material,
        price: p.price,
      },
    });
  }
  // Variants are inserted in chunks to keep the Postgres transaction small.
  const CHUNK = 100;
  for (let i = 0; i < data.variants.length; i += CHUNK) {
    const slice = data.variants.slice(i, i + CHUNK);
    await prisma.$transaction(
      slice.map((v) =>
        prisma.productVariant.create({
          data: {
            id: v.id,
            productId: v.productId,
            collectionId: v.collectionId,
            grainTier: v.grainTier,
            internalVariantKey: v.internalVariantKey,
            styleSlug: v.styleSlug ?? "modern",
            colorName: v.colorName ?? "",
            incidentalDetails: v.incidentalDetails ?? "[]",
            displayName: v.displayName ?? "",
            price: v.price ?? 0,
            urlHash: v.urlHash,
            primaryImage: v.primaryImage,
            galleryImages: v.galleryImages,
            title: v.title,
            altText: v.altText,
          },
        }),
      ),
    );
  }

  const counts = {
    categories: await prisma.category.count(),
    collections: await prisma.collection.count(),
    products: await prisma.product.count(),
    variants: await prisma.productVariant.count(),
  };
  console.log("[seed-v2] done:", counts);
}

main()
  .catch((e) => {
    console.error(e);
    process.exit(1);
  })
  .finally(() => prisma.$disconnect());
