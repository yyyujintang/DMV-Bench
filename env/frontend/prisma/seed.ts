import { PrismaClient } from "@prisma/client";
import { readFileSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = dirname(fileURLToPath(import.meta.url));
const prisma = new PrismaClient();

const seedPath = process.env.SEED_FILE ?? join(__dirname, "..", "..", "scripts", "seed.json");
const namingPath = join(__dirname, "..", "..", "scripts", "pricing_naming.json");

const data = JSON.parse(readFileSync(seedPath, "utf8"));
const naming = JSON.parse(readFileSync(namingPath, "utf8")) as {
  collections: Record<
    string,
    {
      slug: string;
      categorySlug: string;
      tier: number;
      styleSlug: string;
      tierFlavour: string;
      materialSlug: string;
      materialDescriptor: string;
      displayName: string;
      displayOrder: number;
    }
  >;
  variants: Record<
    string,
    {
      displayName: string;
      colorName: string;
      price: number;
      styleSlug: string;
      collectionSlug: string;
      tier: number;
      tierFlavour: string;
      categorySlug: string;
      materialSlug: string;
      materialDescriptor: string;
      incidentalDetails?: string[];
    }
  >;
};

async function main() {
  console.log(`[seed] reading ${seedPath}`);
  console.log(`[seed] reading ${namingPath} (${Object.keys(naming.collections).length} collections, ${Object.keys(naming.variants).length} variants)`);
  // Order matters: dependent rows before parent rows.
  await prisma.taskCueInjection.deleteMany();
  await prisma.peripheralCue.deleteMany();
  await prisma.variantSubPage.deleteMany();
  await prisma.categorySubPage.deleteMany();
  await prisma.review.deleteMany();
  await prisma.productVariant.deleteMany();
  await prisma.product.deleteMany();
  await prisma.collection.deleteMany();
  await prisma.category.deleteMany();

  // Pass 1 — Category + Collection rows up-front so variants can FK into them.
  const collectionIdBySlug = new Map<string, string>();
  for (const cat of data.categories) {
    const dbCat = await prisma.category.create({
      data: {
        slug: cat.slug,
        display: cat.display,
        description: cat.description,
        sortOrder: cat.sortOrder ?? 0,
      },
    });
    // Each category gets the 3 collections owned by it (one per tier).
    for (const coll of Object.values(naming.collections)) {
      if (coll.categorySlug !== cat.slug) continue;
      const dbColl = await prisma.collection.create({
        data: {
          slug: coll.slug,
          categoryId: dbCat.id,
          tier: coll.tier,
          styleSlug: coll.styleSlug,
          tierFlavour: coll.tierFlavour,
          materialSlug: coll.materialSlug,
          materialDescriptor: coll.materialDescriptor,
          displayName: coll.displayName,
          displayOrder: coll.displayOrder,
        },
      });
      collectionIdBySlug.set(coll.slug, dbColl.id);
    }
  }

  // Pass 2 — Products + Variants. The variant rows now carry the
  // pricing_naming overrides (displayName, price, styleSlug, collectionId).
  const variantIdByKey = new Map<string, string>();
  for (const cat of data.categories) {
    const dbCat = await prisma.category.findUnique({ where: { slug: cat.slug } });
    if (!dbCat) throw new Error(`Category ${cat.slug} disappeared between passes`);
    for (const p of cat.products) {
      const dbProd = await prisma.product.create({
        data: {
          categoryId: dbCat.id,
          baseName: p.baseName,
          baseDesc: p.baseDesc,
          longDesc: p.longDesc,
          material: p.material,
          price: p.price,
        },
      });
      for (const v of p.variants) {
        const nv = naming.variants[v.urlHash];
        if (!nv) {
          throw new Error(`pricing_naming.json missing variant ${v.urlHash}`);
        }
        const collectionId = collectionIdBySlug.get(nv.collectionSlug);
        if (!collectionId) {
          throw new Error(`unknown collection ${nv.collectionSlug} for variant ${v.urlHash}`);
        }
        const dbVar = await prisma.productVariant.create({
          data: {
            productId: dbProd.id,
            collectionId,
            grainTier: v.grainTier,
            internalVariantKey: v.internalVariantKey,
            styleSlug: nv.styleSlug,
            colorName: nv.colorName,
            incidentalDetails: JSON.stringify(nv.incidentalDetails ?? []),
            displayName: nv.displayName,
            price: nv.price,
            urlHash: v.urlHash,
            primaryImage: v.primaryImage,
            galleryImages: JSON.stringify(v.galleryImages ?? [v.primaryImage]),
            title: nv.displayName,
            altText: v.altText,
            reviews: {
              create: (v.reviews ?? []).map((r: { rating: number; body: string; authorName: string }) => ({
                rating: r.rating,
                body: r.body,
                authorName: r.authorName,
              })),
            },
          },
        });
        variantIdByKey.set(v.urlHash, dbVar.id);
      }
    }
  }

  // Pass 3 — CategorySubPage + VariantSubPage. We keep the W1 sub-page
  // schema rows so legacy code paths still compile, but the new task
  // generator references /collection/<slug> instead of
  // /category/<cat>/<subpage>. Sub-page rows are harmless ballast.
  let subPageCount = 0;
  let assignmentCount = 0;
  for (const cat of data.categories) {
    if (!Array.isArray(cat.subPages) || cat.subPages.length === 0) continue;
    const dbCat = await prisma.category.findUnique({ where: { slug: cat.slug } });
    if (!dbCat) continue;
    const subPageIdBySlug = new Map<string, string>();
    for (const sp of cat.subPages) {
      const created = await prisma.categorySubPage.create({
        data: {
          categoryId: dbCat.id,
          slug: sp.slug,
          title: sp.title,
          displayOrder: sp.displayOrder ?? 0,
        },
      });
      subPageIdBySlug.set(sp.slug, created.id);
      subPageCount++;
    }
    for (const p of cat.products) {
      const posInSubPage = new Map<string, number>();
      for (const v of p.variants) {
        if (!v.subPageSlug) continue;
        const subPageId = subPageIdBySlug.get(v.subPageSlug);
        if (!subPageId) {
          throw new Error(
            `Variant ${v.urlHash} references unknown subPageSlug '${v.subPageSlug}' in category '${cat.slug}'`,
          );
        }
        const variantId = variantIdByKey.get(v.urlHash);
        if (!variantId) {
          throw new Error(`Could not resolve variant id for urlHash ${v.urlHash}`);
        }
        const pos = posInSubPage.get(subPageId) ?? 0;
        posInSubPage.set(subPageId, pos + 1);
        await prisma.variantSubPage.create({
          data: { variantId, subPageId, position: pos },
        });
        assignmentCount++;
      }
    }
  }

  // Pass 4 — peripheral cue catalogue (W3).
  let cueCount = 0;
  const cueManifestPath = join(__dirname, "..", "public", "images", "cues", "manifest.json");
  try {
    const cueManifest = JSON.parse(readFileSync(cueManifestPath, "utf8"));
    if (Array.isArray(cueManifest)) {
      for (const c of cueManifest) {
        await prisma.peripheralCue.create({
          data: {
            cueKey: c.cueKey,
            cueType: c.cueType,
            visualDescriptor: c.visualDescriptor,
            assetUrl: c.assetUrl,
            defaultPosition: c.defaultPosition,
            salienceScore: c.salienceScore,
          },
        });
        cueCount++;
      }
    }
  } catch {
    console.log(`[seed] cue manifest not found at ${cueManifestPath} — skipping cue seed`);
  }

  const counts = {
    categories: await prisma.category.count(),
    collections: await prisma.collection.count(),
    products: await prisma.product.count(),
    variants: await prisma.productVariant.count(),
    reviews: await prisma.review.count(),
    subPages: subPageCount,
    subPageAssignments: assignmentCount,
    peripheralCues: cueCount,
  };
  console.log("[seed] done:", counts);
}

main().catch((e) => { console.error(e); process.exit(1); }).finally(() => prisma.$disconnect());
