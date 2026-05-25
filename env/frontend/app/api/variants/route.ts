/**
 * Variant lookup by urlHash list — used by client-side views (wishlist) that
 * need to hydrate localStorage IDs into full product info.
 *
 * GET /api/variants?ids=h1,h2,h3
 *   → { variants: [{ urlHash, title, primaryImage, altText, price }, ...] }
 *
 * MR1 extension (proposal_tasks_v2.md §3.6): in recall mode, any urlHash
 * in `mode.anchorVariantIds` is returned as `{ urlHash, status: "anchor_hidden" }`
 * with no image / title / price — the only signal a client gets is that
 * the entry exists but is suppressed.
 */

import { NextResponse } from "next/server";
import { prisma } from "@/lib/prisma";
import { getMode } from "@/lib/mode";

export async function GET(req: Request) {
  const url = new URL(req.url);
  const ids = (url.searchParams.get("ids") ?? "")
    .split(",").map((s) => s.trim()).filter(Boolean);
  if (ids.length === 0) {
    return NextResponse.json({ variants: [] });
  }
  const mode = await getMode();
  const variants = await prisma.productVariant.findMany({
    where: { urlHash: { in: ids } },
  });
  const ordered = ids
    .map((h) => variants.find((v) => v.urlHash === h))
    .filter((v): v is NonNullable<typeof v> => Boolean(v))
    .map((v) => {
      if (mode.mode === "recall" && mode.anchorVariantIds.has(v.urlHash)) {
        return { urlHash: v.urlHash, status: "anchor_hidden" as const };
      }
      return {
        variantId: v.id,
        urlHash: v.urlHash,
        title: v.displayName,
        altText: v.altText,
        primaryImage: v.primaryImage,
        price: v.price,
      };
    });
  return NextResponse.json({ variants: ordered });
}
