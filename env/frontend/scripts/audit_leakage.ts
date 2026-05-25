/**
 * L2 leakage CI gate.
 *
 * Crawls every rendered HTML page of the running dev server, strips scripts +
 * css, and greps for forbidden fine-grained vocabulary. Exits non-zero on any
 * hit so the script can sit in CI / pre-deploy.
 *
 * Usage:
 *   npm run audit-leakage           # against http://localhost:3000
 *   AUDIT_BASE=https://...  ./audit_leakage.ts   # against deployed site
 */

import { chromium } from "playwright";
import { auditLeakage } from "../lib/leakage";
import { PrismaClient } from "@prisma/client";

const BASE = process.env.AUDIT_BASE ?? "http://localhost:3000";

async function buildRoutes(): Promise<string[]> {
  const prisma = new PrismaClient();
  try {
    const cats = await prisma.category.findMany({ select: { slug: true } });
    const variants = await prisma.productVariant.findMany({
      take: 12, select: { urlHash: true },
    });
    const slugs = cats.map((c) => c.slug);
    const hashes = variants.map((v) => v.urlHash);
    return [
      "/",
      ...slugs.map((s) => `/category/${s}`),
      ...slugs.map((s) => `/category/${s}?tier=2`),
      ...slugs.map((s) => `/category/${s}?tier=3`),
      ...hashes.map((h) => `/product/${h}`),
      `/compare?ids=${hashes.slice(0, 4).join(",")}`,
      "/search?q=vase",
      "/search?q=chair",
      "/cart",
      "/checkout",
      "/about",
    ];
  } finally {
    await prisma.$disconnect();
  }
}

function visibleText(html: string): string {
  return html
    .replace(/<script[^>]*>[\s\S]*?<\/script>/gi, " ")
    .replace(/<style[^>]*>[\s\S]*?<\/style>/gi, " ")
    .replace(/<noscript[^>]*>[\s\S]*?<\/noscript>/gi, " ")
    .replace(/<[^>]+>/g, " ")
    .replace(/\s+/g, " ");
}

async function main() {
  const routes = await buildRoutes();
  const browser = await chromium.launch();
  const ctx = await browser.newContext({ viewport: { width: 1280, height: 900 } });
  let totalHits = 0;
  const violations: Array<{ route: string; words: string[]; sample: string }> = [];

  for (const r of routes) {
    const page = await ctx.newPage();
    const resp = await page.goto(BASE + r, { waitUntil: "networkidle" });
    if (!resp || !resp.ok()) {
      console.error(`  ✗ ${r}  HTTP ${resp?.status() ?? "?"}`);
      await page.close();
      continue;
    }
    const html = await page.content();
    const text = visibleText(html);
    const hits = auditLeakage(text);
    if (hits.length > 0) {
      const words = Array.from(new Set(hits.map((h) => h.word.toLowerCase())));
      totalHits += hits.length;
      violations.push({
        route: r,
        words,
        sample: text.slice(Math.max(0, hits[0].index - 30), hits[0].index + 60),
      });
      console.error(`  ✗ ${r}  ${hits.length} hits  [${words.join(", ")}]`);
    } else {
      console.log(`  ✓ ${r}`);
    }
    await page.close();
  }
  await browser.close();

  console.log(`\n[audit] ${routes.length} routes scanned, ${totalHits} hits across ${violations.length} pages`);
  if (totalHits > 0) {
    console.error("L2 leakage audit FAILED.");
    for (const v of violations) {
      console.error(`  ${v.route}  →  ${v.words.join(", ")}  ("${v.sample}")`);
    }
    process.exit(1);
  } else {
    console.log("L2 leakage audit PASSED.");
  }
}

main().catch((e) => {
  console.error(e);
  process.exit(2);
});
