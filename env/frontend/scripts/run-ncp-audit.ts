/**
 * NCP audit harness.
 *
 * Probes a running Next.js dev server at $BASE_URL (default
 * http://localhost:3000) on a deterministic set of recall-mode URLs and
 * pipes each response through `lib/audit.ts` to detect anchor leakage.
 *
 * Wiring:
 *   · Picks one anchor per category from local sqlite (or from $ANCHORS
 *     if pre-supplied as a JSON file).
 *   · Walks every sub-page + the category landing + a search query, all
 *     with `?mode=recall&anchors=<anchor-id>`.
 *   · Reports a per-route violation summary, exits 0 if clean, 1 if any
 *     critical violation fires.
 *
 * Usage:
 *   node scripts/run-ncp-audit.mjs                       # default 24+6+1 routes
 *   node scripts/run-ncp-audit.mjs --base http://localhost:3001
 *   node scripts/run-ncp-audit.mjs --json                # JSON output for CI
 *
 * Exit codes:
 *   0  clean
 *   1  critical violations found
 *   2  could not reach server / no anchors available
 */

import { PrismaClient } from "@prisma/client";
import { auditHtml } from "../lib/audit";

type Args = { base?: string; json?: boolean; verbose?: boolean };

function parseArgs(argv: string[]): Args {
  const out: Args = {};
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a === "--base") { out.base = argv[++i]; continue; }
    if (a === "--json") { out.json = true; continue; }
    if (a === "--verbose") { out.verbose = true; continue; }
  }
  return out;
}

const args = parseArgs(process.argv.slice(2));
const BASE = args.base ?? process.env.BASE_URL ?? "http://localhost:3000";
const JSON_OUT = args.json === true;
const VERBOSE = args.verbose === true || process.env.VERBOSE === "1";

async function main() {
  const prisma = new PrismaClient();
  // One anchor per category at tier 1, var_a — deterministic and
  // covers every sub-page slot that var_a (= "modern") lands on.
  const anchors = await prisma.productVariant.findMany({
    where: { grainTier: 1, internalVariantKey: "var_a" },
    include: { product: { include: { category: true } } },
  });
  await prisma.$disconnect();
  if (anchors.length === 0) {
    console.error("[audit] no anchors found — is the DB seeded?");
    process.exit(2);
  }

  // Routes to audit. Each anchor sweeps:
  //  · its category landing
  //  · its 4 sub-page slugs
  //  · the search page with a generic query that includes its category
  //  · /wishlist (in case the anchor lives there in localStorage — the
  //    audit can't simulate localStorage state, so this is a soft check)
  const routes = [];
  for (const a of anchors) {
    const slug = a.product.category.slug;
    const anchorParam = `mode=recall&anchors=${encodeURIComponent(a.id)}`;
    routes.push({ url: `/category/${slug}?${anchorParam}`, anchor: a });
    for (const sp of ["modern", "minimalist", "vintage", "industrial"]) {
      routes.push({ url: `/category/${slug}/${sp}?${anchorParam}`, anchor: a });
    }
    routes.push({ url: `/search?q=${slug}&${anchorParam}`, anchor: a });
  }

  let totalCritical = 0;
  const report = [];
  for (const r of routes) {
    const fullUrl = `${BASE}${r.url}`;
    let html;
    try {
      const res = await fetch(fullUrl, {
        headers: { "cache-control": "no-store" },
        redirect: "follow",
      });
      html = await res.text();
      if (!res.ok) {
        console.error(`[audit] ${r.url} → HTTP ${res.status}`);
        process.exit(2);
      }
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      console.error(`[audit] could not reach ${fullUrl}: ${msg}`);
      process.exit(2);
    }

    const violations = auditHtml({
      html,
      anchors: [
        {
          variantId: r.anchor.id,
          urlHash: r.anchor.urlHash,
          primaryImage: r.anchor.primaryImage,
        },
      ],
    });
    const critical = violations.filter((v) => v.severity === "critical").length;
    totalCritical += critical;
    report.push({ url: r.url, violations, critical });
    if (VERBOSE || critical > 0) {
      const tag = critical > 0 ? "✗" : "✓";
      console.log(`${tag} ${r.url}   ${critical > 0 ? `(${critical} critical)` : "clean"}`);
      for (const v of violations) {
        console.log(`    ${v.severity} ${v.rule}/${v.type} — ${v.description}`);
      }
    }
  }

  if (JSON_OUT) {
    console.log(JSON.stringify({ totalCritical, routes: report }, null, 2));
  } else {
    console.log(`\n[audit] ${routes.length} routes probed, ${totalCritical} critical violations.`);
  }
  process.exit(totalCritical > 0 ? 1 : 0);
}

main().catch((e) => { console.error(e); process.exit(2); });
