// Swap the prisma provider between "sqlite" (local dev) and "postgresql"
// (production / migrations against Supabase).
//
// Usage:
//   node scripts/swap-provider.mjs sqlite
//   node scripts/swap-provider.mjs postgresql
//
// We keep the working-copy schema at sqlite by default so `npm run dev`
// works with no extra setup. Run `npm run prisma:postgres` before invoking
// `prisma migrate` against Supabase, then `npm run prisma:sqlite` to flip
// back when you want local dev again.

import fs from "node:fs";
import path from "node:path";

const target = process.argv[2];
if (target !== "sqlite" && target !== "postgresql") {
  console.error('usage: node scripts/swap-provider.mjs <sqlite|postgresql>');
  process.exit(2);
}

const SCHEMA_PATH = path.resolve("prisma/schema.prisma");
const text = fs.readFileSync(SCHEMA_PATH, "utf8");
const re = /provider = "(sqlite|postgresql)"/;
const m = text.match(re);
if (!m) {
  console.error("[swap-provider] no `provider = \"…\"` line found in schema");
  process.exit(1);
}
if (m[1] === target) {
  console.log(`[swap-provider] schema already on ${target}, nothing to do`);
  process.exit(0);
}
fs.writeFileSync(SCHEMA_PATH, text.replace(re, `provider = "${target}"`));
console.log(`[swap-provider] ${m[1]} → ${target}`);
