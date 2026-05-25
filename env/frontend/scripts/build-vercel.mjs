// Vercel build entry point. Runs on Vercel's ephemeral build VM and
// rewrites schema.prisma from sqlite → postgresql before next build.
// The user's local schema is never touched.
//
// Triggered by the `build:vercel` npm script, which Vercel runs via the
// project's Build Command setting (see DEPLOY.md).

import { spawnSync } from "node:child_process";

const swap = spawnSync(
  process.execPath,
  ["scripts/swap-provider.mjs", "postgresql"],
  { stdio: "inherit" },
);
if (swap.status !== 0) process.exit(swap.status ?? 1);
