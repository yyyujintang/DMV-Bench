#!/usr/bin/env node
/**
 * Copy NCP-validated task instances from the Python pool into
 * env/frontend/public/tasks/validated/<sub-task>/<task_id>.json so the
 * annotation API routes can read them at runtime.
 *
 * Source: <repo>/tasks/pool/validated/{NC,SA,IC,VL}/*.json
 * Dest:   <repo>/VisMem-Diag/env/frontend/public/tasks/validated/<same shape>
 *
 * Idempotent: re-writes existing files only when content differs (so dev
 * server file-watchers don't trigger a needless rebuild on every npm run dev).
 *
 * Rejected tasks are intentionally NOT copied.
 */
import { promises as fs } from "node:fs";
import path from "node:path";

const SUB_TASKS = ["NC", "SA", "IC", "VL", "PD"];

// frontend lives at <repo>/VisMem-Diag/env/frontend, this script is at
// frontend/scripts/, so the source pool is four levels up.
const FRONTEND = path.resolve(path.dirname(new URL(import.meta.url).pathname), "..");
const SRC_ROOT = path.resolve(FRONTEND, "..", "..", "..", "tasks", "pool", "validated");
const DST_ROOT = path.resolve(FRONTEND, "public", "tasks", "validated");

async function copyIfChanged(src, dst) {
  const incoming = await fs.readFile(src);
  try {
    const existing = await fs.readFile(dst);
    if (existing.equals(incoming)) return false;
  } catch {
    /* dst absent — fall through to write */
  }
  await fs.mkdir(path.dirname(dst), { recursive: true });
  await fs.writeFile(dst, incoming);
  return true;
}

async function listJson(dir) {
  try {
    const entries = await fs.readdir(dir, { withFileTypes: true });
    return entries
      .filter((e) => e.isFile() && e.name.endsWith(".json") && e.name !== "reason.json")
      .map((e) => e.name);
  } catch (e) {
    if (e.code === "ENOENT") return [];
    throw e;
  }
}

async function main() {
  // Defensive: on a build host that DOESN'T have the Python pool
  // mounted (Vercel uploads only env/frontend, so ../../../tasks/...
  // doesn't exist there), enumerating 0 src files would otherwise
  // PRUNE every dst file we just uploaded with the deploy. Bail
  // early so the pre-synced public/tasks/validated/ survives.
  try {
    await fs.access(SRC_ROOT);
  } catch (e) {
    if (e.code === "ENOENT") {
      console.log(`[sync-task-pool] SRC_ROOT ${SRC_ROOT} absent — skipping (assume pre-synced)`);
      return;
    }
    throw e;
  }
  let copied = 0;
  let total = 0;
  let pruned = 0;
  for (const sub of SUB_TASKS) {
    const srcDir = path.join(SRC_ROOT, sub);
    const dstDir = path.join(DST_ROOT, sub);
    await fs.mkdir(dstDir, { recursive: true });
    const names = await listJson(srcDir);
    const srcSet = new Set(names);
    total += names.length;
    for (const name of names) {
      const changed = await copyIfChanged(path.join(srcDir, name), path.join(dstDir, name));
      if (changed) copied += 1;
    }
    // Prune stale: any *.json in dst that's not in src must be a
    // leftover from a previous generation. Delete it so worker picks
    // can never land on an obsolete task spec.
    const dstNames = await listJson(dstDir);
    for (const name of dstNames) {
      if (!srcSet.has(name)) {
        await fs.unlink(path.join(dstDir, name));
        pruned += 1;
      }
    }
  }
  console.log(`[sync-task-pool] ${total} validated tasks (${copied} updated, ${pruned} pruned) → public/tasks/validated/`);
}

main().catch((err) => {
  console.error("[sync-task-pool] failed:", err);
  process.exit(1);
});
