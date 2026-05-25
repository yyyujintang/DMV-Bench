/**
 * Annotation task pool — enumerates validated TaskInstance JSONs at
 * `public/tasks/validated/<sub-task>/` and picks one for a given worker.
 *
 * The pool is small (12 today). We read directory listings on demand and
 * skip a per-process cache; an unattempted-task lookup hits the DB anyway,
 * so the I/O budget is the DB roundtrip, not the readdir.
 */
import * as fs from "node:fs/promises";
import path from "node:path";
import { prisma } from "@/lib/prisma";
import { SUB_TASKS, type SubTask } from "@/lib/annotate/task-instance";

const POOL_ROOT = path.resolve(process.cwd(), "public", "tasks", "validated");

export async function listTaskIds(sub: SubTask): Promise<string[]> {
  const dir = path.join(POOL_ROOT, sub);
  try {
    const names = await fs.readdir(dir);
    return names
      .filter((n) => n.endsWith(".json"))
      .map((n) => n.slice(0, -".json".length))
      .sort();
  } catch (e) {
    if ((e as NodeJS.ErrnoException).code === "ENOENT") return [];
    throw e;
  }
}

export async function listAllTaskIds(): Promise<Record<SubTask, string[]>> {
  const entries = await Promise.all(
    SUB_TASKS.map(async (s) => [s, await listTaskIds(s)] as const),
  );
  return Object.fromEntries(entries) as Record<SubTask, string[]>;
}

/**
 * Pick the lexicographically-first task ID for `sub` that this worker has
 * not yet attempted. If the worker has attempted every task in the pool,
 * cycle: pick the oldest attempted-but-not-completed-correctly task. If
 * the worker has correctly completed every task, return null.
 */
export async function pickTaskForWorker(
  sub: SubTask,
  workerId: string,
): Promise<string | null> {
  const pool = await listTaskIds(sub);
  if (pool.length === 0) return null;
  const attempts = await prisma.annotationSubTaskAttempt.findMany({
    where: { workerId, subTask: sub },
    select: { taskId: true, correct: true },
  });
  const seenIds = new Set(attempts.map((a) => a.taskId));
  const correctIds = new Set(attempts.filter((a) => a.correct).map((a) => a.taskId));
  // First pass: an unattempted task.
  const unattempted = pool.find((id) => !seenIds.has(id));
  if (unattempted) return unattempted;
  // Second pass: a task the worker tried but didn't get right.
  const retryable = pool.find((id) => !correctIds.has(id));
  if (retryable) return retryable;
  return null;
}
