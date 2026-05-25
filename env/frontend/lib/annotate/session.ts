/**
 * Annotation session helpers — Prolific URL parsing, worker upsert,
 * and the cookie that ties subsequent page loads to a worker row.
 *
 * Cookie: `dmv_worker_id` (httpOnly, sameSite=lax, 6-hour max age).
 *   - written by the entry route after upsert
 *   - read by every other /annotate/* page and /api/annotate/* route
 */

import { cookies } from "next/headers";
import { prisma } from "@/lib/prisma";

export const WORKER_COOKIE = "dmv_worker_id";

export type ProlificParams = {
  prolificId: string;
  studyId: string;
  sessionId: string | null;
};

// Real Prolific IDs are 24-char alphanumeric. We accept slightly longer
// strings for forward-compat but cap at a sane length and reject anything
// with separator characters that could indicate URL tampering.
// Test PIDs from internal smoke tests skip this check via ALLOW_TEST_PIDS.
const PROLIFIC_PID_RE = /^[A-Za-z0-9_-]{6,40}$/;
const STUDY_ID_RE = /^[A-Za-z0-9_-]{6,40}$/;
const SESSION_ID_RE = /^[A-Za-z0-9_-]{6,80}$/;

function looksLikeProlificId(v: string): boolean {
  return PROLIFIC_PID_RE.test(v);
}

export function parseProlificParams(
  sp: Record<string, string | string[] | undefined>,
): ProlificParams | null {
  const pick = (k: string): string | null => {
    const v = sp[k] ?? sp[k.toLowerCase()];
    if (Array.isArray(v)) return v[0] ?? null;
    return typeof v === "string" && v.length > 0 ? v : null;
  };
  const prolificId = pick("PROLIFIC_PID");
  const studyId = pick("STUDY_ID");
  if (!prolificId || !studyId) return null;
  if (!looksLikeProlificId(prolificId)) return null;
  if (!STUDY_ID_RE.test(studyId)) return null;
  const sessionId = pick("SESSION_ID");
  if (sessionId && !SESSION_ID_RE.test(sessionId)) return null;
  return { prolificId, studyId, sessionId };
}

export type WorkerState = {
  id: string;
  status: string;
  consentGiven: boolean;
  practiceCompleted: boolean;
};

export async function upsertWorker(p: ProlificParams): Promise<WorkerState> {
  const w = await prisma.annotationWorker.upsert({
    where: { prolificId: p.prolificId },
    create: {
      prolificId: p.prolificId,
      studyId: p.studyId,
      sessionId: p.sessionId,
    },
    update: {
      // If the worker reconnects with a new SESSION_ID we record the latest
      // one, but we never overwrite consent / status — those are sticky.
      sessionId: p.sessionId ?? undefined,
    },
    select: {
      id: true,
      status: true,
      consentGiven: true,
      practiceCompleted: true,
    },
  });
  return w;
}

// Cookie writes happen in a Route Handler (/api/annotate/entry), not here,
// because Next.js 14 forbids `cookies().set()` inside server components.

export async function getCurrentWorker(): Promise<WorkerState | null> {
  const id = cookies().get(WORKER_COOKIE)?.value;
  if (!id) return null;
  const w = await prisma.annotationWorker.findUnique({
    where: { id },
    select: {
      id: true,
      status: true,
      consentGiven: true,
      practiceCompleted: true,
    },
  });
  return w ?? null;
}
