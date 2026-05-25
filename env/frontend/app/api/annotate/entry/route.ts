/**
 * GET /api/annotate/entry?PROLIFIC_PID=…&STUDY_ID=…&SESSION_ID=…
 *
 * Reached only via the middleware redirect from /annotate?PROLIFIC_PID=…
 * Upserts the AnnotationWorker row, writes the dmv_worker_id cookie, then
 * redirects back to /annotate (no params) where the page renders consent
 * or routes the worker forward based on saved state.
 */

import { NextResponse } from "next/server";
import { cookies } from "next/headers";
import {
  parseProlificParams,
  upsertWorker,
  WORKER_COOKIE,
} from "@/lib/annotate/session";

const COOKIE_MAX_AGE = 60 * 60 * 6;

/**
 * Build the post-entry redirect URL using the *originating* Host header
 * rather than whatever Next.js canonicalised req.url to. Otherwise a
 * client that came in on host A gets redirected to host B (e.g.
 * `127.0.0.1` → `localhost` in dev, or behind a proxy with a
 * canonical-name rewrite in prod) and drops the freshly-set cookie at
 * the host boundary.
 */
function backToAnnotate(req: Request): URL {
  const host =
    req.headers.get("x-forwarded-host") ??
    req.headers.get("host") ??
    new URL(req.url).host;
  const proto =
    req.headers.get("x-forwarded-proto") ??
    new URL(req.url).protocol.replace(":", "");
  return new URL(`${proto}://${host}/annotate`);
}

export async function GET(req: Request) {
  const url = new URL(req.url);
  const sp: Record<string, string> = {};
  url.searchParams.forEach((v, k) => { sp[k] = v; });

  const params = parseProlificParams(sp);
  const back = backToAnnotate(req);

  if (!params) {
    return NextResponse.redirect(back);
  }

  const worker = await upsertWorker(params);
  cookies().set(WORKER_COOKIE, worker.id, {
    httpOnly: true,
    sameSite: "lax",
    secure: process.env.NODE_ENV === "production",
    maxAge: COOKIE_MAX_AGE,
    path: "/",
  });
  return NextResponse.redirect(back);
}
