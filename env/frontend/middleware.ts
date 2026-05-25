/**
 * Middleware does three things:
 *
 *   1. Funnels Prolific-entry traffic ( /annotate?PROLIFIC_PID=… ) through
 *      a Route Handler that can write the session cookie. Server components
 *      in App Router cannot call `cookies().set()`, so we redirect once
 *      through /api/annotate/entry, which upserts the worker row, writes
 *      the dmv_worker_id cookie, and redirects back to /annotate (no params
 *      so the loop stops on the second hit).
 *
 *   2. Exposes the current pathname as an `x-pathname` request header so
 *      the root layout can skip the shop chrome (SiteHeader / SiteFooter)
 *      on /annotate/* without a route-group refactor of every existing page.
 *
 *   3. Resolves the NCP rendering mode (W1 of proposal_website.md). Until
 *      the W4 session API lands, mode + anchor list are passed as URL query
 *      params for testing: `?mode=recall&anchors=v1,v2`. Middleware forwards
 *      them as `x-mode` and `x-anchors` request headers; server components
 *      read those via `getMode()` in `lib/mode.ts`. Encoding is the default
 *      when no header is set.
 */

import { NextResponse, type NextRequest } from "next/server";

export const config = {
  matcher: ["/((?!_next/|favicon\\.ico|images/|fonts/).*)"],
};

const ANCHOR_LIST_RE = /^[A-Za-z0-9_,-]{1,2048}$/;

export function middleware(req: NextRequest) {
  const url = req.nextUrl;

  // Prolific entry — only redirect if the prolific param is actually present,
  // so refreshes of /annotate (without params) don't bounce.
  if (
    url.pathname === "/annotate" &&
    url.searchParams.has("PROLIFIC_PID")
  ) {
    const dest = url.clone();
    dest.pathname = "/api/annotate/entry";
    return NextResponse.redirect(dest);
  }

  const headers = new Headers(req.headers);
  headers.set("x-pathname", url.pathname);

  // NCP mode resolution.
  //   Priority: session cookie (W4) > URL params (W1/W2 testing) > encoding default.
  //   Middleware (Edge runtime) can't query Prisma directly, so we just
  //   forward the cookie value; `lib/mode.ts` does the DB lookup.
  const sessionId = req.cookies.get("dmv_shop_session_id")?.value;
  if (sessionId && /^[A-Za-z0-9_-]{1,64}$/.test(sessionId)) {
    headers.set("x-shop-session-id", sessionId);
  }
  const rawMode = url.searchParams.get("mode");
  if (rawMode === "recall") {
    headers.set("x-mode", "recall");
  } else if (rawMode === "encoding") {
    headers.set("x-mode", "encoding");
  }
  const rawAnchors = url.searchParams.get("anchors");
  if (rawAnchors && ANCHOR_LIST_RE.test(rawAnchors)) {
    headers.set("x-anchors", rawAnchors);
  }

  // W3 peripheral cue keys (URL-param fallback; sessions take priority).
  const rawCues = url.searchParams.get("cues");
  if (rawCues && /^[A-Za-z0-9_,-]{1,400}$/.test(rawCues)) {
    headers.set("x-cue-keys", rawCues);
  }

  return NextResponse.next({ request: { headers } });
}
