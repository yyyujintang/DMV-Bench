/**
 * POST /api/encode
 *   body: { variantId?: string, payload?: unknown }
 *
 * Records an encoding event tied to the current session + turn. Used by
 * the evaluation harness / memory architectures to log "I just observed
 * X here". The website doesn't *act* on these events functionally —
 * they're audit / replay data.
 */

import { NextResponse } from "next/server";
import { cookies } from "next/headers";
import { prisma } from "@/lib/prisma";
import { SHOP_SESSION_COOKIE } from "@/lib/session";

export async function POST(req: Request) {
  const id = cookies().get(SHOP_SESSION_COOKIE)?.value;
  if (!id) return NextResponse.json({ error: "no_session" }, { status: 403 });

  let body: { variantId?: unknown; payload?: unknown };
  try {
    body = await req.json();
  } catch {
    return NextResponse.json({ error: "bad_json" }, { status: 400 });
  }
  const variantId = typeof body.variantId === "string" ? body.variantId : null;
  const payloadJson = JSON.stringify(body.payload ?? {});
  if (payloadJson.length > 16_000) {
    return NextResponse.json({ error: "payload_too_large" }, { status: 413 });
  }

  const session = await prisma.shopSession.findUnique({
    where: { id },
    select: { id: true, currentTurn: true, endedAt: true },
  });
  if (!session) return NextResponse.json({ error: "not_found" }, { status: 404 });
  if (session.endedAt) return NextResponse.json({ error: "session_ended" }, { status: 409 });

  const event = await prisma.encodingEvent.create({
    data: {
      sessionId: session.id,
      turnIndex: session.currentTurn,
      variantId,
      payload: payloadJson,
    },
  });
  return NextResponse.json({ ok: true, eventId: event.id, turnIndex: event.turnIndex });
}
