/**
 * POST /api/annotate/workers/[id]/consent
 *   body: { consent: boolean }
 *
 * Records the worker's response to the consent form. Declining sets the
 * worker status to "abandoned"; consenting flips consentGiven=true and
 * stamps consentGivenAt. Either way the row already exists (created at
 * the entry route on first hit), so this is a pure update.
 */

import { NextResponse } from "next/server";
import { cookies } from "next/headers";
import { revalidatePath } from "next/cache";
import { prisma } from "@/lib/prisma";
import { WORKER_COOKIE } from "@/lib/annotate/session";

export async function POST(
  req: Request,
  { params }: { params: { id: string } },
) {
  const cookieWorkerId = cookies().get(WORKER_COOKIE)?.value;
  if (!cookieWorkerId || cookieWorkerId !== params.id) {
    return NextResponse.json({ error: "no_session" }, { status: 403 });
  }

  let body: { consent?: unknown };
  try {
    body = await req.json();
  } catch {
    return NextResponse.json({ error: "bad_json" }, { status: 400 });
  }
  if (typeof body.consent !== "boolean") {
    return NextResponse.json({ error: "consent_required" }, { status: 400 });
  }

  if (body.consent) {
    await prisma.annotationWorker.update({
      where: { id: params.id },
      data: { consentGiven: true, consentGivenAt: new Date() },
    });
    // /annotate and /annotate/intro both read worker state server-side;
    // invalidate them so the next navigation sees the flipped flag.
    revalidatePath("/annotate");
    revalidatePath("/annotate/intro");
    return NextResponse.json({ ok: true, next: "/annotate/intro" });
  } else {
    await prisma.annotationWorker.update({
      where: { id: params.id },
      data: {
        consentGiven: false,
        status: "abandoned",
        rejectionReason: "declined_consent",
        sessionEndedAt: new Date(),
      },
    });
    revalidatePath("/annotate");
    return NextResponse.json({ ok: true, next: "/annotate/declined" });
  }
}
