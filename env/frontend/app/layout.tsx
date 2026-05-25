import type { Metadata } from "next";
import { Inter } from "next/font/google";
import { headers, cookies } from "next/headers";
import "./globals.css";
import { cn } from "@/lib/utils";
import { SiteHeader } from "@/components/site-header";
import { SiteFooter } from "@/components/site-footer";
import { ModeProvider } from "@/components/mode-provider";
import { RecentlyViewed } from "@/components/recently-viewed";
import { AnnotateOverlay } from "@/components/annotate/annotate-overlay";
import { getMode } from "@/lib/mode";
import { SHOP_SESSION_COOKIE } from "@/lib/session";
import { WORKER_COOKIE } from "@/lib/annotate/session";
import { prisma } from "@/lib/prisma";

const sansFont = Inter({ subsets: ["latin"], variable: "--font-sans" });

export const metadata: Metadata = {
  title: "Studio Living — Modern Furniture",
  description: "Modern furniture and home goods for contemporary interiors.",
};

export default async function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  // The annotation tool runs in a focused study UI without category nav.
  // middleware.ts sets x-pathname; we strip the shop chrome on /annotate/*.
  const pathname = headers().get("x-pathname") ?? "";
  const isAnnotate = pathname.startsWith("/annotate");

  // Server-resolved NCP mode flows into the shop subtree via ModeProvider
  // so client components (RecentlyViewed, WishlistView) can read it from
  // context. Annotate keeps its own ungoverned layout.
  const mode = await getMode();

  // Annotation overlay mounts only when the worker actually has an
  // OPEN attempt in the DB (finishedAt IS NULL). Without this DB
  // check, stale cookies left over from prior testing make the
  // overlay appear on the public shop pages even when the worker
  // isn't doing anything — collapsed cleanly by gating on the
  // attempt record.
  const c = cookies();
  const workerCookie = c.get(WORKER_COOKIE);
  const sessionCookie = c.get(SHOP_SESSION_COOKIE);
  let showAnnotateOverlay = false;
  if (workerCookie && sessionCookie) {
    const openAttempt = await prisma.annotationSubTaskAttempt.findFirst({
      where: {
        workerId: workerCookie.value,
        shopSessionId: sessionCookie.value,
        finishedAt: null,
      },
      select: { id: true },
    });
    showAnnotateOverlay = !!openAttempt;
  }

  return (
    <html lang="en" className={cn(sansFont.variable)}>
      <body className="font-sans antialiased bg-stone-50 text-stone-900 min-h-screen flex flex-col">
        {isAnnotate ? (
          <>
            <main className="flex-1">{children}</main>
          </>
        ) : (
          <ModeProvider value={mode}>
            {showAnnotateOverlay && <AnnotateOverlay />}
            <SiteHeader />
            <main className="flex-1">{children}</main>
            <RecentlyViewed />
            <SiteFooter />
          </ModeProvider>
        )}
      </body>
    </html>
  );
}
