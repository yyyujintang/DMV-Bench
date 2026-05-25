import Link from "next/link";
import { Search, User } from "lucide-react";
import { prisma } from "@/lib/prisma";
import { HeaderCounters } from "@/components/header-counters";

export async function SiteHeader() {
  const categories = await prisma.category.findMany({
    orderBy: { sortOrder: "asc" },
    select: { slug: true, display: true },
  });

  return (
    <header className="sticky top-0 z-40 bg-white/95 backdrop-blur border-b border-stone-200">
      <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
        <div className="flex h-16 items-center justify-between">
          <Link href="/" className="text-xl font-semibold tracking-tight text-stone-900">
            STUDIO LIVING
          </Link>
          <nav className="hidden md:flex items-center gap-8 text-sm">
            {categories.map((c) => (
              <Link
                key={c.slug}
                href={`/category/${c.slug}`}
                className="text-stone-600 hover:text-stone-900 transition-colors"
              >
                {c.display}
              </Link>
            ))}
          </nav>
          <div className="flex items-center gap-3">
            <Link href="/search" aria-label="Search" className="text-stone-700 hover:text-stone-900">
              <Search className="h-5 w-5" />
            </Link>
            <Link href="/about" aria-label="Account" className="text-stone-700 hover:text-stone-900">
              <User className="h-5 w-5" />
            </Link>
            <HeaderCounters />
          </div>
        </div>
      </div>
    </header>
  );
}
