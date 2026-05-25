/**
 * Annotation tool layout — bypasses the e-commerce SiteHeader / SiteFooter
 * so workers see a focused study interface without category navigation
 * leaking task-relevant information.
 */

import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "DMV-Bench Annotation Study",
  robots: { index: false, follow: false },
};

export default function AnnotateLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <div className="min-h-screen bg-stone-50 text-stone-900 flex flex-col">
      <div className="border-b border-stone-200 bg-white">
        <div className="mx-auto max-w-3xl px-4 sm:px-6 lg:px-8 h-12 flex items-center">
          <span className="text-xs font-medium uppercase tracking-widest text-stone-500">
            DMV-Bench Annotation Study
          </span>
        </div>
      </div>
      <main className="flex-1 mx-auto w-full max-w-3xl px-4 sm:px-6 lg:px-8 py-10">
        {children}
      </main>
    </div>
  );
}
