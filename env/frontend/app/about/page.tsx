import Link from "next/link";

export default function AboutPage() {
  return (
    <>
      <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8 pt-6 pb-2 text-xs text-stone-500">
        <Link href="/" className="hover:text-stone-900">Home</Link>
        <span className="mx-2">/</span>
        <span className="text-stone-900">About</span>
      </div>
      <section className="mx-auto max-w-3xl px-4 sm:px-6 lg:px-8 py-16 prose prose-stone">
        <h1 className="text-3xl font-light tracking-tight text-stone-900 mb-6">
          About Studio Living
        </h1>
        <p className="text-stone-700 leading-relaxed mb-4">
          Studio Living designs furniture and home objects for contemporary apartments and reading
          nooks. Our collection is built around a small palette of materials and silhouettes that
          work in rooms big and small.
        </p>
        <p className="text-stone-700 leading-relaxed mb-4">
          We work with a small group of manufacturers committed to honest construction and modern
          form. Each piece is sized for everyday use.
        </p>
        <p className="text-stone-700 leading-relaxed">
          Studio Living is also a research environment for studying how agents and memory systems
          handle visually-similar products in contemporary commerce.
        </p>
      </section>
    </>
  );
}
