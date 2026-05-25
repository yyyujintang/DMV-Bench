import Link from "next/link";

export default function CartPage() {
  return (
    <>
      <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8 pt-6 pb-2 text-xs text-stone-500">
        <Link href="/" className="hover:text-stone-900">Home</Link>
        <span className="mx-2">/</span>
        <span className="text-stone-900">Cart</span>
      </div>
      <section className="mx-auto max-w-3xl px-4 sm:px-6 lg:px-8 py-24 text-center">
        <h1 className="text-3xl font-light tracking-tight text-stone-900 mb-3">Your cart</h1>
        <p className="text-stone-500 mb-8">Your cart is empty.</p>
        <Link
          href="/category/chairs"
          className="inline-block bg-stone-900 text-white px-8 py-3 text-sm hover:bg-stone-800 transition-colors"
        >
          Browse the catalog
        </Link>
      </section>
    </>
  );
}
