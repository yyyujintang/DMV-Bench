import Link from "next/link";

export default function CheckoutPage() {
  return (
    <>
      <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8 pt-6 pb-2 text-xs text-stone-500">
        <Link href="/" className="hover:text-stone-900">Home</Link>
        <span className="mx-2">/</span>
        <span className="text-stone-900">Checkout</span>
      </div>
      <section className="mx-auto max-w-3xl px-4 sm:px-6 lg:px-8 py-12">
        <h1 className="text-3xl font-light tracking-tight text-stone-900 mb-6">Checkout</h1>
        <form className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <input className="border border-stone-300 px-3 py-2 text-sm" placeholder="Email" disabled />
          <input className="border border-stone-300 px-3 py-2 text-sm" placeholder="Full name" disabled />
          <input className="border border-stone-300 px-3 py-2 text-sm md:col-span-2" placeholder="Address" disabled />
          <input className="border border-stone-300 px-3 py-2 text-sm" placeholder="City" disabled />
          <input className="border border-stone-300 px-3 py-2 text-sm" placeholder="Postal code" disabled />
        </form>
        <p className="text-xs text-stone-500 mt-6">
          This is a mock checkout — Studio Living is a research environment and no real orders are
          accepted.
        </p>
      </section>
    </>
  );
}
