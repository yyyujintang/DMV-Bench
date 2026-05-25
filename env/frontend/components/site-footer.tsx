import Link from "next/link";

export function SiteFooter() {
  return (
    <footer className="border-t border-stone-200 bg-white">
      <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8 py-12">
        <div className="grid grid-cols-2 md:grid-cols-4 gap-8">
          <div>
            <h3 className="text-sm font-semibold text-stone-900 mb-3">Shop</h3>
            <ul className="space-y-2 text-sm text-stone-600">
              <li><Link href="/category/vases">Vases</Link></li>
              <li><Link href="/category/lamps">Lamps</Link></li>
              <li><Link href="/category/rugs">Rugs</Link></li>
              <li><Link href="/category/tables">Tables</Link></li>
            </ul>
          </div>
          <div>
            <h3 className="text-sm font-semibold text-stone-900 mb-3">Studio Living</h3>
            <ul className="space-y-2 text-sm text-stone-600">
              <li><Link href="/about">About</Link></li>
              <li><Link href="/about">Stores</Link></li>
              <li><Link href="/about">Contact</Link></li>
            </ul>
          </div>
          <div>
            <h3 className="text-sm font-semibold text-stone-900 mb-3">Service</h3>
            <ul className="space-y-2 text-sm text-stone-600">
              <li><Link href="/about">Shipping</Link></li>
              <li><Link href="/about">Returns</Link></li>
              <li><Link href="/about">FAQ</Link></li>
            </ul>
          </div>
          <div>
            <h3 className="text-sm font-semibold text-stone-900 mb-3">Newsletter</h3>
            <p className="text-sm text-stone-600">Get updates on new collections.</p>
          </div>
        </div>
        <div className="mt-12 pt-8 border-t border-stone-200 text-xs text-stone-500">
          © Studio Living. Modern goods for everyday rooms.
        </div>
      </div>
    </footer>
  );
}
