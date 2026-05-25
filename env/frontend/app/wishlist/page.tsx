import Link from "next/link";
import { WishlistView } from "@/components/wishlist-view";

export default function WishlistPage() {
  return (
    <>
      <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8 pt-6 pb-2 text-xs text-stone-500">
        <Link href="/" className="hover:text-stone-900">Home</Link>
        <span className="mx-2">/</span>
        <span className="text-stone-900">Wishlist</span>
      </div>
      <header className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8 py-6 border-b border-stone-200">
        <h1 className="text-3xl lg:text-4xl font-light tracking-tight text-stone-900">
          Your wishlist
        </h1>
        <p className="text-stone-600 mt-2 max-w-2xl">
          Products you saved while browsing. The wishlist is stored locally in your browser.
        </p>
      </header>
      <WishlistView />
    </>
  );
}
