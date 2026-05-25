"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { Heart, ShoppingBag } from "lucide-react";

function readCount(key: string): number {
  try {
    const list = JSON.parse(localStorage.getItem(key) ?? "[]");
    return Array.isArray(list) ? list.length : 0;
  } catch {
    return 0;
  }
}

export function HeaderCounters() {
  const [wishlist, setWishlist] = useState(0);

  useEffect(() => {
    const update = () => {
      setWishlist(readCount("dmv_wishlist"));
    };
    update();
    window.addEventListener("storage", update);
    window.addEventListener("focus", update);
    const t = window.setInterval(update, 1000);
    return () => {
      window.removeEventListener("storage", update);
      window.removeEventListener("focus", update);
      window.clearInterval(t);
    };
  }, []);

  return (
    <div className="flex items-center gap-4">
      <Link href="/wishlist" aria-label="Wishlist" className="text-stone-700 hover:text-stone-900 relative">
        <Heart className="h-5 w-5" />
        {wishlist > 0 && (
          <span className="absolute -top-1 -right-2 bg-stone-900 text-white text-[10px] rounded-full w-4 h-4 flex items-center justify-center">
            {wishlist}
          </span>
        )}
      </Link>
      <Link href="/cart" aria-label="Cart" className="text-stone-700 hover:text-stone-900 relative">
        <ShoppingBag className="h-5 w-5" />
      </Link>
    </div>
  );
}
