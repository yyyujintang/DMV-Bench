"use client";

import { useEffect, useState } from "react";

const KEY = "dmv_wishlist";

interface Props {
  urlHash: string;
  title: string;
}

export function AddToWishlist({ urlHash, title }: Props) {
  const [inList, setInList] = useState(false);
  const [count, setCount] = useState(0);
  const [flash, setFlash] = useState<string | null>(null);

  useEffect(() => {
    const list: string[] = JSON.parse(localStorage.getItem(KEY) ?? "[]");
    setInList(list.includes(urlHash));
    setCount(list.length);
  }, [urlHash]);

  function toggle() {
    const list: string[] = JSON.parse(localStorage.getItem(KEY) ?? "[]");
    let next: string[];
    let action: string;
    if (list.includes(urlHash)) {
      next = list.filter((x) => x !== urlHash);
      action = "Removed from wishlist";
    } else {
      next = [...list, urlHash];
      action = "Added to wishlist";
    }
    localStorage.setItem(KEY, JSON.stringify(next));
    setInList(next.includes(urlHash));
    setCount(next.length);
    setFlash(action);
    window.setTimeout(() => setFlash(null), 2000);
  }

  return (
    <div className="space-y-2">
      <button
        onClick={toggle}
        data-dmv-action="add-to-wishlist"
        data-dmv-url-hash={urlHash}
        className={
          "w-full px-8 py-3 text-sm tracking-wide transition-colors " +
          (inList
            ? "bg-stone-200 text-stone-900 hover:bg-stone-300"
            : "bg-stone-900 text-white hover:bg-stone-800")
        }
      >
        {inList ? `♥ In wishlist (${count})` : "Add to wishlist"}
      </button>
      {flash && (
        <p className="text-xs text-stone-600 text-center" aria-live="polite">
          {flash} — "{title}"
        </p>
      )}
    </div>
  );
}
