"use client";

import Image from "next/image";
import { useState } from "react";

interface Props {
  images: string[];
  alt: string;
}

export function ProductGallery({ images, alt }: Props) {
  const [active, setActive] = useState(0);
  const imgs = images.length > 0 ? images : [""];
  return (
    <div className="grid grid-cols-1 gap-4">
      <div className="aspect-square bg-white border border-stone-200 relative">
        {imgs[active] && (
          <Image
            src={imgs[active]}
            alt={alt}
            fill
            sizes="(min-width: 1024px) 50vw, 100vw"
            className="object-contain"
            priority
          />
        )}
      </div>
      {imgs.length > 1 && (
        <div className="grid grid-cols-5 gap-2">
          {imgs.map((src, i) => (
            <button
              key={i}
              onClick={() => setActive(i)}
              className={
                "aspect-square bg-white border relative transition-colors " +
                (i === active ? "border-stone-900" : "border-stone-200 hover:border-stone-400")
              }
              aria-label={`Show image ${i + 1}`}
            >
              <Image src={src} alt={`${alt} thumbnail ${i + 1}`} fill sizes="20vw" className="object-contain" />
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
