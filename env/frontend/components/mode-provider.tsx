/**
 * Client-side mirror of the server's resolved NCP mode.
 *
 * Server components consume `getMode()` from `lib/mode.ts` directly via
 * `next/headers`. Client components — Recently Viewed sidebar, wishlist
 * thumbnails, and other reactive surfaces landing in W2 — read the same
 * snapshot from this context. W1 publishes the shape so W2 has a stable
 * contract to plug into; it doesn't itself need a rendered provider yet.
 */

"use client";

import { createContext, useContext, type ReactNode } from "react";
import type { Mode, ResolvedMode } from "@/lib/mode";

const EMPTY_VALUE: ResolvedMode = {
  mode: "encoding",
  anchorVariantIds: new Set(),
  taskId: null,
};

export const ModeContext = createContext<ResolvedMode>(EMPTY_VALUE);

export function ModeProvider({
  value,
  children,
}: {
  value: ResolvedMode;
  children: ReactNode;
}) {
  return <ModeContext.Provider value={value}>{children}</ModeContext.Provider>;
}

export function useMode(): ResolvedMode {
  return useContext(ModeContext);
}

export type { Mode, ResolvedMode };
