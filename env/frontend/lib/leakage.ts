/**
 * L2 leakage contract — the single most important invariant of DMV-Bench.
 *
 * Allowed text channels: product class + COARSE attribute.
 *   ✅ "Sofa", "Lamp", "Rug", "Modern Gray Loveseat"
 * Forbidden:
 *   ❌ Fine descriptors that would let text-only memory rank within-category.
 *
 * Every product title, description, alt, URL slug, and review body must pass
 * `auditLeakage(text)` before being rendered. The Phase 1 acceptance criterion
 * is **0 forbidden-vocabulary hits across all rendered HTML**.
 */

export const FORBIDDEN_FINE_DESCRIPTORS: ReadonlyArray<string> = [
  // ---- fine color names ----
  "charcoal", "slate", "dove", "heather", "gunmetal",
  "navy", "cobalt", "azure", "teal", "turquoise",
  "sage", "olive", "moss", "fern",
  "ivory", "ecru", "taupe", "mushroom",
  "mauve", "rust", "terracotta", "marsala", "ochre",
  "blush", "salmon", "coral", "peach",
  "burgundy", "wine", "claret",
  "mustard", "saffron", "amber",
  "indigo", "periwinkle", "lavender",
  "emerald", "jade", "forest",
  "graphite", "onyx", "obsidian",
  "champagne", "pearl", "cream", "bone",
  // ---- fine material names ----
  "velvet", "suede", "tweed", "boucle", "bouclé", "chenille", "corduroy",
  "linen", "jute", "sisal", "rattan",
  "oak", "walnut", "ash", "mahogany", "cherry", "teak", "maple", "birch",
  "marble", "travertine", "granite", "quartz",
  // ---- style descriptors that imply a visual signature ----
  "mid-century", "midcentury", "art-deco", "artdeco", "brutalist",
  "scandinavian", "bauhaus", "industrial", "victorian", "rustic",
  "minimalist", "maximalist", "boho",
];

const FORBIDDEN_REGEX = new RegExp(
  "\\b(" + FORBIDDEN_FINE_DESCRIPTORS.map(w => w.replace("-", "[-\\s]?")).join("|") + ")\\b",
  "gi"
);

export interface LeakageHit {
  word: string;
  index: number;
}

/**
 * Returns the list of forbidden-vocabulary hits in `text`. Empty array = clean.
 */
export function auditLeakage(text: string): LeakageHit[] {
  // Materialise the matchAll iterator into an array so this compiles
  // under tsconfig targets older than ES2015 (Next.js build is stricter
  // than `tsc --noEmit`).
  return Array.from(text.matchAll(FORBIDDEN_REGEX)).map((m) => ({
    word: m[0],
    index: m.index ?? -1,
  }));
}

/**
 * Throws if `text` contains any forbidden vocabulary. Use at seed time
 * and (optionally) at render time to gate any user-visible string.
 */
export function assertCleanLeakage(text: string, context: string = ""): void {
  const hits = auditLeakage(text);
  if (hits.length > 0) {
    const words = Array.from(new Set(hits.map(h => h.word.toLowerCase()))).join(", ");
    throw new Error(
      `L2 leakage violation${context ? ` in ${context}` : ""}: forbidden words [${words}] in "${text.slice(0, 120)}..."`
    );
  }
}
