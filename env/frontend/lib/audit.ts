/**
 * DOM-level NCP audit.
 *
 * Pure function: given the rendered HTML for a recall-mode page plus the
 * known anchor identity (variant id, urlHash, primaryImage path), return
 * the violations detected. Used by `scripts/run-ncp-audit.mjs` (the W2
 * harness) and reusable from the future W4 per-request audit hook.
 *
 * Detection strategy is intentionally string-level (not a DOM parser):
 *   · cheaper than a JSDOM parse for thousands of pages
 *   · resilient to minor markup changes
 *   · the placeholder element keeps `data-variant-id` on purpose, so we
 *     scope the "attribute visible" check to non-placeholder occurrences
 *     using a regex that requires the missing aria-hidden marker.
 */

export type ViolationRule = "MR1" | "MR2";
export type ViolationSeverity = "critical" | "warning";

export type ViolationType =
  | "anchor_image_visible"
  | "anchor_link_visible"
  | "anchor_attribute_visible"
  | "direct_link_to_answer";

export interface Violation {
  rule: ViolationRule;
  type: ViolationType;
  severity: ViolationSeverity;
  description: string;
  evidence: string;
}

export interface AnchorIdentity {
  variantId: string;
  urlHash: string;
  primaryImage: string;
}

export interface AuditInput {
  html: string;
  anchors: readonly AnchorIdentity[];
}

/**
 * Run all MR1 checks. Returns the empty array if the page is clean.
 */
export function auditHtml({ html, anchors }: AuditInput): Violation[] {
  const violations: Violation[] = [];

  for (const a of anchors) {
    // --- MR1: anchor image source must not appear in src= or srcset= ---
    // Next.js Image rewrites paths through `/_next/image?url=…&w=…&q=…`.
    // Match either the literal path or the encoded form inside Next's
    // optimisation URL.
    const literalImgRe = new RegExp(
      `src=["'][^"']*${escapeRegex(a.primaryImage)}[^"']*["']`,
      "g",
    );
    if (literalImgRe.test(html)) {
      violations.push({
        rule: "MR1",
        type: "anchor_image_visible",
        severity: "critical",
        description: `Anchor image ${a.primaryImage} appears in a rendered src= attribute`,
        evidence: a.primaryImage,
      });
    }
    const encodedPath = encodeURIComponent(a.primaryImage);
    if (encodedPath !== a.primaryImage) {
      const optimisedRe = new RegExp(
        `/_next/image\\?[^"']*url=${escapeRegex(encodedPath)}`,
        "g",
      );
      if (optimisedRe.test(html)) {
        violations.push({
          rule: "MR1",
          type: "anchor_image_visible",
          severity: "critical",
          description: `Anchor image ${a.primaryImage} appears through Next's image optimiser`,
          evidence: encodedPath,
        });
      }
    }

    // --- MR1: anchor /product/<urlHash> link must not appear in any
    //        href= or anchor element. Placeholders never emit <a> tags. ---
    const hrefRe = new RegExp(
      `href=["']\/product\/${escapeRegex(a.urlHash)}["']`,
      "g",
    );
    if (hrefRe.test(html)) {
      violations.push({
        rule: "MR1",
        type: "anchor_link_visible",
        severity: "critical",
        description: `Anchor link /product/${a.urlHash} appears in an href`,
        evidence: a.urlHash,
      });
    }

    // --- MR1: data-variant-id="<anchor>" attribute visible.
    //        Placeholder elements carry the id intentionally and set
    //        aria-hidden="true"; those occurrences are exempt.
    const attrRe = new RegExp(
      `data-variant-id=["']${escapeRegex(a.variantId)}["']`,
      "g",
    );
    let attrMatch: RegExpExecArray | null;
    const occurrences: number[] = [];
    while ((attrMatch = attrRe.exec(html)) !== null) {
      occurrences.push(attrMatch.index);
    }
    for (const idx of occurrences) {
      // Locate the opening element containing this attribute, then check
      // whether that element also carries aria-hidden="true".
      const elementStart = html.lastIndexOf("<", idx);
      const elementEnd = html.indexOf(">", idx);
      if (elementStart < 0 || elementEnd < 0) continue;
      const elementOpen = html.slice(elementStart, elementEnd + 1);
      if (/aria-hidden=["']true["']/.test(elementOpen)) continue; // placeholder, exempt
      violations.push({
        rule: "MR1",
        type: "anchor_attribute_visible",
        severity: "critical",
        description: `Anchor variant id ${a.variantId} is rendered on a non-placeholder element`,
        evidence: elementOpen.slice(0, 200),
      });
    }
  }

  return violations;
}

function escapeRegex(s: string): string {
  return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}
