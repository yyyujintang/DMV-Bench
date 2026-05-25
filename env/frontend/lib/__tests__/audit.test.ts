import { describe, expect, it } from "vitest";
import { auditHtml, type AnchorIdentity } from "@/lib/audit";

const ANCHOR: AnchorIdentity = {
  variantId: "anchor_id_xyz",
  urlHash: "abc123ef",
  primaryImage: "/images/chair/var_a_t1.png",
};

describe("auditHtml", () => {
  it("returns no violations for clean HTML (placeholder only)", () => {
    const html = `
      <html><body>
        <div data-testid="product-card-placeholder"
             data-variant-id="anchor_id_xyz"
             aria-hidden="true">…</div>
        <a href="/product/some_other_hash">other</a>
        <img src="/images/lamp/var_b_t1.png" alt="" />
      </body></html>
    `;
    const v = auditHtml({ html, anchors: [ANCHOR] });
    expect(v).toEqual([]);
  });

  it("flags anchor_image_visible when src= references anchor's image path", () => {
    const html = `
      <body>
        <img src="/images/chair/var_a_t1.png" alt="leak" />
      </body>
    `;
    const v = auditHtml({ html, anchors: [ANCHOR] });
    expect(v.length).toBeGreaterThanOrEqual(1);
    const types = v.map((x) => x.type);
    expect(types).toContain("anchor_image_visible");
    expect(v.every((x) => x.severity === "critical")).toBe(true);
  });

  it("flags anchor_image_visible via Next image optimiser URL", () => {
    const encoded = encodeURIComponent("/images/chair/var_a_t1.png");
    const html = `<img srcset="/_next/image?url=${encoded}&w=200&q=75 1x" />`;
    const v = auditHtml({ html, anchors: [ANCHOR] });
    expect(v.some((x) => x.type === "anchor_image_visible")).toBe(true);
  });

  it("flags anchor_link_visible when href references anchor product URL", () => {
    const html = `<a href="/product/abc123ef" class="group">leak</a>`;
    const v = auditHtml({ html, anchors: [ANCHOR] });
    expect(v.some((x) => x.type === "anchor_link_visible")).toBe(true);
  });

  it("flags anchor_attribute_visible on non-placeholder elements", () => {
    const html = `<button data-variant-id="anchor_id_xyz">click</button>`;
    const v = auditHtml({ html, anchors: [ANCHOR] });
    expect(v.some((x) => x.type === "anchor_attribute_visible")).toBe(true);
  });

  it("EXEMPTS data-variant-id on placeholders (aria-hidden=true)", () => {
    const html = `<div data-variant-id="anchor_id_xyz" aria-hidden="true">…</div>`;
    const v = auditHtml({ html, anchors: [ANCHOR] });
    expect(v.length).toBe(0);
  });

  it("multiplexes across multiple anchors", () => {
    const second: AnchorIdentity = {
      variantId: "second_id",
      urlHash: "ffffffff",
      primaryImage: "/images/vase/var_b_t2.png",
    };
    const html = `
      <img src="/images/chair/var_a_t1.png" />
      <a href="/product/ffffffff">also leak</a>
    `;
    const v = auditHtml({ html, anchors: [ANCHOR, second] });
    expect(v.some((x) => x.type === "anchor_image_visible")).toBe(true);
    expect(v.some((x) => x.type === "anchor_link_visible")).toBe(true);
  });
});
