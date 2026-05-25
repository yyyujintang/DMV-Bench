import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import { ProductCard, type ProductCardData } from "@/components/product-card";

const sample: ProductCardData = {
  variantId: "v_anchor",
  urlHash: "abc123ef",
  title: "Modern Accent Chair",
  altText: "Modern Accent Chair",
  primaryImage: "/images/chair/var_a_t1.png",
  price: 249,
};

describe("ProductCard", () => {
  it("renders link, image, and price in default (no-anchor) mode", () => {
    const { container } = render(<ProductCard data={sample} />);
    const link = container.querySelector('a[href="/product/abc123ef"]');
    expect(link).not.toBeNull();
    const img = container.querySelector("img");
    expect(img).not.toBeNull();
    expect(img!.getAttribute("alt")).toBe("Modern Accent Chair");
    expect(screen.getByText("$249.00")).toBeInTheDocument();
    expect(
      container.querySelector('[data-testid="product-card-placeholder"]'),
    ).toBeNull();
  });

  it("renders an inert placeholder (no link, no img) when variant is an anchor in recall", () => {
    const anchors = new Set(["v_anchor"]);
    const { container } = render(
      <ProductCard data={sample} anchorVariantIds={anchors} />,
    );
    expect(container.querySelectorAll("a").length).toBe(0);
    expect(container.querySelectorAll("img").length).toBe(0);
    // Product title text must also be suppressed — leaking it would
    // identify the anchor even without the image.
    expect(screen.queryByText("Modern Accent Chair")).toBeNull();
    expect(screen.queryByText("$249.00")).toBeNull();
    const placeholder = container.querySelector(
      '[data-testid="product-card-placeholder"]',
    );
    expect(placeholder).not.toBeNull();
    expect(placeholder!.getAttribute("data-variant-id")).toBe("v_anchor");
    expect(placeholder!.getAttribute("aria-hidden")).toBe("true");
  });

  it("renders normally when the variant is NOT in the anchor set (recall, non-target)", () => {
    const anchors = new Set(["some_other_id"]);
    const { container } = render(
      <ProductCard data={sample} anchorVariantIds={anchors} />,
    );
    expect(container.querySelector('a[href="/product/abc123ef"]')).not.toBeNull();
    expect(container.querySelector("img")).not.toBeNull();
  });
});
