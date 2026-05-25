import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import { ModeProvider } from "@/components/mode-provider";
import { WishlistView } from "@/components/wishlist-view";

const VARIANTS = [
  {
    variantId: "v_anchor",
    urlHash: "aaa11111",
    title: "Modern Accent Chair",
    altText: "Modern Accent Chair",
    primaryImage: "/images/chair/var_a_t1.png",
    price: 249,
  },
  {
    variantId: "v_other",
    urlHash: "bbb22222",
    title: "Studio Table Lamp",
    altText: "Studio Table Lamp",
    primaryImage: "/images/lamp/var_b_t1.png",
    price: 129,
  },
];

beforeEach(() => {
  localStorage.setItem("dmv_wishlist", JSON.stringify(["aaa11111", "bbb22222"]));
  // Stub fetch — WishlistView calls /api/variants?ids=…
  vi.stubGlobal(
    "fetch",
    vi.fn(async () => ({
      ok: true,
      json: async () => ({ variants: VARIANTS }),
    })) as unknown as typeof fetch,
  );
});

afterEach(() => {
  localStorage.clear();
  vi.unstubAllGlobals();
});

describe("WishlistView", () => {
  it("renders all wishlist items with full thumbnails in encoding mode", async () => {
    render(
      <ModeProvider value={{ mode: "encoding", anchorVariantIds: new Set(), taskId: null }}>
        <WishlistView />
      </ModeProvider>,
    );
    await waitFor(() => expect(screen.getByTestId("wishlist-view")).toBeInTheDocument());
    expect(screen.getByText("Modern Accent Chair")).toBeInTheDocument();
    expect(screen.getByText("Studio Table Lamp")).toBeInTheDocument();
    expect(screen.queryByTestId("wishlist-anchor-placeholder")).toBeNull();
    // both thumbnails rendered
    const imgs = screen.getAllByRole("img");
    expect(imgs.length).toBe(2);
  });

  it("replaces anchor thumbnails with placeholder in recall mode, keeps title", async () => {
    render(
      <ModeProvider value={{ mode: "recall", anchorVariantIds: new Set(["v_anchor"]), taskId: null }}>
        <WishlistView />
      </ModeProvider>,
    );
    await waitFor(() => expect(screen.getByTestId("wishlist-view")).toBeInTheDocument());
    const placeholder = screen.getByTestId("wishlist-anchor-placeholder");
    expect(placeholder.getAttribute("data-variant-id")).toBe("v_anchor");
    // Title preserved (L2-safe by contract)
    expect(screen.getByText("Modern Accent Chair")).toBeInTheDocument();
    // Anchor image gone
    const imgs = screen.queryAllByRole("img");
    expect(imgs.length).toBe(1);
    expect(imgs[0].getAttribute("src")).not.toContain("var_a_t1");
    // Anchor price hidden (revealing $249 would re-identify the anchor)
    expect(screen.queryByText("$249.00")).toBeNull();
    // Non-anchor still has its price
    expect(screen.getByText("$129.00")).toBeInTheDocument();
  });
});
