import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import { ModeProvider } from "@/components/mode-provider";
import { RecentlyViewed } from "@/components/recently-viewed";

const ITEMS = [
  {
    variantId: "v_anchor",
    urlHash: "aaa11111",
    title: "Modern Accent Chair",
    altText: "Modern Accent Chair",
    primaryImage: "/images/chair/var_a_t1.png",
    viewedAt: 1000,
  },
  {
    variantId: "v_other_1",
    urlHash: "bbb22222",
    title: "Studio Table Lamp",
    altText: "Studio Table Lamp",
    primaryImage: "/images/lamp/var_b_t1.png",
    viewedAt: 900,
  },
  {
    variantId: "v_other_2",
    urlHash: "ccc33333",
    title: "Modern Vase",
    altText: "Modern Vase",
    primaryImage: "/images/vase/var_c_t1.png",
    viewedAt: 800,
  },
];

beforeEach(() => {
  localStorage.setItem("dmv_recently_viewed", JSON.stringify(ITEMS));
});

afterEach(() => {
  localStorage.clear();
});

describe("RecentlyViewed", () => {
  it("renders all items in encoding mode (after hydration)", async () => {
    render(
      <ModeProvider value={{ mode: "encoding", anchorVariantIds: new Set(), taskId: null }}>
        <RecentlyViewed />
      </ModeProvider>,
    );
    await waitFor(() =>
      expect(screen.getByTestId("recently-viewed")).toBeInTheDocument(),
    );
    expect(screen.getAllByTestId("recently-viewed-item")).toHaveLength(3);
    expect(screen.getByText("Modern Accent Chair")).toBeInTheDocument();
  });

  it("filters anchors out of the list in recall mode", async () => {
    render(
      <ModeProvider value={{ mode: "recall", anchorVariantIds: new Set(["v_anchor"]), taskId: null }}>
        <RecentlyViewed />
      </ModeProvider>,
    );
    await waitFor(() =>
      expect(screen.getByTestId("recently-viewed")).toBeInTheDocument(),
    );
    const items = screen.getAllByTestId("recently-viewed-item");
    expect(items).toHaveLength(2);
    expect(screen.queryByText("Modern Accent Chair")).toBeNull();
    expect(screen.getByText("Studio Table Lamp")).toBeInTheDocument();
  });

  it("collapses entirely when every history item is an anchor", async () => {
    render(
      <ModeProvider
        value={{
          mode: "recall",
          anchorVariantIds: new Set(["v_anchor", "v_other_1", "v_other_2"]),
          taskId: null,
        }}
      >
        <RecentlyViewed />
      </ModeProvider>,
    );
    // Wait a tick for the hydration effect to run.
    await new Promise((r) => setTimeout(r, 30));
    expect(screen.queryByTestId("recently-viewed")).toBeNull();
  });
});
