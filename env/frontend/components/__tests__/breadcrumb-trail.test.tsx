import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import { BreadcrumbTrail, type Crumb } from "@/components/breadcrumb-trail";

const crumbs: Crumb[] = [
  { label: "Home", href: "/" },
  {
    label: "Chairs",
    href: "/category/chairs",
    thumbnail: "/images/chair/var_a_t1.png",
    thumbnailAlt: "Chairs preview",
  },
  { label: "Modern" },
];

describe("BreadcrumbTrail", () => {
  it("renders crumb thumbnail in encoding mode", () => {
    const { container } = render(
      <BreadcrumbTrail mode="encoding" crumbs={crumbs} />,
    );
    expect(screen.getByText("Home")).toBeInTheDocument();
    expect(screen.getByText("Chairs")).toBeInTheDocument();
    expect(screen.getByText("Modern")).toBeInTheDocument();
    const thumb = container.querySelector(
      '[data-testid="breadcrumb-thumbnail"]',
    );
    expect(thumb).not.toBeNull();
    expect(container.querySelectorAll("img").length).toBe(1);
    const trail = container.querySelector('[data-testid="breadcrumb-trail"]');
    expect(trail?.getAttribute("data-mode")).toBe("encoding");
  });

  it("strips thumbnails in recall mode but keeps labels", () => {
    const { container } = render(
      <BreadcrumbTrail mode="recall" crumbs={crumbs} />,
    );
    // Labels still readable — labels themselves are L2-safe text by design.
    expect(screen.getByText("Home")).toBeInTheDocument();
    expect(screen.getByText("Chairs")).toBeInTheDocument();
    expect(screen.getByText("Modern")).toBeInTheDocument();
    // No thumbnail rendered.
    expect(
      container.querySelector('[data-testid="breadcrumb-thumbnail"]'),
    ).toBeNull();
    expect(container.querySelectorAll("img").length).toBe(0);
    const trail = container.querySelector('[data-testid="breadcrumb-trail"]');
    expect(trail?.getAttribute("data-mode")).toBe("recall");
  });

  it("renders cleanly when no crumb carries a thumbnail (no-op recall)", () => {
    const plain: Crumb[] = [{ label: "Home", href: "/" }, { label: "Chairs" }];
    const enc = render(<BreadcrumbTrail mode="encoding" crumbs={plain} />);
    expect(enc.container.querySelectorAll("img").length).toBe(0);
    enc.unmount();
    const rec = render(<BreadcrumbTrail mode="recall" crumbs={plain} />);
    expect(rec.container.querySelectorAll("img").length).toBe(0);
  });
});
