import { describe, expect, it, vi, beforeEach, afterEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";

vi.mock("next/navigation", () => ({
  useRouter: () => ({ push: vi.fn(), refresh: vi.fn() }),
  usePathname: () => "/category/sofas",
}));

import { AnnotateOverlay } from "@/components/annotate/annotate-overlay";

const fetchMock = vi.fn();

beforeEach(() => {
  fetchMock.mockReset();
  (globalThis as { fetch: typeof fetch }).fetch =
    fetchMock as unknown as typeof fetch;
});

afterEach(() => {
  vi.restoreAllMocks();
});

function snapshot(overrides: Record<string, unknown> = {}) {
  return {
    attemptId: "att1",
    subTask: "NC",
    taskId: "nc_test",
    targetUrl: "/product/B",
    acceptedAlternatives: [],
    totalTurns: 6,
    currentTurn: 3,
    currentTurnData: {
      turn_index: 3,
      role: "user",
      mode: "encoding",
      content: "Show me this minimalist one.",
      is_rejection: false,
      references_variant: "A",
      expected_url: null,
    },
    isRecall: false,
    isAtTarget: false,
    canSubmit: false,
    ...overrides,
  };
}

function mockAttempt(snap: ReturnType<typeof snapshot>) {
  fetchMock.mockImplementation((url: string) => {
    if (url.startsWith("/api/annotate/attempt/current")) {
      return Promise.resolve(
        new Response(JSON.stringify(snap), { status: 200 }),
      );
    }
    return Promise.resolve(new Response("{}", { status: 200 }));
  });
}

describe("AnnotateOverlay", () => {
  it("renders nothing when no active attempt", async () => {
    fetchMock.mockResolvedValue(new Response("{}", { status: 404 }));
    const { container } = render(<AnnotateOverlay />);
    await new Promise((r) => setTimeout(r, 0));
    expect(container.querySelector("[data-testid=annotate-overlay]")).toBeNull();
  });

  it("shows Continue + customer bubble on encoding turn", async () => {
    mockAttempt(snapshot());
    render(<AnnotateOverlay />);
    await waitFor(() => {
      expect(screen.getByTestId("annotate-overlay")).toBeInTheDocument();
    });
    expect(screen.getByText(/Show me this minimalist one/)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /Continue/ })).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /Submit answer/ })).toBeNull();
  });

  it("shows Submit (disabled) + Give up on recall turn when not on submittable page", async () => {
    mockAttempt(
      snapshot({
        currentTurnData: {
          turn_index: 5,
          role: "agent",
          mode: "recall",
          content: null,
          is_rejection: false,
          references_variant: null,
          expected_url: null,
        },
        isRecall: true,
        isAtTarget: false,
        canSubmit: false,
      }),
    );
    render(<AnnotateOverlay />);
    await waitFor(() => {
      expect(screen.getByTestId("annotate-overlay")).toBeInTheDocument();
    });
    expect(screen.getByRole("button", { name: /Submit answer/ })).toBeDisabled();
    expect(screen.getByRole("button", { name: /Give up/ })).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /^Continue$/ })).toBeNull();
  });

  it("enables Submit when canSubmit=true (worker is on a /product or /collection page)", async () => {
    mockAttempt(
      snapshot({
        currentTurnData: {
          turn_index: 5,
          role: "agent",
          mode: "recall",
          content: null,
          is_rejection: false,
          references_variant: null,
          expected_url: null,
        },
        isRecall: true,
        isAtTarget: false,    // not on target...
        canSubmit: true,      // but on a submittable page — Submit enables anyway
      }),
    );
    render(<AnnotateOverlay />);
    await waitFor(() => {
      expect(screen.getByRole("button", { name: /Submit answer/ })).not.toBeDisabled();
    });
  });
});
