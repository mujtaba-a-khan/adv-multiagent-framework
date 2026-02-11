import { renderHook, act } from "@testing-library/react";
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { useIsMobile } from "@/hooks/use-mobile";

describe("useIsMobile", () => {
  let listeners: Array<() => void>;
  let mockMatchMedia: ReturnType<typeof vi.fn>;

  beforeEach(() => {
    listeners = [];
    mockMatchMedia = vi.fn().mockReturnValue({
      addEventListener: (_: string, cb: () => void) => listeners.push(cb),
      removeEventListener: vi.fn(),
    });
    Object.defineProperty(window, "matchMedia", {
      writable: true,
      value: mockMatchMedia,
    });
  });

  afterEach(() => vi.restoreAllMocks());

  it("returns false for desktop width", () => {
    Object.defineProperty(window, "innerWidth", {
      writable: true,
      value: 1024,
    });
    const { result } = renderHook(() => useIsMobile());
    expect(result.current).toBe(false);
  });

  it("returns true for mobile width", () => {
    Object.defineProperty(window, "innerWidth", {
      writable: true,
      value: 500,
    });
    const { result } = renderHook(() => useIsMobile());
    expect(result.current).toBe(true);
  });

  it("reacts to resize via media query listener", () => {
    Object.defineProperty(window, "innerWidth", {
      writable: true,
      value: 1024,
    });
    const { result } = renderHook(() => useIsMobile());
    expect(result.current).toBe(false);

    act(() => {
      Object.defineProperty(window, "innerWidth", {
        writable: true,
        value: 500,
      });
      for (const cb of listeners) cb();
    });
    expect(result.current).toBe(true);
  });
});
