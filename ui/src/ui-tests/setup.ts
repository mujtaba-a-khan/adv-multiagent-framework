import "@testing-library/jest-dom/vitest";

// ── jsdom polyfills for Radix UI components ──────────────────

// scrollTo is used by auto-scroll logic
Element.prototype.scrollTo = (() => {}) as typeof Element.prototype.scrollTo;

// scrollIntoView is used by Radix Select
Element.prototype.scrollIntoView = (() => {}) as typeof Element.prototype.scrollIntoView;

// ResizeObserver is used by Radix Popover / Tooltip
globalThis.ResizeObserver = class ResizeObserver {
  observe() {}
  unobserve() {}
  disconnect() {}
} as unknown as typeof globalThis.ResizeObserver;
