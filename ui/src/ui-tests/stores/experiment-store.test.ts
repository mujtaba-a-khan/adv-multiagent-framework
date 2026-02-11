import { describe, it, expect, beforeEach } from "vitest";
import { useExperimentStore } from "@/stores/experiment-store";
import type { Experiment, Session } from "@/lib/types";

describe("useExperimentStore", () => {
  beforeEach(() => {
    useExperimentStore.setState({
      activeExperiment: null,
      activeSession: null,
    });
  });

  it("has correct initial state", () => {
    const state = useExperimentStore.getState();
    expect(state.activeExperiment).toBeNull();
    expect(state.activeSession).toBeNull();
  });

  it("setActiveExperiment updates state", () => {
    const exp = { id: "e1", name: "Test" } as Experiment;
    useExperimentStore.getState().setActiveExperiment(exp);
    expect(useExperimentStore.getState().activeExperiment).toEqual(exp);
  });

  it("setActiveExperiment can clear with null", () => {
    const exp = { id: "e1", name: "Test" } as Experiment;
    useExperimentStore.getState().setActiveExperiment(exp);
    useExperimentStore.getState().setActiveExperiment(null);
    expect(useExperimentStore.getState().activeExperiment).toBeNull();
  });

  it("setActiveSession updates state", () => {
    const session = { id: "s1", status: "running" } as Session;
    useExperimentStore.getState().setActiveSession(session);
    expect(useExperimentStore.getState().activeSession).toEqual(session);
  });

  it("setActiveSession can clear with null", () => {
    const session = { id: "s1", status: "running" } as Session;
    useExperimentStore.getState().setActiveSession(session);
    useExperimentStore.getState().setActiveSession(null);
    expect(useExperimentStore.getState().activeSession).toBeNull();
  });
});
