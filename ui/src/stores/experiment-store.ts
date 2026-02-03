import { create } from "zustand";
import type { Experiment, Session } from "@/lib/types";

interface ExperimentStore {
  /** Currently viewed experiment */
  activeExperiment: Experiment | null;
  setActiveExperiment: (exp: Experiment | null) => void;

  /** Currently active session (live battle) */
  activeSession: Session | null;
  setActiveSession: (session: Session | null) => void;
}

export const useExperimentStore = create<ExperimentStore>((set) => ({
  activeExperiment: null,
  setActiveExperiment: (exp) => set({ activeExperiment: exp }),

  activeSession: null,
  setActiveSession: (session) => set({ activeSession: session }),
}));
