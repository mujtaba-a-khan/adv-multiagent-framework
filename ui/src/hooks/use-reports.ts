import { useQuery } from "@tanstack/react-query";
import { getSessionReport, listSessions, listExperiments } from "@/lib/api-client";

export function useSessionReport(experimentId: string, sessionId: string) {
  return useQuery({
    queryKey: ["report", experimentId, sessionId],
    queryFn: () => getSessionReport(experimentId, sessionId),
    enabled: !!experimentId && !!sessionId,
    staleTime: 60_000,
  });
}

export function useCompletedSessions() {
  return useQuery({
    queryKey: ["completed-sessions"],
    queryFn: async () => {
      const expRes = await listExperiments(0, 100);
      const allSessions = await Promise.all(
        expRes.experiments.map(async (exp) => {
          const sessRes = await listSessions(exp.id, 0, 100);
          return sessRes.sessions
            .filter((s) => s.status === "completed")
            .map((s) => ({ ...s, experiment: exp }));
        }),
      );
      return allSessions.flat();
    },
    staleTime: 30_000,
  });
}
