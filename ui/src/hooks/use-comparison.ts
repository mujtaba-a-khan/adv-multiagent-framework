import { useQuery } from "@tanstack/react-query";
import { compareSessions } from "@/lib/api-client";

export function useComparison(
  experimentId: string,
  attackSessionId: string,
  defenseSessionId: string,
) {
  return useQuery({
    queryKey: ["comparison", experimentId, attackSessionId, defenseSessionId],
    queryFn: () =>
      compareSessions(experimentId, attackSessionId, defenseSessionId),
    enabled: !!experimentId && !!attackSessionId && !!defenseSessionId,
  });
}
