import { useQuery } from "@tanstack/react-query";
import { listTurns, getLatestTurn } from "@/lib/api-client";

export function useTurns(sessionId: string, offset = 0, limit = 100) {
  return useQuery({
    queryKey: ["turns", sessionId, offset, limit],
    queryFn: () => listTurns(sessionId, offset, limit),
    enabled: !!sessionId,
    refetchInterval: 5000,
  });
}

export function useLatestTurn(sessionId: string) {
  return useQuery({
    queryKey: ["latestTurn", sessionId],
    queryFn: () => getLatestTurn(sessionId),
    enabled: !!sessionId,
    refetchInterval: 3000,
  });
}
