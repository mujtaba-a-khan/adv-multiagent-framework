import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import {
  listSessions,
  getSession,
  createSession,
  startSession,
  deleteSession,
} from "@/lib/api-client";
import type { StartSessionRequest } from "@/lib/types";

export function useSessions(experimentId: string, offset = 0, limit = 50) {
  return useQuery({
    queryKey: ["sessions", experimentId, offset, limit],
    queryFn: () => listSessions(experimentId, offset, limit),
    enabled: !!experimentId,
  });
}

export function useSession(experimentId: string, sessionId: string) {
  return useQuery({
    queryKey: ["session", experimentId, sessionId],
    queryFn: () => getSession(experimentId, sessionId),
    enabled: !!experimentId && !!sessionId,
    refetchInterval: (query) => {
      const status = query.state.data?.status;
      return status === "running" || status === "pending" ? 3000 : false;
    },
  });
}

export function useCreateSession(experimentId: string) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (data?: StartSessionRequest) =>
      createSession(experimentId, data),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["sessions", experimentId] });
    },
  });
}

export function useStartSession(experimentId: string) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (sessionId: string) => startSession(experimentId, sessionId),
    onSuccess: (_, sessionId) => {
      qc.invalidateQueries({ queryKey: ["sessions", experimentId] });
      qc.invalidateQueries({
        queryKey: ["session", experimentId, sessionId],
      });
    },
  });
}

export function useDeleteSession(experimentId: string) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (sessionId: string) =>
      deleteSession(experimentId, sessionId),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["sessions", experimentId] });
    },
  });
}
