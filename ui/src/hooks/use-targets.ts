import { useQuery } from "@tanstack/react-query";
import { listModels, checkHealth } from "@/lib/api-client";

export function useModels() {
  return useQuery({
    queryKey: ["models"],
    queryFn: listModels,
    staleTime: 60_000,
  });
}

export function useHealthCheck() {
  return useQuery({
    queryKey: ["health"],
    queryFn: checkHealth,
    refetchInterval: 30_000,
  });
}
