import { useQuery } from "@tanstack/react-query";
import { listStrategies, getStrategy } from "@/lib/api-client";

export function useStrategies() {
  return useQuery({
    queryKey: ["strategies"],
    queryFn: listStrategies,
    staleTime: 5 * 60 * 1000, // strategies rarely change
  });
}

export function useStrategy(name: string) {
  return useQuery({
    queryKey: ["strategy", name],
    queryFn: () => getStrategy(name),
    enabled: !!name,
    staleTime: 5 * 60 * 1000,
  });
}
