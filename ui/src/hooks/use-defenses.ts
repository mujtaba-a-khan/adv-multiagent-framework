import { useQuery } from "@tanstack/react-query";
import { listDefenses } from "@/lib/api-client";

export function useDefenses() {
  return useQuery({
    queryKey: ["defenses"],
    queryFn: listDefenses,
  });
}
