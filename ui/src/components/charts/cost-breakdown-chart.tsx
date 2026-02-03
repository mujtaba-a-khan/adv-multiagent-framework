"use client";

import { Bar, BarChart, CartesianGrid, XAxis, YAxis } from "recharts";
import {
  ChartContainer,
  ChartTooltip,
  ChartTooltipContent,
  ChartLegend,
  ChartLegendContent,
  type ChartConfig,
} from "@/components/ui/chart";
import type { Session } from "@/lib/types";

interface CostBreakdownChartProps {
  session: Session;
}

const chartConfig = {
  attacker: { label: "Attacker", color: "hsl(0, 84%, 60%)" },
  target: { label: "Target", color: "hsl(220, 84%, 60%)" },
  analyzer: { label: "Analyzer", color: "hsl(160, 84%, 39%)" },
  defender: { label: "Defender", color: "hsl(280, 84%, 60%)" },
} satisfies ChartConfig;

export function CostBreakdownChart({ session }: CostBreakdownChartProps) {
  const data = [
    {
      agent: "Token Usage",
      attacker: session.total_attacker_tokens,
      target: session.total_target_tokens,
      analyzer: session.total_analyzer_tokens,
      defender: session.total_defender_tokens,
    },
  ];

  const total =
    session.total_attacker_tokens +
    session.total_target_tokens +
    session.total_analyzer_tokens +
    session.total_defender_tokens;

  if (total === 0) {
    return (
      <div className="flex h-full items-center justify-center text-sm text-muted-foreground">
        No token usage data
      </div>
    );
  }

  return (
    <ChartContainer config={chartConfig} className="h-[200px] w-full">
      <BarChart data={data} layout="vertical" barSize={32}>
        <CartesianGrid strokeDasharray="3 3" horizontal={false} />
        <XAxis type="number" fontSize={12} tickLine={false} axisLine={false} />
        <YAxis
          type="category"
          dataKey="agent"
          fontSize={12}
          tickLine={false}
          axisLine={false}
          width={80}
        />
        <ChartTooltip content={<ChartTooltipContent />} />
        <ChartLegend content={<ChartLegendContent />} />
        <Bar dataKey="attacker" stackId="a" fill="var(--color-attacker)" radius={[0, 0, 0, 0]} />
        <Bar dataKey="target" stackId="a" fill="var(--color-target)" />
        <Bar dataKey="analyzer" stackId="a" fill="var(--color-analyzer)" />
        <Bar dataKey="defender" stackId="a" fill="var(--color-defender)" radius={[0, 4, 4, 0]} />
      </BarChart>
    </ChartContainer>
  );
}
