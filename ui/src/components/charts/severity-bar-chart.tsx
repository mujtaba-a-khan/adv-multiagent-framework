"use client";

import { Bar, BarChart, CartesianGrid, XAxis, YAxis } from "recharts";
import {
  ChartContainer,
  ChartTooltip,
  ChartTooltipContent,
  type ChartConfig,
} from "@/components/ui/chart";
import type { ReportFinding } from "@/lib/types";

interface SeverityBarChartProps {
  findings: ReportFinding[];
}

const chartConfig = {
  severity: { label: "Severity", color: "hsl(0, 84%, 60%)" },
  specificity: { label: "Specificity", color: "hsl(38, 92%, 50%)" },
} satisfies ChartConfig;

export function SeverityBarChart({ findings }: SeverityBarChartProps) {
  const data = findings.map((f) => ({
    turn: `T${f.turn_number}`,
    severity: f.severity,
    specificity: f.specificity,
    strategy: f.strategy_name,
  }));

  if (data.length === 0) {
    return (
      <div className="flex h-full items-center justify-center text-sm text-muted-foreground">
        No findings to display
      </div>
    );
  }

  return (
    <ChartContainer config={chartConfig} className="h-[250px] w-full">
      <BarChart data={data} barGap={2}>
        <CartesianGrid strokeDasharray="3 3" vertical={false} />
        <XAxis dataKey="turn" fontSize={12} tickLine={false} axisLine={false} />
        <YAxis domain={[0, 10]} fontSize={12} tickLine={false} axisLine={false} />
        <ChartTooltip content={<ChartTooltipContent />} />
        <Bar
          dataKey="severity"
          fill="var(--color-severity)"
          radius={[4, 4, 0, 0]}
        />
        <Bar
          dataKey="specificity"
          fill="var(--color-specificity)"
          radius={[4, 4, 0, 0]}
        />
      </BarChart>
    </ChartContainer>
  );
}
