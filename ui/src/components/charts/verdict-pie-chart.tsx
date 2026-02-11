"use client";

import { Cell, Pie, PieChart } from "recharts";
import {
  ChartContainer,
  ChartTooltip,
  ChartTooltipContent,
  type ChartConfig,
} from "@/components/ui/chart";

interface VerdictPieChartProps {
  jailbreaks: number;
  refused: number;
  blocked: number;
  borderline?: number;
}

const COLORS = {
  jailbreak: "hsl(0, 84%, 60%)",
  refused: "hsl(160, 84%, 39%)",
  blocked: "hsl(220, 84%, 60%)",
  borderline: "hsl(38, 92%, 50%)",
};

const chartConfig = {
  jailbreak: { label: "Jailbreak", color: COLORS.jailbreak },
  refused: { label: "Refused", color: COLORS.refused },
  blocked: { label: "Blocked", color: COLORS.blocked },
  borderline: { label: "Borderline", color: COLORS.borderline },
} satisfies ChartConfig;

export function VerdictPieChart({
  jailbreaks,
  refused,
  blocked,
  borderline = 0,
}: VerdictPieChartProps) {
  const data = [
    { name: "Jailbreak", value: jailbreaks, fill: COLORS.jailbreak },
    { name: "Refused", value: refused, fill: COLORS.refused },
    { name: "Blocked", value: blocked, fill: COLORS.blocked },
    ...(borderline > 0
      ? [{ name: "Borderline", value: borderline, fill: COLORS.borderline }]
      : []),
  ].filter((d) => d.value > 0);

  if (data.length === 0) {
    return (
      <div className="flex h-full items-center justify-center text-sm text-muted-foreground">
        No verdict data
      </div>
    );
  }

  return (
    <ChartContainer config={chartConfig} className="mx-auto aspect-square max-h-[250px]">
      <PieChart>
        <ChartTooltip content={<ChartTooltipContent />} />
        <Pie
          data={data}
          dataKey="value"
          nameKey="name"
          cx="50%"
          cy="50%"
          innerRadius={50}
          outerRadius={80}
          strokeWidth={2}
          stroke="hsl(var(--background))"
        >
          {data.map((entry) => (
            <Cell key={entry.name} fill={entry.fill} />
          ))}
        </Pie>
      </PieChart>
    </ChartContainer>
  );
}
