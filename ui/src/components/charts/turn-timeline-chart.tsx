"use client";

import { Area, AreaChart, CartesianGrid, XAxis, YAxis } from "recharts";
import {
  ChartContainer,
  ChartTooltip,
  ChartTooltipContent,
  type ChartConfig,
} from "@/components/ui/chart";
import type { Turn } from "@/lib/types";

interface TurnTimelineChartProps {
  turns: Turn[];
}

const chartConfig = {
  confidence: { label: "Judge Confidence", color: "hsl(220, 84%, 60%)" },
  severity: { label: "Severity", color: "hsl(0, 84%, 60%)" },
} satisfies ChartConfig;

export function TurnTimelineChart({ turns }: TurnTimelineChartProps) {
  const data = turns.map((t) => ({
    turn: t.turn_number,
    confidence: Math.round((t.judge_confidence ?? 0) * 100) / 100,
    severity: t.severity_score ?? 0,
    verdict: t.judge_verdict,
  }));

  if (data.length === 0) {
    return (
      <div className="flex h-full items-center justify-center text-sm text-muted-foreground">
        No turn data
      </div>
    );
  }

  return (
    <ChartContainer config={chartConfig} className="h-[250px] w-full">
      <AreaChart data={data}>
        <CartesianGrid strokeDasharray="3 3" vertical={false} />
        <XAxis
          dataKey="turn"
          fontSize={12}
          tickLine={false}
          axisLine={false}
          label={{ value: "Turn", position: "insideBottom", offset: -5 }}
        />
        <YAxis domain={[0, 10]} fontSize={12} tickLine={false} axisLine={false} />
        <ChartTooltip content={<ChartTooltipContent />} />
        <Area
          type="monotone"
          dataKey="severity"
          fill="var(--color-severity)"
          fillOpacity={0.2}
          stroke="var(--color-severity)"
          strokeWidth={2}
        />
        <Area
          type="monotone"
          dataKey="confidence"
          fill="var(--color-confidence)"
          fillOpacity={0.1}
          stroke="var(--color-confidence)"
          strokeWidth={2}
          strokeDasharray="4 4"
        />
      </AreaChart>
    </ChartContainer>
  );
}
