"use client";

import { Badge } from "@/components/ui/badge";
import { CATEGORY_LABELS, CATEGORY_COLORS } from "@/lib/constants";
import type { Strategy } from "@/lib/types";

interface StrategyCardProps {
  strategy: Strategy;
  selected: boolean;
  onClick: () => void;
}

export function StrategyCard({ strategy, selected, onClick }: StrategyCardProps) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={`flex flex-col gap-2 rounded-lg border p-4 text-left transition-colors ${
        selected
          ? "border-primary bg-primary/5 ring-1 ring-primary"
          : "hover:bg-muted/50"
      }`}
    >
      <div className="flex items-center gap-2">
        <span className="text-sm font-medium">{strategy.display_name}</span>
      </div>
      <Badge
        variant="outline"
        className={`w-fit text-[10px] ${CATEGORY_COLORS[strategy.category] ?? ""}`}
      >
        {CATEGORY_LABELS[strategy.category] ?? strategy.category}
      </Badge>
      <p className="text-xs text-muted-foreground">
        {strategy.description}
      </p>
      <div className="flex flex-wrap gap-2">
        <span className="text-[10px] font-mono text-muted-foreground">
          ASR: {strategy.estimated_asr}
        </span>
        <span className="text-[10px] font-mono text-muted-foreground">
          {strategy.min_turns}
          {strategy.max_turns ? `–${strategy.max_turns}` : "+"} turns
        </span>
      </div>
    </button>
  );
}
