"use client";

import { useState } from "react";
import {
  ChevronDown,
  ExternalLink,
  Filter,
  Search,
  Swords,
} from "lucide-react";
import { Header } from "@/components/layout/header";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import {
  DropdownMenu,
  DropdownMenuCheckboxItem,
  DropdownMenuContent,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { Input } from "@/components/ui/input";
import { Separator } from "@/components/ui/separator";
import { Skeleton } from "@/components/ui/skeleton";
import { useStrategies } from "@/hooks/use-strategies";
import {
  CATEGORY_LABELS,
  CATEGORY_COLORS,
} from "@/lib/constants";
import type { Strategy, StrategyCategory } from "@/lib/types";

const ALL_CATEGORIES: StrategyCategory[] = [
  "prompt_level",
  "optimization",
  "multi_turn",
  "advanced",
  "composite",
];

export default function StrategiesPage() {
  const { data, isLoading } = useStrategies();
  const [search, setSearch] = useState("");
  const [categoryFilter, setCategoryFilter] = useState<Set<StrategyCategory>>(
    new Set(),
  );

  const strategies = data?.strategies ?? [];

  const filtered = strategies.filter((s) => {
    if (categoryFilter.size > 0 && !categoryFilter.has(s.category))
      return false;
    if (search) {
      const q = search.toLowerCase();
      return (
        s.display_name.toLowerCase().includes(q) ||
        s.description.toLowerCase().includes(q) ||
        s.name.toLowerCase().includes(q)
      );
    }
    return true;
  });

  const grouped = ALL_CATEGORIES.reduce(
    (acc, cat) => {
      const items = filtered.filter((s) => s.category === cat);
      if (items.length > 0) acc[cat] = items;
      return acc;
    },
    {} as Record<StrategyCategory, Strategy[]>,
  );

  const toggleCategory = (cat: StrategyCategory) => {
    setCategoryFilter((prev) => {
      const next = new Set(prev);
      if (next.has(cat)) next.delete(cat);
      else next.add(cat);
      return next;
    });
  };

  return (
    <>
      <Header title="Strategies" />

      <div className="flex flex-1 flex-col gap-6 p-6">
        <div>
          <h2 className="text-2xl font-bold tracking-tight">
            Attack Strategy Catalog
          </h2>
          <p className="text-muted-foreground">
            {strategies.length} strategies across {ALL_CATEGORIES.length}{" "}
            categories. Each strategy implements a different adversarial
            technique.
          </p>
        </div>

        <div className="flex items-center gap-3">
          <div className="relative flex-1 max-w-sm">
            <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
            <Input
              placeholder="Search strategies..."
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              className="pl-9"
            />
          </div>
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button variant="outline" size="sm">
                <Filter className="mr-2 h-4 w-4" />
                Category
                {categoryFilter.size > 0 && (
                  <Badge
                    variant="secondary"
                    className="ml-2 h-5 w-5 rounded-full p-0 text-xs"
                  >
                    {categoryFilter.size}
                  </Badge>
                )}
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent>
              {ALL_CATEGORIES.map((cat) => (
                <DropdownMenuCheckboxItem
                  key={cat}
                  checked={categoryFilter.has(cat)}
                  onCheckedChange={() => toggleCategory(cat)}
                >
                  {CATEGORY_LABELS[cat]}
                </DropdownMenuCheckboxItem>
              ))}
            </DropdownMenuContent>
          </DropdownMenu>
        </div>

        {isLoading && (
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
            {["s1", "s2", "s3", "s4", "s5", "s6"].map((id) => (
              <Skeleton key={id} className="h-40" />
            ))}
          </div>
        )}
        {!isLoading && filtered.length === 0 && (
          <div className="flex flex-col items-center justify-center py-16 text-center">
            <Swords className="h-10 w-10 text-muted-foreground" />
            <p className="mt-3 text-sm text-muted-foreground">
              No strategies match your filters.
            </p>
          </div>
        )}
        {!isLoading && filtered.length > 0 && (
          <div className="space-y-8">
            {Object.entries(grouped).map(([cat, items]) => (
              <section key={cat}>
                <div className="mb-4 flex items-center gap-2">
                  <Badge
                    variant="outline"
                    className={`${CATEGORY_COLORS[cat] ?? ""}`}
                  >
                    {CATEGORY_LABELS[cat]}
                  </Badge>
                  <span className="text-xs text-muted-foreground">
                    {items.length} strateg{items.length === 1 ? "y" : "ies"}
                  </span>
                </div>
                <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
                  {items.map((s) => (
                    <StrategyDetailCard key={s.name} strategy={s} />
                  ))}
                </div>
              </section>
            ))}
          </div>
        )}
      </div>
    </>
  );
}

function StrategyDetailCard({ strategy }: Readonly<{ strategy: Strategy }>) {
  return (
    <Collapsible>
      <Card>
        <CardHeader className="pb-3">
          <div className="flex items-start justify-between">
            <div>
              <CardTitle className="text-sm">
                {strategy.display_name}
              </CardTitle>
              <CardDescription className="mt-1 text-xs">
                {strategy.description}
              </CardDescription>
            </div>
          </div>
        </CardHeader>
        <CardContent>
          <div className="flex flex-wrap gap-2 mb-3">
            <Badge variant="outline" className="text-[10px] font-mono">
              ASR: {strategy.estimated_asr}
            </Badge>
            {strategy.supports_multi_turn && (
              <Badge variant="outline" className="text-[10px]">
                Multi-turn
              </Badge>
            )}
            {strategy.requires_white_box && (
              <Badge
                variant="outline"
                className="text-[10px] text-amber-500 border-amber-500/20"
              >
                White-box
              </Badge>
            )}
            <Badge variant="outline" className="text-[10px]">
              {strategy.min_turns}
              {strategy.max_turns ? `–${strategy.max_turns}` : "+"} turns
            </Badge>
          </div>

          <CollapsibleTrigger asChild>
            <Button variant="ghost" size="sm" className="w-full">
              <ChevronDown className="mr-2 h-3 w-3" />
              Details
            </Button>
          </CollapsibleTrigger>
          <CollapsibleContent className="mt-3 space-y-3">
            <Separator />
            {strategy.references.length > 0 && (
              <div>
                <p className="text-xs font-medium mb-1">References</p>
                <div className="space-y-1">
                  {strategy.references.map((ref) => (
                    <a
                      key={ref}
                      href={ref}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="flex items-center gap-1 text-xs text-primary hover:underline"
                    >
                      <ExternalLink className="h-3 w-3" />
                      {ref}
                    </a>
                  ))}
                </div>
              </div>
            )}
            {Object.keys(strategy.parameters).length > 0 && (
              <div>
                <p className="text-xs font-medium mb-1">Parameters</p>
                <pre className="rounded bg-muted p-2 text-xs overflow-x-auto">
                  {JSON.stringify(strategy.parameters, null, 2)}
                </pre>
              </div>
            )}
          </CollapsibleContent>
        </CardContent>
      </Card>
    </Collapsible>
  );
}
