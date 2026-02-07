"use client";

import { useState } from "react";
import { Loader2, Shield, Swords } from "lucide-react";

import { StrategyCard } from "@/components/strategy-card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Separator } from "@/components/ui/separator";
import { Switch } from "@/components/ui/switch";
import { useDefenses } from "@/hooks/use-defenses";
import { useStrategies } from "@/hooks/use-strategies";
import type { DefenseSelectionItem, SessionMode } from "@/lib/types";

interface LaunchSessionModalProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onLaunch: (
    mode: SessionMode,
    defenses: DefenseSelectionItem[],
    strategyName: string,
    maxTurns: number,
    maxCostUsd: number,
    separateReasoning: boolean,
  ) => void;
  isPending?: boolean;
}

export function LaunchSessionModal({
  open,
  onOpenChange,
  onLaunch,
  isPending = false,
}: LaunchSessionModalProps) {
  const [mode, setMode] = useState<SessionMode>("attack");
  const [selectedDefenses, setSelectedDefenses] = useState<Set<string>>(
    new Set(),
  );
  const [strategyName, setStrategyName] = useState<string>("");
  const [maxTurns, setMaxTurns] = useState<number | "">("");
  const [maxCostUsd, setMaxCostUsd] = useState<number | "">("");
  const [separateReasoning, setSeparateReasoning] = useState<boolean>(true);

  const { data: defensesData, isLoading: defensesLoading } = useDefenses();
  const { data: strategiesData, isLoading: strategiesLoading } = useStrategies();

  const defenses = defensesData?.defenses ?? [];
  const strategies = strategiesData?.strategies ?? [];

  const canLaunch =
    strategyName.length > 0 &&
    typeof maxTurns === "number" && maxTurns >= 1 &&
    typeof maxCostUsd === "number" && maxCostUsd > 0;

  function toggleDefense(name: string) {
    setSelectedDefenses((prev) => {
      const next = new Set(prev);
      if (next.has(name)) {
        next.delete(name);
      } else {
        next.add(name);
      }
      return next;
    });
  }

  function handleLaunch() {
    if (!canLaunch) return;
    const items: DefenseSelectionItem[] =
      mode === "defense"
        ? Array.from(selectedDefenses).map((name) => ({ name }))
        : [];
    onLaunch(mode, items, strategyName, maxTurns as number, maxCostUsd as number, separateReasoning);
  }

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-2xl max-h-[90vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle>Launch Session</DialogTitle>
          <DialogDescription>
            Configure the attack strategy, budget, and session mode before
            launching.
          </DialogDescription>
        </DialogHeader>

        {/* Mode selection */}
        <div className="grid grid-cols-2 gap-3">
          <button
            type="button"
            onClick={() => setMode("attack")}
            className={`flex flex-col items-center gap-2 rounded-lg border-2 p-4 text-center transition-colors ${
              mode === "attack"
                ? "border-red-500 bg-red-500/5"
                : "border-border hover:border-red-500/40"
            }`}
          >
            <Swords
              className={`h-6 w-6 ${mode === "attack" ? "text-red-500" : "text-muted-foreground"}`}
            />
            <span className="text-sm font-medium">Attack Mode</span>
            <span className="text-xs text-muted-foreground">
              Pure attack cycle. No defenses. Clean vulnerability baseline.
            </span>
          </button>

          <button
            type="button"
            onClick={() => setMode("defense")}
            className={`flex flex-col items-center gap-2 rounded-lg border-2 p-4 text-center transition-colors ${
              mode === "defense"
                ? "border-blue-500 bg-blue-500/5"
                : "border-border hover:border-blue-500/40"
            }`}
          >
            <Shield
              className={`h-6 w-6 ${mode === "defense" ? "text-blue-500" : "text-muted-foreground"}`}
            />
            <span className="text-sm font-medium">Defense Mode</span>
            <span className="text-xs text-muted-foreground">
              Pre-configured defenses + reactive defender. Arms race.
            </span>
          </button>
        </div>

        {/* Strategy selection */}
        <div className="space-y-3">
          <div className="flex items-center justify-between">
            <Label className="text-sm font-medium">Attack Strategy</Label>
            {strategyName && (
              <Badge variant="outline" className="text-xs">
                {strategies.find((s) => s.name === strategyName)?.display_name ?? strategyName}
              </Badge>
            )}
          </div>
          {strategiesLoading ? (
            <div className="flex items-center justify-center py-4">
              <Loader2 className="h-4 w-4 animate-spin text-muted-foreground" />
            </div>
          ) : (
            <div className="max-h-[280px] overflow-y-auto pr-1">
              <div className="grid gap-3 sm:grid-cols-2">
                {strategies.map((s) => (
                  <StrategyCard
                    key={s.name}
                    strategy={s}
                    selected={strategyName === s.name}
                    onClick={() => setStrategyName(s.name)}
                  />
                ))}
                {strategies.length === 0 && (
                  <p className="col-span-2 text-sm text-muted-foreground text-center py-4">
                    No strategies available. Ensure the backend is running.
                  </p>
                )}
              </div>
            </div>
          )}
          {!strategyName && !strategiesLoading && strategies.length > 0 && (
            <p className="text-xs text-destructive">
              Please select an attack strategy to continue.
            </p>
          )}
        </div>

        <Separator />

        {/* Budget controls */}
        <div className="space-y-3">
          <Label className="text-sm font-medium">Budget</Label>
          <div className="grid gap-4 sm:grid-cols-2">
            <div className="space-y-2">
              <Label htmlFor="modal_max_turns" className="text-xs text-muted-foreground">
                Max Turns
              </Label>
              <Input
                id="modal_max_turns"
                type="number"
                min={1}
                max={100}
                placeholder="e.g. 20"
                value={maxTurns}
                onChange={(e) => {
                  const v = e.target.value;
                  setMaxTurns(v === "" ? "" : Math.max(1, Math.min(100, parseInt(v) || 0)));
                }}
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="modal_max_cost" className="text-xs text-muted-foreground">
                Max Cost (USD)
              </Label>
              <Input
                id="modal_max_cost"
                type="number"
                min={0}
                max={500}
                step={0.5}
                placeholder="e.g. 10.00"
                value={maxCostUsd}
                onChange={(e) => {
                  const v = e.target.value;
                  setMaxCostUsd(v === "" ? "" : Math.max(0, Math.min(500, parseFloat(v) || 0)));
                }}
              />
            </div>
          </div>
        </div>

        {/* Reasoning separation toggle */}
        <div className="flex items-center justify-between rounded-lg border p-4">
          <div className="space-y-0.5">
            <Label className="text-sm font-medium">Separate Attacker Thinking</Label>
            <p className="text-xs text-muted-foreground">
              LLM reasons about strategy separately, then outputs only the clean
              attack prompt. Disable for faster single-call mode.
            </p>
          </div>
          <Switch
            checked={separateReasoning}
            onCheckedChange={setSeparateReasoning}
          />
        </div>

        {/* Defense selection (defense mode only) */}
        {mode === "defense" && (
          <>
            <Separator />
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium">Initial Defenses</span>
                <div className="flex items-center gap-2">
                  {!defensesLoading && defenses.length > 0 && (
                    <button
                      type="button"
                      onClick={() => {
                        if (selectedDefenses.size === defenses.length) {
                          setSelectedDefenses(new Set());
                        } else {
                          setSelectedDefenses(
                            new Set(defenses.map((d) => d.name)),
                          );
                        }
                      }}
                      className="text-xs text-muted-foreground hover:text-foreground transition-colors"
                    >
                      {selectedDefenses.size === defenses.length
                        ? "Deselect all"
                        : "Select all"}
                    </button>
                  )}
                  <Badge variant="outline" className="text-xs">
                    {selectedDefenses.size} selected
                  </Badge>
                </div>
              </div>

              {defensesLoading ? (
                <div className="flex items-center justify-center py-4">
                  <Loader2 className="h-4 w-4 animate-spin text-muted-foreground" />
                </div>
              ) : (
                <div className="max-h-[200px] overflow-y-auto space-y-2 pr-1">
                  {defenses.map((d) => (
                    <label
                      key={d.name}
                      className="flex cursor-pointer items-start gap-3 rounded-md border p-3 transition-colors hover:bg-muted/50"
                    >
                      <Checkbox
                        checked={selectedDefenses.has(d.name)}
                        onCheckedChange={() => toggleDefense(d.name)}
                        className="mt-0.5"
                      />
                      <div className="flex-1 space-y-0.5">
                        <span className="text-sm font-medium">{d.name}</span>
                        {d.description && (
                          <p className="text-xs text-muted-foreground">
                            {d.description}
                          </p>
                        )}
                      </div>
                    </label>
                  ))}
                </div>
              )}

              {selectedDefenses.size === 0 && !defensesLoading && (
                <p className="text-xs text-muted-foreground">
                  No initial defenses selected. The reactive defender will still
                  activate on jailbreak detection.
                </p>
              )}
            </div>
          </>
        )}

        <DialogFooter>
          <Button
            variant="outline"
            onClick={() => onOpenChange(false)}
            disabled={isPending}
          >
            Cancel
          </Button>
          <Button onClick={handleLaunch} disabled={isPending || !canLaunch}>
            {isPending ? (
              <Loader2 className="mr-2 h-4 w-4 animate-spin" />
            ) : mode === "attack" ? (
              <Swords className="mr-2 h-4 w-4" />
            ) : (
              <Shield className="mr-2 h-4 w-4" />
            )}
            Launch {mode === "attack" ? "Attack" : "Defense"} Session
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
