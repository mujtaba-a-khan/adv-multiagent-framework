"use client";

import { useState } from "react";
import { Loader2, Shield, Swords } from "lucide-react";

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
import { useDefenses } from "@/hooks/use-defenses";
import type { DefenseSelectionItem, SessionMode } from "@/lib/types";

interface LaunchSessionModalProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onLaunch: (mode: SessionMode, defenses: DefenseSelectionItem[]) => void;
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
  const { data: defensesData, isLoading: defensesLoading } = useDefenses();

  const defenses = defensesData?.defenses ?? [];

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
    const items: DefenseSelectionItem[] =
      mode === "defense"
        ? Array.from(selectedDefenses).map((name) => ({ name }))
        : [];
    onLaunch(mode, items);
  }

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-lg">
        <DialogHeader>
          <DialogTitle>Launch Session</DialogTitle>
          <DialogDescription>
            Choose a session mode to control how defenses are applied during the
            adversarial cycle.
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

        {/* Defense selection (defense mode only) */}
        {mode === "defense" && (
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
              <div className="max-h-[280px] overflow-y-auto space-y-2 pr-1">
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
        )}

        <DialogFooter>
          <Button
            variant="outline"
            onClick={() => onOpenChange(false)}
            disabled={isPending}
          >
            Cancel
          </Button>
          <Button onClick={handleLaunch} disabled={isPending}>
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
