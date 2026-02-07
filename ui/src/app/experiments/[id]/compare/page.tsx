"use client";

import { useState } from "react";
import Link from "next/link";
import { useParams, useSearchParams } from "next/navigation";
import {
  ArrowLeft,
  ArrowDown,
  ArrowUp,
  ChevronDown,
  ChevronUp,
  Minus,
  Shield,
  Swords,
} from "lucide-react";
import { Header } from "@/components/layout/header";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Separator } from "@/components/ui/separator";
import { Skeleton } from "@/components/ui/skeleton";
import { useComparison } from "@/hooks/use-comparison";
import { ROUTES, VERDICT_BG } from "@/lib/constants";
import type { Session, Turn } from "@/lib/types";

export default function ComparisonPage() {
  const params = useParams<{ id: string }>();
  const searchParams = useSearchParams();
  const attackId = searchParams.get("attack") ?? "";
  const defenseId = searchParams.get("defense") ?? "";

  const { data, isLoading } = useComparison(params.id, attackId, defenseId);

  if (isLoading || !data) {
    return (
      <div className="flex h-dvh flex-col">
        <Header title="Session Comparison" />
        <div className="p-6 space-y-4">
          <Skeleton className="h-8 w-64" />
          <div className="grid grid-cols-2 gap-4">
            <Skeleton className="h-48" />
            <Skeleton className="h-48" />
          </div>
        </div>
      </div>
    );
  }

  const { attack_session, defense_session, attack_turns, defense_turns } = data;

  // Deltas from the defense perspective (defense − attack).
  // Negative = defense reduced the metric, positive = defense increased it.
  const asrDelta =
    (defense_session.asr ?? 0) - (attack_session.asr ?? 0);
  const jailbreakDelta =
    defense_session.total_jailbreaks - attack_session.total_jailbreaks;
  const blockedDelta =
    defense_session.total_blocked - attack_session.total_blocked;

  const maxTurns = Math.max(attack_turns.length, defense_turns.length);

  return (
    <div className="flex h-dvh flex-col overflow-hidden">
      <Header title="Session Comparison" />

      <div className="flex-1 overflow-y-auto p-6">
        <div className="mx-auto max-w-6xl space-y-6">
          {/* Back button + title */}
          <div className="flex items-center gap-4">
            <Button variant="ghost" size="icon" asChild>
              <Link href={ROUTES.experiments.detail(params.id)}>
                <ArrowLeft className="h-4 w-4" />
              </Link>
            </Button>
            <h2 className="text-xl font-bold">Attack vs Defense Comparison</h2>
          </div>

          {/* Metrics delta cards */}
          <div className="grid gap-4 sm:grid-cols-3">
            <DeltaCard
              label="ASR"
              attackValue={`${((attack_session.asr ?? 0) * 100).toFixed(1)}%`}
              defenseValue={`${((defense_session.asr ?? 0) * 100).toFixed(1)}%`}
              delta={asrDelta * 100}
              format={(v) => `${v > 0 ? "+" : ""}${v.toFixed(1)}%`}
            />
            <DeltaCard
              label="Jailbreaks"
              attackValue={String(attack_session.total_jailbreaks)}
              defenseValue={String(defense_session.total_jailbreaks)}
              delta={jailbreakDelta}
              format={(v) => `${v > 0 ? "+" : ""}${v}`}
            />
            <DeltaCard
              label="Blocked"
              attackValue={String(attack_session.total_blocked)}
              defenseValue={String(defense_session.total_blocked)}
              delta={blockedDelta}
              format={(v) => `${v > 0 ? "+" : ""}${v}`}
            />
          </div>

          {/* Session summary row */}
          <div className="grid grid-cols-2 gap-4">
            <SessionSummaryCard session={attack_session} mode="attack" />
            <SessionSummaryCard session={defense_session} mode="defense" />
          </div>

          <Separator />

          {/* Turn-by-turn comparison */}
          <h3 className="text-lg font-semibold">Turn-by-Turn Comparison</h3>
          <div className="space-y-4">
            {Array.from({ length: maxTurns }).map((_, i) => {
              const turnNum = i + 1;
              const aTurn = attack_turns.find((t) => t.turn_number === turnNum);
              const dTurn = defense_turns.find(
                (t) => t.turn_number === turnNum,
              );
              return (
                <div key={turnNum} className="grid grid-cols-2 gap-4">
                  <TurnCard turn={aTurn} turnNumber={turnNum} mode="attack" />
                  <TurnCard turn={dTurn} turnNumber={turnNum} mode="defense" />
                </div>
              );
            })}
          </div>
        </div>
      </div>
    </div>
  );
}

/* ── Sub-components ──────────────────────────────────────────── */

function DeltaCard({
  label,
  attackValue,
  defenseValue,
  delta,
  format,
}: {
  label: string;
  attackValue: string;
  defenseValue: string;
  delta: number;
  format: (v: number) => string;
}) {
  // Directional coloring: red = decrease, green = increase, neutral = zero
  const colorClass =
    delta === 0
      ? "text-muted-foreground"
      : delta < 0
        ? "text-red-500"
        : "text-emerald-500";

  return (
    <Card>
      <CardContent className="pt-6">
        <p className="text-xs text-muted-foreground">{label}</p>
        <div className="mt-2 flex items-baseline justify-between">
          <div className="space-y-1">
            <div className="flex items-center gap-2 text-sm">
              <Swords className="h-3 w-3 text-red-500" />
              <span>{attackValue}</span>
            </div>
            <div className="flex items-center gap-2 text-sm">
              <Shield className="h-3 w-3 text-blue-500" />
              <span>{defenseValue}</span>
            </div>
          </div>
          <div className={`flex items-center gap-1 text-lg font-bold ${colorClass}`}>
            {delta === 0 ? (
              <Minus className="h-4 w-4" />
            ) : delta < 0 ? (
              <ArrowDown className="h-4 w-4" />
            ) : (
              <ArrowUp className="h-4 w-4" />
            )}
            {format(delta)}
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

function SessionSummaryCard({
  session,
  mode,
}: {
  session: Session;
  mode: "attack" | "defense";
}) {
  const isDefense = mode === "defense";
  return (
    <Card
      className={
        isDefense ? "border-blue-500/20" : "border-red-500/20"
      }
    >
      <CardHeader className="pb-3">
        <CardTitle className="flex items-center gap-2 text-sm">
          {isDefense ? (
            <Shield className="h-4 w-4 text-blue-500" />
          ) : (
            <Swords className="h-4 w-4 text-red-500" />
          )}
          {isDefense ? "Defense Session" : "Attack Session"}
        </CardTitle>
      </CardHeader>
      <CardContent className="text-sm space-y-1">
        <Row label="Turns" value={String(session.total_turns)} />
        <Row label="Jailbreaks" value={String(session.total_jailbreaks)} />
        <Row label="Refused" value={String(session.total_refused)} />
        <Row label="Borderline" value={String(session.total_borderline)} />
        <Row label="Blocked" value={String(session.total_blocked)} />
        <Row
          label="Cost"
          value={`$${session.estimated_cost_usd.toFixed(4)}`}
        />
      </CardContent>
    </Card>
  );
}

function Row({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex justify-between">
      <span className="text-muted-foreground">{label}</span>
      <span className="font-medium">{value}</span>
    </div>
  );
}

function TurnCard({
  turn,
  turnNumber,
}: {
  turn: Turn | undefined;
  turnNumber: number;
  mode: "attack" | "defense";
}) {
  const [expanded, setExpanded] = useState(false);

  if (!turn) {
    return (
      <Card className="border-dashed opacity-50">
        <CardContent className="flex items-center justify-center py-6">
          <span className="text-xs text-muted-foreground">
            Turn {turnNumber} — no data
          </span>
        </CardContent>
      </Card>
    );
  }

  const verdictClass = VERDICT_BG[turn.judge_verdict] ?? "";
  const canExpand =
    (turn.attack_prompt?.length ?? 0) > 120 ||
    (turn.target_response?.length ?? 0) > 120;

  return (
    <Card>
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between">
          <CardTitle className="text-xs font-medium text-muted-foreground">
            Turn {turnNumber}
            {turn.strategy_name && ` · ${turn.strategy_name}`}
          </CardTitle>
          <Badge variant="outline" className={verdictClass}>
            {turn.judge_verdict}
          </Badge>
        </div>
      </CardHeader>
      <CardContent className="space-y-2 text-xs">
        <div>
          <span className="font-medium text-muted-foreground">Attack: </span>
          <span className={expanded ? "whitespace-pre-wrap" : "line-clamp-3"}>
            {turn.attack_prompt}
          </span>
        </div>
        {turn.attacker_reasoning && (
          <details className="rounded-md border border-dashed border-purple-500/20 bg-purple-500/5 p-2">
            <summary className="cursor-pointer text-[10px] font-medium text-purple-600 dark:text-purple-400">
              View Attacker Thinking
            </summary>
            <p className="mt-1 whitespace-pre-wrap text-[11px] text-muted-foreground">
              {turn.attacker_reasoning}
            </p>
          </details>
        )}
        <div>
          <span className="font-medium text-muted-foreground">Response: </span>
          <span className={expanded ? "whitespace-pre-wrap" : "line-clamp-3"}>
            {turn.target_response}
          </span>
        </div>
        {turn.target_blocked && (
          <Badge
            variant="outline"
            className="bg-amber-500/10 text-amber-500 border-amber-500/20"
          >
            Blocked
          </Badge>
        )}
        {canExpand && (
          <button
            type="button"
            onClick={() => setExpanded(!expanded)}
            className="flex items-center gap-1 text-xs text-muted-foreground hover:text-foreground transition-colors pt-1"
          >
            {expanded ? (
              <>
                <ChevronUp className="h-3 w-3" />
                Show less
              </>
            ) : (
              <>
                <ChevronDown className="h-3 w-3" />
                Show more
              </>
            )}
          </button>
        )}
      </CardContent>
    </Card>
  );
}
