"use client";

import Link from "next/link";
import { useParams, useSearchParams } from "next/navigation";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  ArrowDown,
  ArrowLeft,
  Bot,
  Radio,
  Shield,
  ShieldCheck,
  Swords,
  Target,
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
import { Progress } from "@/components/ui/progress";
import { Separator } from "@/components/ui/separator";
import { Skeleton } from "@/components/ui/skeleton";
import { useExperiment } from "@/hooks/use-experiments";
import { useSession } from "@/hooks/use-sessions";
import { useTurns } from "@/hooks/use-turns";
import { useWSStore, type PendingTurn } from "@/stores/ws-store";
import { ROUTES, VERDICT_BG, CATEGORY_LABELS } from "@/lib/constants";
import type { Turn } from "@/lib/types";

const SCROLL_THRESHOLD = 150;

export default function LiveBattlePage() {
  const params = useParams<{ id: string }>();
  const searchParams = useSearchParams();
  const sessionId = searchParams.get("session") ?? "";

  const { data: experiment } = useExperiment(params.id);
  const { data: session } = useSession(params.id, sessionId);
  const { data: turnsData } = useTurns(sessionId);
  const { connect, disconnect, liveTurns, pendingTurn } = useWSStore();

  const chatRef = useRef<HTMLDivElement>(null);
  const [isNearBottom, setIsNearBottom] = useState(true);
  const [showJumpBtn, setShowJumpBtn] = useState(false);

  // Connect WebSocket immediately when we have a sessionId (don't wait for
  // status — avoids the race where events arrive before the first status fetch).
  useEffect(() => {
    if (sessionId) {
      connect(sessionId);
      return () => disconnect();
    }
  }, [sessionId, connect, disconnect]);

  // Merge polled turns (from DB) with live turns (from WebSocket),
  // deduplicating by turn_number. Polled turns take precedence (have DB id).
  const allTurns = useMemo(() => {
    const polled = turnsData?.turns ?? [];
    const polledNumbers = new Set(polled.map((t) => t.turn_number));
    const wsOnly = liveTurns.filter((t) => !polledNumbers.has(t.turn_number));
    return [...polled, ...wsOnly];
  }, [turnsData?.turns, liveTurns]);

  // Track scroll position for smart auto-scroll
  const handleScroll = useCallback(() => {
    const el = chatRef.current;
    if (!el) return;
    const nearBottom =
      el.scrollHeight - el.scrollTop - el.clientHeight < SCROLL_THRESHOLD;
    setIsNearBottom(nearBottom);
    setShowJumpBtn(!nearBottom);
  }, []);

  // Determine pending phase for scroll dependency (stable string, not object)
  const pendingPhase = pendingTurn
    ? !pendingTurn.attack_prompt
      ? "thinking"
      : !pendingTurn.target_response
        ? "awaiting"
        : "analyzing"
    : null;

  // Auto-scroll when new content arrives (only if user is near the bottom)
  useEffect(() => {
    if (isNearBottom && chatRef.current) {
      chatRef.current.scrollTo({
        top: chatRef.current.scrollHeight,
        behavior: "smooth",
      });
    }
  }, [allTurns.length, pendingPhase, isNearBottom]);

  const jumpToLatest = useCallback(() => {
    chatRef.current?.scrollTo({
      top: chatRef.current.scrollHeight,
      behavior: "smooth",
    });
  }, []);

  const isRunning = session?.status === "running";
  const isStarting = session?.status === "pending" || session?.status === "running";
  const asr =
    session?.asr !== null && session?.asr !== undefined
      ? (session.asr * 100).toFixed(1)
      : "0.0";

  const turnsProgress = session
    ? (session.total_turns / (experiment?.max_turns ?? 20)) * 100
    : 0;

  return (
    <div className="flex h-dvh flex-col overflow-hidden">
      <Header title="Live Battle" />

      <div className="flex min-h-0 flex-1 flex-col lg:flex-row">
        {/* ── Chat panel ──────────────────────────────────── */}
        <div className="flex min-h-0 flex-1 flex-col border-r">
          {/* Toolbar */}
          <div className="flex shrink-0 items-center gap-3 border-b px-4 py-3">
            <Button variant="ghost" size="icon" asChild>
              <Link href={ROUTES.experiments.detail(params.id)}>
                <ArrowLeft className="h-4 w-4" />
              </Link>
            </Button>
            <div className="min-w-0 flex-1">
              <h3 className="truncate text-sm font-medium">
                {experiment?.name ?? "Battle Session"}
              </h3>
              <p className="truncate text-xs text-muted-foreground">
                {experiment?.strategy_name
                  ? (CATEGORY_LABELS[experiment.strategy_name] ??
                    experiment.strategy_name)
                  : ""}{" "}
                vs {experiment?.target_model}
              </p>
            </div>
            {isRunning && (
              <Badge
                variant="outline"
                className="shrink-0 border-blue-500/20 bg-blue-500/10 text-blue-500"
              >
                <Radio className="mr-1 h-3 w-3 animate-pulse" />
                Live
              </Badge>
            )}
            {session?.status === "completed" && (
              <Badge
                variant="outline"
                className="shrink-0 border-emerald-500/20 bg-emerald-500/10 text-emerald-500"
              >
                <ShieldCheck className="mr-1 h-3 w-3" />
                Completed
              </Badge>
            )}
          </div>

          {/* Scrollable chat area */}
          <div className="relative min-h-0 flex-1">
            <div
              ref={chatRef}
              onScroll={handleScroll}
              className="h-full overflow-y-auto p-4"
            >
              <div className="mx-auto max-w-3xl space-y-4 pb-4">
                {/* Empty state — only show when session is done with no turns */}
                {allTurns.length === 0 && !isStarting && !pendingTurn && (
                  <div className="flex flex-col items-center justify-center py-16 text-center">
                    <Swords className="h-10 w-10 text-muted-foreground" />
                    <p className="mt-3 text-sm text-muted-foreground">
                      No turns recorded yet.
                    </p>
                  </div>
                )}

                {/* Session starting — show baseline objective preview immediately */}
                {isStarting &&
                  allTurns.length === 0 &&
                  !pendingTurn && (
                    <div className="space-y-3">
                      <div className="flex items-center gap-2">
                        <span className="font-mono text-xs text-muted-foreground">
                          Turn 1
                        </span>
                        <Badge
                          variant="outline"
                          className="animate-pulse border-blue-500/20 bg-blue-500/10 text-[10px] text-blue-500"
                        >
                          Initializing models&hellip;
                        </Badge>
                        <Badge
                          variant="outline"
                          className="border-cyan-500/20 bg-cyan-500/10 text-[10px] text-cyan-500"
                        >
                          Baseline
                        </Badge>
                      </div>
                      <MessageBubble role="attacker" label="Attacker (Baseline)">
                        <p className="whitespace-pre-wrap break-words text-sm [overflow-wrap:anywhere]">
                          {experiment?.attack_objective ?? ""}
                        </p>
                      </MessageBubble>
                      <MessageBubble role="target">
                        <BouncingDots color="bg-blue-500/60" />
                      </MessageBubble>
                    </div>
                  )}

                {/* Completed turns */}
                {allTurns.map((turn) => (
                  <TurnMessage key={turn.id ?? turn.turn_number} turn={turn} />
                ))}

                {/* Progressive pending turn */}
                {pendingTurn && <PendingTurnMessage pending={pendingTurn} />}
              </div>
            </div>

            {/* Jump to latest button */}
            {showJumpBtn && (
              <button
                type="button"
                onClick={jumpToLatest}
                className="absolute bottom-4 left-1/2 z-10 flex -translate-x-1/2 items-center gap-1.5 rounded-full border bg-background/95 px-3 py-1.5 text-xs font-medium shadow-lg backdrop-blur transition-colors hover:bg-accent"
              >
                <ArrowDown className="h-3 w-3" />
                Jump to latest
              </button>
            )}
          </div>
        </div>

        {/* ── Metrics sidebar ─────────────────────────────── */}
        <div className="w-full shrink-0 overflow-y-auto border-t lg:w-80 lg:border-t-0">
          <div className="space-y-4 p-4">
            <Card>
              <CardHeader className="pb-3">
                <CardTitle className="text-sm">Session Metrics</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                {session ? (
                  <>
                    <MetricRow label="Status" value={session.status} />
                    <MetricRow
                      label="Turns"
                      value={`${session.total_turns} / ${experiment?.max_turns ?? "?"}`}
                    />
                    <Progress value={turnsProgress} className="h-1.5" />
                    <Separator />
                    <MetricRow
                      label="ASR"
                      value={`${asr}%`}
                      valueClass={
                        parseFloat(asr) > 0
                          ? "text-red-500 font-bold"
                          : "text-emerald-500"
                      }
                    />
                    <MetricRow
                      label="Jailbreaks"
                      value={String(session.total_jailbreaks)}
                      valueClass="text-red-500"
                    />
                    <MetricRow
                      label="Borderline"
                      value={String(session.total_borderline)}
                      valueClass="text-amber-500"
                    />
                    <MetricRow
                      label="Refused"
                      value={String(session.total_refused)}
                      valueClass="text-emerald-500"
                    />
                    <MetricRow
                      label="Blocked"
                      value={String(session.total_blocked)}
                    />
                    <Separator />
                    <MetricRow
                      label="Est. Cost"
                      value={`$${session.estimated_cost_usd.toFixed(4)}`}
                    />
                    <MetricRow
                      label="Attacker Tokens"
                      value={session.total_attacker_tokens.toLocaleString()}
                    />
                    <MetricRow
                      label="Target Tokens"
                      value={session.total_target_tokens.toLocaleString()}
                    />
                    <MetricRow
                      label="Analyzer Tokens"
                      value={session.total_analyzer_tokens.toLocaleString()}
                    />
                  </>
                ) : (
                  <div className="space-y-3">
                    {Array.from({ length: 6 }).map((_, i) => (
                      <Skeleton key={i} className="h-4 w-full" />
                    ))}
                  </div>
                )}
              </CardContent>
            </Card>

            {experiment && (
              <Card>
                <CardHeader className="pb-3">
                  <CardTitle className="text-sm">Agent Models</CardTitle>
                </CardHeader>
                <CardContent className="space-y-2">
                  <AgentBadge
                    icon={<Swords className="h-3 w-3" />}
                    label="Attacker"
                    model={experiment.attacker_model}
                  />
                  <AgentBadge
                    icon={<Bot className="h-3 w-3" />}
                    label="Analyzer"
                    model={experiment.analyzer_model}
                  />
                  <AgentBadge
                    icon={<Shield className="h-3 w-3" />}
                    label="Defender"
                    model={experiment.defender_model}
                  />
                  <AgentBadge
                    icon={<Target className="h-3 w-3" />}
                    label="Target"
                    model={experiment.target_model}
                  />
                </CardContent>
              </Card>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

/* ── Bouncing dots animation ──────────────────────────────────────────────── */

function BouncingDots({ color = "bg-current" }: { color?: string }) {
  return (
    <span className="inline-flex items-center gap-1">
      <span
        className={`h-1.5 w-1.5 rounded-full ${color} animate-bounce [animation-delay:0ms]`}
      />
      <span
        className={`h-1.5 w-1.5 rounded-full ${color} animate-bounce [animation-delay:150ms]`}
      />
      <span
        className={`h-1.5 w-1.5 rounded-full ${color} animate-bounce [animation-delay:300ms]`}
      />
    </span>
  );
}

/* ── Message bubble used by both completed and pending turns ───────────── */

function MessageBubble({
  role,
  label,
  blocked,
  children,
}: {
  role: "attacker" | "target";
  label?: string;
  blocked?: boolean;
  children: React.ReactNode;
}) {
  const isAttacker = role === "attacker";
  const borderColor = isAttacker
    ? "border-red-500/10 bg-red-500/5"
    : blocked
      ? "border-amber-500/10 bg-amber-500/5"
      : "border-blue-500/10 bg-blue-500/5";
  const iconColor = isAttacker
    ? "bg-red-500/10 text-red-500"
    : "bg-blue-500/10 text-blue-500";
  const labelColor = isAttacker
    ? "text-red-500"
    : blocked
      ? "text-amber-500"
      : "text-blue-500";
  const Icon = isAttacker ? Swords : Target;
  const displayLabel =
    label ?? (isAttacker ? "Attacker" : blocked ? "Target (Blocked)" : "Target");

  return (
    <div className="flex gap-3">
      <div
        className={`flex h-7 w-7 shrink-0 items-center justify-center rounded-full ${iconColor}`}
      >
        <Icon className="h-3.5 w-3.5" />
      </div>
      <div
        className={`min-w-0 flex-1 rounded-lg border p-3 ${borderColor}`}
      >
        <p className={`mb-1 text-xs font-medium ${labelColor}`}>
          {displayLabel}
        </p>
        {children}
      </div>
    </div>
  );
}

/* ── Completed turn ────────────────────────────────────────────────────── */

function TurnMessage({ turn }: { turn: Turn }) {
  const isBaseline = turn.turn_number === 1;

  return (
    <div className="space-y-3">
      {/* Turn header */}
      <div className="flex flex-wrap items-center gap-2">
        <span className="font-mono text-xs text-muted-foreground">
          Turn {turn.turn_number}
        </span>
        <Badge
          variant="outline"
          className={`text-[10px] ${VERDICT_BG[turn.judge_verdict] ?? ""}`}
        >
          {turn.judge_verdict}
          {turn.judge_confidence > 0 &&
            ` (${(turn.judge_confidence * 100).toFixed(0)}%)`}
        </Badge>
        {isBaseline && (
          <Badge
            variant="outline"
            className="border-cyan-500/20 bg-cyan-500/10 text-[10px] text-cyan-500"
          >
            Baseline
          </Badge>
        )}
        {turn.vulnerability_category && (
          <Badge variant="outline" className="text-[10px]">
            {turn.vulnerability_category}
          </Badge>
        )}
      </div>

      {/* Attacker message */}
      <MessageBubble
        role="attacker"
        label={isBaseline ? "Attacker (Baseline)" : undefined}
      >
        <p className="whitespace-pre-wrap break-words text-sm [overflow-wrap:anywhere]">
          {turn.attack_prompt}
        </p>
      </MessageBubble>

      {/* Target response */}
      <MessageBubble role="target" blocked={turn.target_blocked}>
        <p className="whitespace-pre-wrap break-words text-sm [overflow-wrap:anywhere]">
          {turn.target_response}
        </p>
      </MessageBubble>

      {/* Score row */}
      {(turn.severity_score !== null ||
        turn.specificity_score !== null ||
        turn.coherence_score !== null) && (
        <div className="ml-10 flex flex-wrap gap-3 text-xs text-muted-foreground">
          {turn.severity_score !== null && (
            <span>Severity: {turn.severity_score}/10</span>
          )}
          {turn.specificity_score !== null && (
            <span>Specificity: {turn.specificity_score}/10</span>
          )}
          {turn.coherence_score !== null && (
            <span>Coherence: {turn.coherence_score}/10</span>
          )}
        </div>
      )}

      <Separator className="my-1" />
    </div>
  );
}

/* ── Pending turn (built up progressively from WS events) ──────────────── */

function PendingTurnMessage({ pending }: { pending: PendingTurn }) {
  const hasPrompt = !!pending.attack_prompt;
  const hasResponse = !!pending.target_response;
  const isBaseline = pending.is_baseline || pending.turn_number === 1;
  // Before the attacker node completes, show the objective from the turn_start event
  const previewText = pending.attack_objective;

  const phaseLabel = !hasPrompt
    ? isBaseline
      ? "Sending baseline\u2026"
      : "Generating attack\u2026"
    : !hasResponse
      ? "Awaiting response\u2026"
      : "Analyzing\u2026";

  return (
    <div className="space-y-3">
      {/* Turn header */}
      <div className="flex items-center gap-2">
        <span className="font-mono text-xs text-muted-foreground">
          Turn {pending.turn_number}
        </span>
        <Badge
          variant="outline"
          className="animate-pulse border-blue-500/20 bg-blue-500/10 text-[10px] text-blue-500"
        >
          {phaseLabel}
        </Badge>
        {isBaseline && (
          <Badge
            variant="outline"
            className="border-cyan-500/20 bg-cyan-500/10 text-[10px] text-cyan-500"
          >
            Baseline
          </Badge>
        )}
      </div>

      {/* Attacker message — show objective preview if prompt not yet arrived */}
      <MessageBubble
        role="attacker"
        label={isBaseline ? "Attacker (Baseline)" : undefined}
      >
        {hasPrompt ? (
          <p className="whitespace-pre-wrap break-words text-sm [overflow-wrap:anywhere]">
            {pending.attack_prompt}
          </p>
        ) : previewText ? (
          <p className="whitespace-pre-wrap break-words text-sm text-muted-foreground [overflow-wrap:anywhere]">
            {previewText}
          </p>
        ) : (
          <BouncingDots color="bg-red-500/60" />
        )}
      </MessageBubble>

      {/* Target response (or waiting dots) — show once attack is ready or preview is shown */}
      {(hasPrompt || previewText) && (
        <MessageBubble role="target" blocked={pending.target_blocked}>
          {hasResponse ? (
            <p className="whitespace-pre-wrap break-words text-sm [overflow-wrap:anywhere]">
              {pending.target_response}
            </p>
          ) : (
            <BouncingDots color="bg-blue-500/60" />
          )}
        </MessageBubble>
      )}

      {/* Analyzing indicator — only show once target has responded */}
      {hasResponse && (
        <div className="ml-10 flex animate-pulse items-center gap-2 text-sm text-muted-foreground">
          <Bot className="h-4 w-4" />
          Analyzing response&hellip;
        </div>
      )}
    </div>
  );
}

/* ── Small helpers ─────────────────────────────────────────────────────── */

function MetricRow({
  label,
  value,
  valueClass,
}: {
  label: string;
  value: string;
  valueClass?: string;
}) {
  return (
    <div className="flex items-center justify-between">
      <span className="text-xs text-muted-foreground">{label}</span>
      <span className={`text-sm font-medium ${valueClass ?? ""}`}>
        {value}
      </span>
    </div>
  );
}

function AgentBadge({
  icon,
  label,
  model,
}: {
  icon: React.ReactNode;
  label: string;
  model: string;
}) {
  return (
    <div className="flex items-center gap-2 rounded-md bg-muted/50 px-2.5 py-1.5">
      {icon}
      <span className="text-xs text-muted-foreground">{label}:</span>
      <code className="text-xs font-medium">{model}</code>
    </div>
  );
}
