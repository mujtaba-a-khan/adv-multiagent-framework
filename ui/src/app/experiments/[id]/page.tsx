"use client";

import { useState } from "react";
import Link from "next/link";
import { useParams, useRouter } from "next/navigation";
import { formatDistanceToNow, format } from "date-fns";
import {
  ArrowLeft,
  DollarSign,
  GitCompareArrows,
  Play,
  Radio,
  Shield,
  Swords,
  Target,
} from "lucide-react";
import { toast } from "sonner";
import { Header } from "@/components/layout/header";
import { LaunchSessionModal } from "@/components/launch-session-modal";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Separator } from "@/components/ui/separator";
import { Skeleton } from "@/components/ui/skeleton";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { useExperiment } from "@/hooks/use-experiments";
import {
  useSessions,
  useCreateSession,
  useStartSession,
} from "@/hooks/use-sessions";
import { ROUTES, STATUS_COLORS, CATEGORY_LABELS } from "@/lib/constants";
import type { DefenseSelectionItem, Session, SessionMode } from "@/lib/types";

export default function ExperimentDetailPage() {
  const params = useParams<{ id: string }>();
  const router = useRouter();
  const { data: experiment, isLoading: expLoading } = useExperiment(params.id);
  const { data: sessionsData, isLoading: sessLoading } = useSessions(
    params.id,
  );
  const createSession = useCreateSession(params.id);
  const startSession = useStartSession(params.id);

  const sessions = sessionsData?.sessions ?? [];
  const [showLaunchModal, setShowLaunchModal] = useState(false);

  const handleLaunch = (
    mode: SessionMode,
    defenses: DefenseSelectionItem[],
  ) => {
    createSession.mutate(
      { session_mode: mode, initial_defenses: defenses },
      {
        onSuccess: (session) => {
          toast.success("Session created");
          startSession.mutate(session.id, {
            onSuccess: () => {
              setShowLaunchModal(false);
              router.push(ROUTES.experiments.live(params.id, session.id));
            },
            onError: () => toast.error("Failed to start session"),
          });
        },
        onError: () => toast.error("Failed to create session"),
      },
    );
  };

  // Check if comparison is possible (at least one completed attack + one completed defense)
  const completedAttacks = sessions.filter(
    (s) => s.session_mode === "attack" && s.status === "completed",
  );
  const completedDefenses = sessions.filter(
    (s) => s.session_mode === "defense" && s.status === "completed",
  );
  const canCompare = completedAttacks.length > 0 && completedDefenses.length > 0;

  if (expLoading) {
    return (
      <>
        <Header />
        <div className="p-6 space-y-4">
          <Skeleton className="h-8 w-64" />
          <Skeleton className="h-40 w-full" />
        </div>
      </>
    );
  }

  if (!experiment) {
    return (
      <>
        <Header />
        <div className="flex flex-col items-center justify-center p-12">
          <p className="text-muted-foreground">Experiment not found.</p>
          <Button variant="ghost" className="mt-4" asChild>
            <Link href={ROUTES.experiments.list}>
              <ArrowLeft className="mr-2 h-4 w-4" />
              Back to experiments
            </Link>
          </Button>
        </div>
      </>
    );
  }

  return (
    <>
      <Header title={experiment.name} />

      <div className="flex flex-1 flex-col gap-6 p-6">
        <div className="flex items-center gap-4">
          <Button variant="ghost" size="icon" asChild>
            <Link href={ROUTES.experiments.list}>
              <ArrowLeft className="h-4 w-4" />
            </Link>
          </Button>
          <div className="flex-1">
            <h2 className="text-2xl font-bold tracking-tight">
              {experiment.name}
            </h2>
            {experiment.description && (
              <p className="text-sm text-muted-foreground">
                {experiment.description}
              </p>
            )}
          </div>
          <Button onClick={() => setShowLaunchModal(true)}>
            <Play className="mr-2 h-4 w-4" />
            Launch Session
          </Button>
        </div>

        {/* Config summary */}
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
          <ConfigCard
            icon={<Target className="h-4 w-4" />}
            label="Target"
            value={experiment.target_model}
            sub={experiment.target_provider}
          />
          <ConfigCard
            icon={<Swords className="h-4 w-4" />}
            label="Strategy"
            value={
              CATEGORY_LABELS[experiment.strategy_name] ??
              experiment.strategy_name
            }
            sub={`Max ${experiment.max_turns} turns`}
          />
          <ConfigCard
            icon={<Shield className="h-4 w-4" />}
            label="Agents"
            value={experiment.attacker_model.split(":")[0]}
            sub={`Judge: ${experiment.analyzer_model.split(":")[0]}`}
          />
          <ConfigCard
            icon={<DollarSign className="h-4 w-4" />}
            label="Budget"
            value={`$${experiment.max_cost_usd.toFixed(2)}`}
            sub={`Created ${formatDistanceToNow(new Date(experiment.created_at), { addSuffix: true })}`}
          />
        </div>

        {/* Objective card */}
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Attack Objective</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-sm whitespace-pre-wrap">
              {experiment.attack_objective}
            </p>
          </CardContent>
        </Card>

        {/* Sessions table */}
        <Card>
          <CardHeader className="flex flex-row items-center justify-between">
            <div>
              <CardTitle>Sessions</CardTitle>
              <CardDescription>
                {sessions.length} session{sessions.length !== 1 && "s"} run
              </CardDescription>
            </div>
            {canCompare && (
              <Button variant="outline" size="sm" asChild>
                <Link
                  href={ROUTES.experiments.compare(
                    params.id,
                    completedAttacks[0].id,
                    completedDefenses[0].id,
                  )}
                >
                  <GitCompareArrows className="mr-2 h-4 w-4" />
                  Compare
                </Link>
              </Button>
            )}
          </CardHeader>
          <CardContent>
            {sessLoading ? (
              <div className="space-y-3">
                {Array.from({ length: 3 }).map((_, i) => (
                  <Skeleton key={i} className="h-12 w-full" />
                ))}
              </div>
            ) : sessions.length === 0 ? (
              <div className="flex flex-col items-center justify-center py-12 text-center">
                <Radio className="h-8 w-8 text-muted-foreground" />
                <p className="mt-3 text-sm text-muted-foreground">
                  No sessions yet. Launch one to start the adversarial battle.
                </p>
              </div>
            ) : (
              <div className="rounded-md border">
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>Status</TableHead>
                      <TableHead>Mode</TableHead>
                      <TableHead>Turns</TableHead>
                      <TableHead>Jailbreaks</TableHead>
                      <TableHead>ASR</TableHead>
                      <TableHead>Cost</TableHead>
                      <TableHead>Started</TableHead>
                      <TableHead className="w-12" />
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {sessions.map((s) => (
                      <SessionRow
                        key={s.id}
                        session={s}
                        experimentId={params.id}
                      />
                    ))}
                  </TableBody>
                </Table>
              </div>
            )}
          </CardContent>
        </Card>
      </div>

      <LaunchSessionModal
        open={showLaunchModal}
        onOpenChange={setShowLaunchModal}
        onLaunch={handleLaunch}
        isPending={createSession.isPending || startSession.isPending}
      />
    </>
  );
}

function ConfigCard({
  icon,
  label,
  value,
  sub,
}: {
  icon: React.ReactNode;
  label: string;
  value: string;
  sub: string;
}) {
  return (
    <Card>
      <CardContent className="pt-6">
        <div className="flex items-center gap-3">
          <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg bg-muted">
            {icon}
          </div>
          <div>
            <p className="text-xs text-muted-foreground">{label}</p>
            <p className="text-sm font-medium">{value}</p>
            <p className="text-xs text-muted-foreground">{sub}</p>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

function SessionRow({
  session,
  experimentId,
}: {
  session: Session;
  experimentId: string;
}) {
  const asr =
    session.asr !== null ? `${(session.asr * 100).toFixed(1)}%` : "—";

  return (
    <TableRow>
      <TableCell>
        <Badge
          variant="outline"
          className={STATUS_COLORS[session.status] ?? ""}
        >
          {session.status}
        </Badge>
      </TableCell>
      <TableCell>
        <Badge
          variant="outline"
          className={
            session.session_mode === "defense"
              ? "bg-blue-500/10 text-blue-500 border-blue-500/20"
              : "bg-red-500/10 text-red-500 border-red-500/20"
          }
        >
          {session.session_mode === "defense" ? (
            <>
              <Shield className="mr-1 h-3 w-3" />
              Defense
            </>
          ) : (
            <>
              <Swords className="mr-1 h-3 w-3" />
              Attack
            </>
          )}
        </Badge>
      </TableCell>
      <TableCell>{session.total_turns}</TableCell>
      <TableCell className="text-red-500 font-medium">
        {session.total_jailbreaks}
      </TableCell>
      <TableCell className="font-mono text-sm">{asr}</TableCell>
      <TableCell className="text-muted-foreground">
        ${session.estimated_cost_usd.toFixed(4)}
      </TableCell>
      <TableCell className="text-muted-foreground text-xs">
        {session.started_at
          ? formatDistanceToNow(new Date(session.started_at), {
              addSuffix: true,
            })
          : "—"}
      </TableCell>
      <TableCell>
        {(session.status === "running" || session.status === "completed") && (
          <Button variant="ghost" size="sm" asChild>
            <Link
              href={ROUTES.experiments.live(experimentId, session.id)}
            >
              <Radio className="mr-1 h-3 w-3" />
              View
            </Link>
          </Button>
        )}
      </TableCell>
    </TableRow>
  );
}
