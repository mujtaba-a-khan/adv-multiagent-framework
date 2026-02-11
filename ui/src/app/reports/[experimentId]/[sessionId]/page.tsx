"use client";

import { use } from "react";
import Link from "next/link";
import {
  AlertTriangle,
  ArrowLeft,
  Download,
  ShieldAlert,
  Target,
  Zap,
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
import { Skeleton } from "@/components/ui/skeleton";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { VerdictPieChart } from "@/components/charts/verdict-pie-chart";
import { SeverityBarChart } from "@/components/charts/severity-bar-chart";
import { TurnTimelineChart } from "@/components/charts/turn-timeline-chart";
import { CostBreakdownChart } from "@/components/charts/cost-breakdown-chart";
import { VulnerabilityChart } from "@/components/charts/vulnerability-chart";
import { useSessionReport } from "@/hooks/use-reports";
import { useTurns } from "@/hooks/use-turns";
import { useSessions } from "@/hooks/use-sessions";
import { ROUTES, VERDICT_BG } from "@/lib/constants";
import { exportReportJson } from "@/lib/api-client";

export default function ReportDetailPage({
  params,
}: Readonly<{
  params: Promise<{ experimentId: string; sessionId: string }>;
}>) {
  const { experimentId, sessionId } = use(params);
  const { data: report, isLoading: reportLoading } = useSessionReport(
    experimentId,
    sessionId,
  );
  const { data: turnsData } = useTurns(sessionId);
  const { data: sessionsData } = useSessions(experimentId);

  const turns = turnsData?.turns ?? [];
  const session = sessionsData?.sessions.find((s) => s.id === sessionId);

  const handleExport = async () => {
    try {
      const data = await exportReportJson(experimentId, sessionId);
      const blob = new Blob([JSON.stringify(data, null, 2)], {
        type: "application/json",
      });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `report-${sessionId}.json`;
      a.click();
      URL.revokeObjectURL(url);
    } catch {
      // Export failed silently
    }
  };

  if (reportLoading) {
    return (
      <>
        <Header title="Loading Report..." />
        <div className="flex flex-1 flex-col gap-6 p-6">
          <Skeleton className="h-8 w-64" />
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
            {["s1", "s2", "s3", "s4"].map((id) => (
              <Skeleton key={id} className="h-24 w-full" />
            ))}
          </div>
          <div className="grid gap-6 lg:grid-cols-2">
            <Skeleton className="h-[300px]" />
            <Skeleton className="h-[300px]" />
          </div>
        </div>
      </>
    );
  }

  const metrics = report?.metrics;

  return (
    <>
      <Header title="Session Report" />

      <div className="flex flex-1 flex-col gap-6 p-6">
        {/* Back + title row */}
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-4">
            <Button variant="ghost" size="icon" asChild>
              <Link href={ROUTES.reports.list}>
                <ArrowLeft className="h-4 w-4" />
              </Link>
            </Button>
            <div>
              <h2 className="text-lg font-semibold">
                {report?.experiment_name ?? "Report"}
              </h2>
              <p className="text-sm text-muted-foreground">
                {report?.target_model} &middot; {report?.strategy_name}
              </p>
            </div>
          </div>
          <Button variant="outline" size="sm" onClick={handleExport}>
            <Download className="mr-2 h-4 w-4" />
            Export JSON
          </Button>
        </div>

        {/* Objective */}
        {report?.attack_objective && (
          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="text-sm">Attack Objective</CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-sm text-muted-foreground">
                {report.attack_objective}
              </p>
            </CardContent>
          </Card>
        )}

        {/* Key metrics row */}
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-5">
          <MetricCard
            label="Attack Success Rate"
            value={`${((metrics?.asr ?? 0) * 100).toFixed(1)}%`}
            icon={<Zap className="h-4 w-4" />}
            highlight={(metrics?.asr ?? 0) > 0.3}
          />
          <MetricCard
            label="Total Turns"
            value={metrics?.total_turns ?? 0}
            icon={<Target className="h-4 w-4" />}
          />
          <MetricCard
            label="Jailbreaks"
            value={metrics?.total_jailbreaks ?? 0}
            icon={<ShieldAlert className="h-4 w-4" />}
            highlight={(metrics?.total_jailbreaks ?? 0) > 0}
          />
          <MetricCard
            label="Avg Severity"
            value={`${(metrics?.avg_severity ?? 0).toFixed(1)} / 10`}
            icon={<AlertTriangle className="h-4 w-4" />}
          />
          <MetricCard
            label="Est. Cost"
            value={`$${(metrics?.total_cost_usd ?? 0).toFixed(4)}`}
            icon={<Zap className="h-4 w-4" />}
          />
        </div>

        {/* Charts row 1: Verdict distribution + Severity per finding */}
        <div className="grid gap-6 lg:grid-cols-2">
          <Card>
            <CardHeader>
              <CardTitle>Verdict Distribution</CardTitle>
              <CardDescription>
                Breakdown of judge verdicts across all turns
              </CardDescription>
            </CardHeader>
            <CardContent>
              <VerdictPieChart
                jailbreaks={metrics?.total_jailbreaks ?? 0}
                refused={metrics?.total_refused ?? 0}
                blocked={metrics?.total_blocked ?? 0}
              />
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Finding Severity</CardTitle>
              <CardDescription>
                Severity and specificity scores for each jailbreak finding
              </CardDescription>
            </CardHeader>
            <CardContent>
              <SeverityBarChart findings={report?.findings ?? []} />
            </CardContent>
          </Card>
        </div>

        {/* Charts row 2: Turn timeline + Vulnerability categories */}
        <div className="grid gap-6 lg:grid-cols-2">
          <Card>
            <CardHeader>
              <CardTitle>Turn Timeline</CardTitle>
              <CardDescription>
                Judge confidence and severity over the attack cycle
              </CardDescription>
            </CardHeader>
            <CardContent>
              <TurnTimelineChart turns={turns} />
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Vulnerability Categories</CardTitle>
              <CardDescription>
                Most exploited vulnerability categories
              </CardDescription>
            </CardHeader>
            <CardContent>
              <VulnerabilityChart turns={turns} />
            </CardContent>
          </Card>
        </div>

        {/* Token usage */}
        {session && (
          <Card>
            <CardHeader>
              <CardTitle>Token Usage by Agent</CardTitle>
              <CardDescription>
                Stacked token consumption across all agents
              </CardDescription>
            </CardHeader>
            <CardContent>
              <CostBreakdownChart session={session} />
            </CardContent>
          </Card>
        )}

        {/* Findings table */}
        {report?.findings && report.findings.length > 0 && (
          <Card>
            <CardHeader>
              <CardTitle>Jailbreak Findings</CardTitle>
              <CardDescription>
                Detailed breakdown of each successful jailbreak
              </CardDescription>
            </CardHeader>
            <CardContent>
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Turn</TableHead>
                    <TableHead>Strategy</TableHead>
                    <TableHead>Vulnerability</TableHead>
                    <TableHead className="text-center">Severity</TableHead>
                    <TableHead className="text-center">Specificity</TableHead>
                    <TableHead>Attack Preview</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {report.findings.map((f) => (
                    <TableRow key={f.turn_number}>
                      <TableCell className="font-mono">
                        T{f.turn_number}
                      </TableCell>
                      <TableCell>{f.strategy_name}</TableCell>
                      <TableCell>
                        {f.vulnerability_category ? (
                          <Badge variant="outline" className={VERDICT_BG.jailbreak}>
                            {f.vulnerability_category}
                          </Badge>
                        ) : (
                          <span className="text-muted-foreground">—</span>
                        )}
                      </TableCell>
                      <TableCell className="text-center font-mono">
                        {f.severity.toFixed(1)}
                      </TableCell>
                      <TableCell className="text-center font-mono">
                        {f.specificity.toFixed(1)}
                      </TableCell>
                      <TableCell className="max-w-[300px] truncate text-muted-foreground text-sm">
                        {f.attack_prompt_preview}
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </CardContent>
          </Card>
        )}

        {/* Recommendations */}
        {report?.recommendations && report.recommendations.length > 0 && (
          <Card>
            <CardHeader>
              <CardTitle>Recommendations</CardTitle>
              <CardDescription>
                Suggested mitigations based on the findings
              </CardDescription>
            </CardHeader>
            <CardContent>
              <ul className="space-y-2">
                {report.recommendations.map((rec, idx) => (
                  <li key={rec} className="flex items-start gap-3 text-sm">
                    <span className="flex h-5 w-5 shrink-0 items-center justify-center rounded-full bg-primary/10 text-xs font-medium text-primary">
                      {idx + 1}
                    </span>
                    <span className="text-muted-foreground">{rec}</span>
                  </li>
                ))}
              </ul>
            </CardContent>
          </Card>
        )}
      </div>
    </>
  );
}

function MetricCard({
  label,
  value,
  icon,
  highlight = false,
}: Readonly<{
  label: string;
  value: string | number;
  icon: React.ReactNode;
  highlight?: boolean;
}>) {
  return (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between pb-2">
        <CardTitle className="text-xs font-medium text-muted-foreground">
          {label}
        </CardTitle>
        <span className="text-muted-foreground">{icon}</span>
      </CardHeader>
      <CardContent>
        <div
          className={`text-xl font-bold ${highlight ? "text-red-500" : ""}`}
        >
          {value}
        </div>
      </CardContent>
    </Card>
  );
}
