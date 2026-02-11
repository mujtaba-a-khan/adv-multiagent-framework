"use client";

import Link from "next/link";
import { formatDistanceToNow } from "date-fns";
import {
  ArrowRight,
  BarChart3,
  FileText,
  ShieldAlert,
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
import { useCompletedSessions } from "@/hooks/use-reports";
import { ROUTES, VERDICT_BG } from "@/lib/constants";

export default function ReportsPage() {
  const { data: sessions, isLoading } = useCompletedSessions();

  const totalJailbreaks = sessions?.reduce((s, r) => s + r.total_jailbreaks, 0) ?? 0;
  const avgASR =
    sessions && sessions.length > 0
      ? sessions.reduce((s, r) => s + (r.asr ?? 0), 0) / sessions.length
      : 0;

  return (
    <>
      <Header title="Reports" />

      <div className="flex flex-1 flex-col gap-6 p-6">
        {/* Summary stats */}
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
          <StatCard
            title="Completed Sessions"
            value={isLoading ? null : (sessions?.length ?? 0)}
            icon={<FileText className="h-4 w-4 text-muted-foreground" />}
          />
          <StatCard
            title="Total Jailbreaks"
            value={isLoading ? null : totalJailbreaks}
            icon={<ShieldAlert className="h-4 w-4 text-muted-foreground" />}
            valueClass="text-red-500"
          />
          <StatCard
            title="Avg. ASR"
            value={isLoading ? null : `${(avgASR * 100).toFixed(1)}%`}
            icon={<BarChart3 className="h-4 w-4 text-muted-foreground" />}
          />
          <StatCard
            title="Total Cost"
            value={
              isLoading
                ? null
                : `$${(sessions?.reduce((s, r) => s + r.estimated_cost_usd, 0) ?? 0).toFixed(4)}`
            }
            icon={<BarChart3 className="h-4 w-4 text-muted-foreground" />}
          />
        </div>

        {/* Session reports table */}
        <Card>
          <CardHeader>
            <CardTitle>Session Reports</CardTitle>
            <CardDescription>
              View detailed analysis for completed adversarial sessions
            </CardDescription>
          </CardHeader>
          <CardContent>
            {isLoading && (
              <div className="space-y-3">
                {["s1", "s2", "s3", "s4", "s5"].map((id) => (
                  <Skeleton key={id} className="h-12 w-full" />
                ))}
              </div>
            )}
            {!isLoading && (!sessions || sessions.length === 0) && (
              <EmptyReports />
            )}
            {!isLoading && sessions && sessions.length > 0 && (
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Experiment</TableHead>
                    <TableHead>Target Model</TableHead>
                    <TableHead className="text-center">Turns</TableHead>
                    <TableHead className="text-center">ASR</TableHead>
                    <TableHead className="text-center">Jailbreaks</TableHead>
                    <TableHead className="text-center">Cost</TableHead>
                    <TableHead>Completed</TableHead>
                    <TableHead />
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {sessions.map((session) => (
                    <TableRow key={session.id}>
                      <TableCell className="font-medium max-w-[200px] truncate">
                        {session.experiment.name}
                      </TableCell>
                      <TableCell className="text-muted-foreground text-sm">
                        {session.experiment.target_model}
                      </TableCell>
                      <TableCell className="text-center">
                        {session.total_turns}
                      </TableCell>
                      <TableCell className="text-center">
                        <Badge
                          variant="outline"
                          className={
                            (session.asr ?? 0) > 0.3
                              ? VERDICT_BG.jailbreak
                              : VERDICT_BG.refused
                          }
                        >
                          {((session.asr ?? 0) * 100).toFixed(1)}%
                        </Badge>
                      </TableCell>
                      <TableCell className="text-center text-red-500 font-medium">
                        {session.total_jailbreaks}
                      </TableCell>
                      <TableCell className="text-center text-muted-foreground">
                        ${session.estimated_cost_usd.toFixed(4)}
                      </TableCell>
                      <TableCell className="text-muted-foreground text-sm">
                        {session.completed_at
                          ? formatDistanceToNow(new Date(session.completed_at), {
                              addSuffix: true,
                            })
                          : "—"}
                      </TableCell>
                      <TableCell>
                        <Button variant="ghost" size="sm" asChild>
                          <Link
                            href={ROUTES.reports.detail(
                              session.experiment_id,
                              session.id,
                            )}
                          >
                            View
                            <ArrowRight className="ml-1 h-3 w-3" />
                          </Link>
                        </Button>
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            )}
          </CardContent>
        </Card>
      </div>
    </>
  );
}

function StatCard({
  title,
  value,
  icon,
  valueClass,
}: Readonly<{
  title: string;
  value: string | number | null;
  icon: React.ReactNode;
  valueClass?: string;
}>) {
  return (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between pb-2">
        <CardTitle className="text-sm font-medium">{title}</CardTitle>
        {icon}
      </CardHeader>
      <CardContent>
        {value === null ? (
          <Skeleton className="h-7 w-16" />
        ) : (
          <div className={`text-2xl font-bold ${valueClass ?? ""}`}>
            {value}
          </div>
        )}
      </CardContent>
    </Card>
  );
}

function EmptyReports() {
  return (
    <div className="flex flex-col items-center justify-center py-12 text-center">
      <div className="flex h-12 w-12 items-center justify-center rounded-full bg-muted">
        <FileText className="h-6 w-6 text-muted-foreground" />
      </div>
      <h3 className="mt-4 text-sm font-medium">No reports yet</h3>
      <p className="mt-1 text-sm text-muted-foreground">
        Reports are generated when adversarial sessions complete.
      </p>
      <Button className="mt-4" size="sm" asChild>
        <Link href={ROUTES.experiments.list}>
          Go to Experiments
          <ArrowRight className="ml-2 h-4 w-4" />
        </Link>
      </Button>
    </div>
  );
}
