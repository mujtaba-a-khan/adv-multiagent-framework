"use client";

import Link from "next/link";
import { useParams, useRouter } from "next/navigation";
import { useEffect, useRef } from "react";
import { formatDistanceToNow } from "date-fns";
import {
  ArrowLeft,
  CheckCircle2,
  Clock,
  Play,
  Trash2,
  XCircle,
} from "lucide-react";
import { toast } from "sonner";
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
import { Progress } from "@/components/ui/progress";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Skeleton } from "@/components/ui/skeleton";
import {
  useFineTuningJob,
  useStartFineTuningJob,
  useCancelFineTuningJob,
  useDeleteFineTuningJob,
} from "@/hooks/use-finetuning";
import { useFTWSStore } from "@/stores/ft-ws-store";
import {
  ROUTES,
  STATUS_COLORS,
  JOB_TYPE_LABELS,
  JOB_TYPE_COLORS,
} from "@/lib/constants";

export default function WorkshopJobDetailPage() {
  const params = useParams();
  const router = useRouter();
  const jobId = params.jobId as string;
  const logEndRef = useRef<HTMLDivElement>(null);

  const { data: job, isLoading } = useFineTuningJob(jobId);
  const startMutation = useStartFineTuningJob();
  const cancelMutation = useCancelFineTuningJob();
  const deleteMutation = useDeleteFineTuningJob();

  const {
    connect,
    disconnect,
    progressPct: wsProgress,
    currentStep: wsStep,
    status: wsStatus,
    logs,
    error: wsError,
    outputModel,
    durationS,
  } = useFTWSStore();

  // Connect WS when job is running
  useEffect(() => {
    if (job?.status === "running") {
      connect(jobId);
    }
    return () => disconnect();
  }, [jobId, job?.status, connect, disconnect]);

  // Auto-scroll logs
  useEffect(() => {
    logEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [logs.length]);

  if (isLoading) {
    return (
      <>
        <Header title="Model Workshop" />
        <div className="flex flex-1 flex-col gap-6 p-6">
          <Skeleton className="h-8 w-64" />
          <Skeleton className="h-48 w-full" />
        </div>
      </>
    );
  }

  if (!job) {
    return (
      <>
        <Header title="Model Workshop" />
        <div className="flex flex-1 flex-col items-center justify-center p-6">
          <p className="text-muted-foreground">Job not found.</p>
          <Button variant="link" asChild className="mt-2">
            <Link href={ROUTES.workshop.list}>Back to Workshop</Link>
          </Button>
        </div>
      </>
    );
  }

  // Use WS data when available, fall back to REST-polled data
  const progress = wsProgress > 0 ? wsProgress : job.progress_pct;
  const step = wsStep ?? job.current_step;
  const status = wsStatus ?? job.status;
  const error = wsError ?? job.error_message;

  const handleStart = () => {
    startMutation.mutate(jobId, {
      onSuccess: () => toast.success("Job started"),
      onError: () => toast.error("Failed to start job"),
    });
  };

  const handleCancel = () => {
    cancelMutation.mutate(jobId, {
      onSuccess: () => toast.success("Cancellation requested"),
      onError: () => toast.error("Failed to cancel"),
    });
  };

  const handleDelete = () => {
    deleteMutation.mutate(jobId, {
      onSuccess: () => {
        toast.success("Job deleted");
        router.push(ROUTES.workshop.list);
      },
      onError: () => toast.error("Failed to delete"),
    });
  };

  const duration =
    durationS ?? job.total_duration_seconds;

  return (
    <>
      <Header title="Model Workshop" />

      <div className="flex flex-1 flex-col gap-6 p-6">
        {/* Back link + title */}
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <Button variant="ghost" size="icon" asChild>
              <Link href={ROUTES.workshop.list}>
                <ArrowLeft className="h-4 w-4" />
              </Link>
            </Button>
            <div>
              <h2 className="text-2xl font-bold tracking-tight">{job.name}</h2>
              <div className="flex items-center gap-2 mt-1">
                <Badge
                  variant="outline"
                  className={JOB_TYPE_COLORS[job.job_type] ?? ""}
                >
                  {JOB_TYPE_LABELS[job.job_type] ?? job.job_type}
                </Badge>
                <Badge
                  variant="outline"
                  className={STATUS_COLORS[status] ?? ""}
                >
                  {status}
                </Badge>
              </div>
            </div>
          </div>

          <div className="flex gap-2">
            {status === "pending" && (
              <Button onClick={handleStart} disabled={startMutation.isPending}>
                <Play className="mr-2 h-4 w-4" />
                Start
              </Button>
            )}
            {status === "running" && (
              <Button
                variant="outline"
                onClick={handleCancel}
                disabled={cancelMutation.isPending}
              >
                <XCircle className="mr-2 h-4 w-4" />
                Cancel
              </Button>
            )}
            {status !== "running" && (
              <Button
                variant="destructive"
                onClick={handleDelete}
                disabled={deleteMutation.isPending}
              >
                <Trash2 className="mr-2 h-4 w-4" />
                Delete
              </Button>
            )}
          </div>
        </div>

        {/* Progress bar */}
        {(status === "running" || status === "completed") && (
          <Card>
            <CardContent className="pt-6">
              <div className="flex items-center gap-4">
                <Progress value={progress} className="flex-1 h-3" />
                <span className="text-sm font-medium w-12 text-right">
                  {Math.round(progress)}%
                </span>
              </div>
              {step && (
                <p className="mt-2 text-sm text-muted-foreground">{step}</p>
              )}
              {status === "completed" && (
                <div className="mt-3 flex items-center gap-2 text-sm text-emerald-500">
                  <CheckCircle2 className="h-4 w-4" />
                  <span>
                    Completed
                    {outputModel && (
                      <>
                        {" "}
                        &mdash; model{" "}
                        <code className="rounded bg-muted px-1.5 py-0.5 text-xs">
                          {outputModel}
                        </code>{" "}
                        ready in Ollama
                      </>
                    )}
                  </span>
                </div>
              )}
              {error && (
                <p className="mt-3 text-sm text-red-500">Error: {error}</p>
              )}
            </CardContent>
          </Card>
        )}

        {/* Details grid */}
        <div className="grid gap-6 md:grid-cols-2">
          <Card>
            <CardHeader>
              <CardTitle className="text-base">Job Details</CardTitle>
            </CardHeader>
            <CardContent className="grid gap-3 text-sm">
              <div className="flex justify-between">
                <span className="text-muted-foreground">Source Model</span>
                <code className="rounded bg-muted px-1.5 py-0.5 text-xs">
                  {job.source_model}
                </code>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">Output Name</span>
                <code className="rounded bg-muted px-1.5 py-0.5 text-xs">
                  {job.output_model_name}
                </code>
              </div>
              {duration != null && (
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Duration</span>
                  <span className="flex items-center gap-1">
                    <Clock className="h-3.5 w-3.5" />
                    {Math.round(duration)}s
                  </span>
                </div>
              )}
              {job.peak_memory_gb != null && (
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Peak Memory</span>
                  <span>{job.peak_memory_gb.toFixed(1)} GB</span>
                </div>
              )}
              <div className="flex justify-between">
                <span className="text-muted-foreground">Created</span>
                <span>
                  {formatDistanceToNow(new Date(job.created_at), {
                    addSuffix: true,
                  })}
                </span>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle className="text-base">Configuration</CardTitle>
              <CardDescription>
                Job-specific parameters
              </CardDescription>
            </CardHeader>
            <CardContent>
              {Object.keys(job.config).length === 0 ? (
                <p className="text-sm text-muted-foreground">
                  Default configuration
                </p>
              ) : (
                <pre className="rounded bg-muted p-3 text-xs overflow-auto max-h-48">
                  {JSON.stringify(job.config, null, 2)}
                </pre>
              )}
            </CardContent>
          </Card>
        </div>

        {/* Live log stream */}
        {(status === "running" || logs.length > 0) && (
          <Card>
            <CardHeader>
              <CardTitle className="text-base">Live Logs</CardTitle>
            </CardHeader>
            <CardContent>
              <ScrollArea className="h-64 rounded border bg-zinc-950 p-3">
                <div className="space-y-1 font-mono text-xs">
                  {logs.length === 0 && (
                    <p className="text-zinc-500">Waiting for log output...</p>
                  )}
                  {logs.map((entry, i) => (
                    <div key={i} className="flex gap-2">
                      <span
                        className={
                          entry.level === "error"
                            ? "text-red-400"
                            : entry.level === "warning"
                              ? "text-amber-400"
                              : "text-zinc-400"
                        }
                      >
                        [{entry.level.toUpperCase()}]
                      </span>
                      <span className="text-zinc-200">{entry.message}</span>
                    </div>
                  ))}
                  <div ref={logEndRef} />
                </div>
              </ScrollArea>
            </CardContent>
          </Card>
        )}
      </div>
    </>
  );
}
