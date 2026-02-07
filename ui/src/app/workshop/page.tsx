"use client";

import Link from "next/link";
import { useState } from "react";
import { formatDistanceToNow } from "date-fns";
import {
  Download,
  MoreHorizontal,
  Play,
  Plus,
  Search,
  Trash2,
  Wrench,
  XCircle,
} from "lucide-react";
import { toast } from "sonner";
import { Header } from "@/components/layout/header";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardHeader,
} from "@/components/ui/card";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Progress } from "@/components/ui/progress";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Skeleton } from "@/components/ui/skeleton";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import {
  useFineTuningJobs,
  useCreateFineTuningJob,
  useStartFineTuningJob,
  useCancelFineTuningJob,
  useDeleteFineTuningJob,
} from "@/hooks/use-finetuning";
import {
  ROUTES,
  STATUS_COLORS,
  JOB_TYPE_LABELS,
  JOB_TYPE_COLORS,
} from "@/lib/constants";
import type { FineTuningJobType } from "@/lib/types";

export default function WorkshopPage() {
  const [search, setSearch] = useState("");
  const [dialogOpen, setDialogOpen] = useState(false);
  const [jobType, setJobType] = useState<FineTuningJobType>("pull_abliterated");
  const [formName, setFormName] = useState("");
  const [sourceModel, setSourceModel] = useState("");
  const [outputName, setOutputName] = useState("");

  const { data, isLoading } = useFineTuningJobs();
  const createMutation = useCreateFineTuningJob();
  const startMutation = useStartFineTuningJob();
  const cancelMutation = useCancelFineTuningJob();
  const deleteMutation = useDeleteFineTuningJob();

  const jobs = data?.jobs ?? [];
  const filtered = search
    ? jobs.filter(
        (j) =>
          j.name.toLowerCase().includes(search.toLowerCase()) ||
          j.source_model.toLowerCase().includes(search.toLowerCase()) ||
          j.output_model_name.toLowerCase().includes(search.toLowerCase()),
      )
    : jobs;

  const handleCreate = () => {
    if (!formName || !sourceModel || !outputName) {
      toast.error("All fields are required");
      return;
    }
    createMutation.mutate(
      {
        name: formName,
        job_type: jobType,
        source_model: sourceModel,
        output_model_name: outputName,
      },
      {
        onSuccess: () => {
          toast.success("Job created");
          setDialogOpen(false);
          setFormName("");
          setSourceModel("");
          setOutputName("");
        },
        onError: () => toast.error("Failed to create job"),
      },
    );
  };

  const handleStart = (id: string) => {
    startMutation.mutate(id, {
      onSuccess: () => toast.success("Job started"),
      onError: () => toast.error("Failed to start job"),
    });
  };

  const handleCancel = (id: string) => {
    cancelMutation.mutate(id, {
      onSuccess: () => toast.success("Cancellation requested"),
      onError: () => toast.error("Failed to cancel job"),
    });
  };

  const handleDelete = (id: string, name: string) => {
    deleteMutation.mutate(id, {
      onSuccess: () => toast.success(`Deleted "${name}"`),
      onError: () => toast.error("Failed to delete job"),
    });
  };

  return (
    <>
      <Header title="Model Workshop" />

      <div className="flex flex-1 flex-col gap-6 p-6">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-2xl font-bold tracking-tight">
              Model Workshop
            </h2>
            <p className="text-muted-foreground">
              Fine-tune and abliterate models to reduce refusals and improve
              openness for red-teaming.
            </p>
          </div>

          <Dialog open={dialogOpen} onOpenChange={setDialogOpen}>
            <DialogTrigger asChild>
              <Button>
                <Plus className="mr-2 h-4 w-4" />
                New Job
              </Button>
            </DialogTrigger>
            <DialogContent>
              <DialogHeader>
                <DialogTitle>Create Fine-Tuning Job</DialogTitle>
                <DialogDescription>
                  Set up a new model modification job.
                </DialogDescription>
              </DialogHeader>

              <div className="grid gap-4 py-4">
                <div className="grid gap-2">
                  <Label>Job Name</Label>
                  <Input
                    placeholder="My abliterated llama"
                    value={formName}
                    onChange={(e) => setFormName(e.target.value)}
                  />
                </div>

                <div className="grid gap-2">
                  <Label>Job Type</Label>
                  <Select
                    value={jobType}
                    onValueChange={(v) =>
                      setJobType(v as FineTuningJobType)
                    }
                  >
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="pull_abliterated">
                        <div className="flex items-center gap-2">
                          <Download className="h-3.5 w-3.5" />
                          Pull Abliterated (pre-built)
                        </div>
                      </SelectItem>
                      <SelectItem value="abliterate">
                        <div className="flex items-center gap-2">
                          <Wrench className="h-3.5 w-3.5" />
                          Custom Abliterate
                        </div>
                      </SelectItem>
                      <SelectItem value="sft">
                        <div className="flex items-center gap-2">
                          <Wrench className="h-3.5 w-3.5" />
                          Fine-Tune (SFT / QLoRA)
                        </div>
                      </SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                <div className="grid gap-2">
                  <Label>Source Model</Label>
                  <Input
                    placeholder={
                      jobType === "pull_abliterated"
                        ? "huihui_ai/qwen3-abliterated:8b"
                        : "llama3:8b"
                    }
                    value={sourceModel}
                    onChange={(e) => setSourceModel(e.target.value)}
                  />
                  <p className="text-xs text-muted-foreground">
                    {jobType === "pull_abliterated"
                      ? "Ollama Hub tag for the pre-built abliterated model."
                      : jobType === "abliterate"
                        ? "HuggingFace model ID (e.g. meta-llama/Meta-Llama-3-8B-Instruct)."
                        : "HuggingFace model ID. Max 8B params on 16GB RAM."}
                  </p>
                </div>

                <div className="grid gap-2">
                  <Label>Output Model Name</Label>
                  <Input
                    placeholder="my-abliterated-llama3"
                    value={outputName}
                    onChange={(e) => setOutputName(e.target.value)}
                  />
                  <p className="text-xs text-muted-foreground">
                    Name for the model in Ollama after processing.
                  </p>
                </div>
              </div>

              <DialogFooter>
                <Button
                  variant="outline"
                  onClick={() => setDialogOpen(false)}
                >
                  Cancel
                </Button>
                <Button
                  onClick={handleCreate}
                  disabled={createMutation.isPending}
                >
                  {createMutation.isPending ? "Creating..." : "Create Job"}
                </Button>
              </DialogFooter>
            </DialogContent>
          </Dialog>
        </div>

        <Card>
          <CardHeader>
            <div className="flex items-center gap-4">
              <div className="relative flex-1">
                <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
                <Input
                  placeholder="Search jobs..."
                  value={search}
                  onChange={(e) => setSearch(e.target.value)}
                  className="pl-9"
                />
              </div>
            </div>
          </CardHeader>
          <CardContent>
            {isLoading ? (
              <div className="space-y-3">
                {Array.from({ length: 5 }).map((_, i) => (
                  <Skeleton key={i} className="h-14 w-full" />
                ))}
              </div>
            ) : filtered.length === 0 ? (
              <div className="flex flex-col items-center justify-center py-16 text-center">
                <Wrench className="h-10 w-10 text-muted-foreground" />
                <h3 className="mt-4 text-sm font-medium">
                  {search ? "No matching jobs" : "No jobs yet"}
                </h3>
                <p className="mt-1 text-sm text-muted-foreground">
                  {search
                    ? "Try a different search term."
                    : "Create your first fine-tuning job to get started."}
                </p>
              </div>
            ) : (
              <div className="rounded-md border">
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>Name</TableHead>
                      <TableHead>Type</TableHead>
                      <TableHead>Source</TableHead>
                      <TableHead>Output</TableHead>
                      <TableHead>Status</TableHead>
                      <TableHead>Progress</TableHead>
                      <TableHead>Created</TableHead>
                      <TableHead className="w-12" />
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {filtered.map((job) => (
                      <TableRow key={job.id} className="group">
                        <TableCell>
                          <Link
                            href={ROUTES.workshop.detail(job.id)}
                            className="font-medium hover:underline"
                          >
                            {job.name}
                          </Link>
                        </TableCell>
                        <TableCell>
                          <Badge
                            variant="outline"
                            className={JOB_TYPE_COLORS[job.job_type] ?? ""}
                          >
                            {JOB_TYPE_LABELS[job.job_type] ?? job.job_type}
                          </Badge>
                        </TableCell>
                        <TableCell>
                          <code className="rounded bg-muted px-1.5 py-0.5 text-xs">
                            {job.source_model}
                          </code>
                        </TableCell>
                        <TableCell>
                          <code className="rounded bg-muted px-1.5 py-0.5 text-xs">
                            {job.output_model_name}
                          </code>
                        </TableCell>
                        <TableCell>
                          <Badge
                            variant="outline"
                            className={STATUS_COLORS[job.status] ?? ""}
                          >
                            {job.status}
                          </Badge>
                        </TableCell>
                        <TableCell className="min-w-[120px]">
                          {job.status === "running" ? (
                            <div className="flex items-center gap-2">
                              <Progress
                                value={job.progress_pct}
                                className="h-2 flex-1"
                              />
                              <span className="text-xs text-muted-foreground w-10 text-right">
                                {Math.round(job.progress_pct)}%
                              </span>
                            </div>
                          ) : job.status === "completed" ? (
                            <span className="text-xs text-emerald-500">
                              100%
                            </span>
                          ) : (
                            <span className="text-xs text-muted-foreground">
                              —
                            </span>
                          )}
                        </TableCell>
                        <TableCell className="text-muted-foreground text-xs">
                          {formatDistanceToNow(new Date(job.created_at), {
                            addSuffix: true,
                          })}
                        </TableCell>
                        <TableCell>
                          <DropdownMenu>
                            <DropdownMenuTrigger asChild>
                              <Button
                                variant="ghost"
                                size="icon"
                                className="h-8 w-8 opacity-0 group-hover:opacity-100"
                              >
                                <MoreHorizontal className="h-4 w-4" />
                              </Button>
                            </DropdownMenuTrigger>
                            <DropdownMenuContent align="end">
                              {job.status === "pending" && (
                                <DropdownMenuItem
                                  onClick={() => handleStart(job.id)}
                                >
                                  <Play className="mr-2 h-4 w-4" />
                                  Start
                                </DropdownMenuItem>
                              )}
                              {job.status === "running" && (
                                <DropdownMenuItem
                                  onClick={() => handleCancel(job.id)}
                                >
                                  <XCircle className="mr-2 h-4 w-4" />
                                  Cancel
                                </DropdownMenuItem>
                              )}
                              {job.status !== "running" && (
                                <DropdownMenuItem
                                  className="text-destructive"
                                  onClick={() =>
                                    handleDelete(job.id, job.name)
                                  }
                                >
                                  <Trash2 className="mr-2 h-4 w-4" />
                                  Delete
                                </DropdownMenuItem>
                              )}
                            </DropdownMenuContent>
                          </DropdownMenu>
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </div>
            )}
          </CardContent>
        </Card>
      </div>
    </>
  );
}
