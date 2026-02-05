"use client";

import Link from "next/link";
import { useState } from "react";
import { formatDistanceToNow } from "date-fns";
import {
  ArrowRight,
  FlaskConical,
  MoreHorizontal,
  Plus,
  Search,
  Trash2,
} from "lucide-react";
import { toast } from "sonner";
import { Header } from "@/components/layout/header";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardHeader,
} from "@/components/ui/card";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { Input } from "@/components/ui/input";
import { Skeleton } from "@/components/ui/skeleton";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { useExperiments, useDeleteExperiment } from "@/hooks/use-experiments";
import { ROUTES } from "@/lib/constants";

export default function ExperimentsPage() {
  const [search, setSearch] = useState("");
  const { data, isLoading } = useExperiments();
  const deleteMutation = useDeleteExperiment();

  const experiments = data?.experiments ?? [];
  const filtered = search
    ? experiments.filter(
        (e) =>
          e.name.toLowerCase().includes(search.toLowerCase()) ||
          e.attack_objective.toLowerCase().includes(search.toLowerCase()) ||
          e.target_model.toLowerCase().includes(search.toLowerCase()),
      )
    : experiments;

  const handleDelete = (id: string, name: string) => {
    deleteMutation.mutate(id, {
      onSuccess: () => toast.success(`Deleted "${name}"`),
      onError: () => toast.error("Failed to delete experiment"),
    });
  };

  return (
    <>
      <Header title="Experiments" />

      <div className="flex flex-1 flex-col gap-6 p-6">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-2xl font-bold tracking-tight">Experiments</h2>
            <p className="text-muted-foreground">
              Manage and monitor your adversarial testing experiments.
            </p>
          </div>
          <Button asChild>
            <Link href={ROUTES.experiments.new}>
              <Plus className="mr-2 h-4 w-4" />
              New Experiment
            </Link>
          </Button>
        </div>

        <Card>
          <CardHeader>
            <div className="flex items-center gap-4">
              <div className="relative flex-1">
                <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
                <Input
                  placeholder="Search experiments..."
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
                <FlaskConical className="h-10 w-10 text-muted-foreground" />
                <h3 className="mt-4 text-sm font-medium">
                  {search ? "No matching experiments" : "No experiments yet"}
                </h3>
                <p className="mt-1 text-sm text-muted-foreground">
                  {search
                    ? "Try a different search term."
                    : "Create your first experiment to get started."}
                </p>
              </div>
            ) : (
              <div className="rounded-md border">
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>Name</TableHead>
                      <TableHead>Target</TableHead>
                      <TableHead>Attacker</TableHead>
                      <TableHead>Judge</TableHead>
                      <TableHead>Defender</TableHead>
                      <TableHead>Created</TableHead>
                      <TableHead className="w-12" />
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {filtered.map((exp) => (
                      <TableRow key={exp.id} className="group">
                        <TableCell>
                          <Link
                            href={ROUTES.experiments.detail(exp.id)}
                            className="font-medium hover:underline"
                          >
                            {exp.name}
                          </Link>
                          <p className="text-xs text-muted-foreground line-clamp-1 max-w-xs">
                            {exp.attack_objective}
                          </p>
                        </TableCell>
                        <TableCell>
                          <code className="rounded bg-muted px-1.5 py-0.5 text-xs">
                            {exp.target_model}
                          </code>
                        </TableCell>
                        <TableCell>
                          <code className="rounded bg-muted px-1.5 py-0.5 text-xs">
                            {exp.attacker_model}
                          </code>
                        </TableCell>
                        <TableCell>
                          <code className="rounded bg-muted px-1.5 py-0.5 text-xs">
                            {exp.analyzer_model}
                          </code>
                        </TableCell>
                        <TableCell>
                          <code className="rounded bg-muted px-1.5 py-0.5 text-xs">
                            {exp.defender_model}
                          </code>
                        </TableCell>
                        <TableCell className="text-muted-foreground text-xs">
                          {formatDistanceToNow(new Date(exp.created_at), {
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
                              <DropdownMenuItem asChild>
                                <Link
                                  href={ROUTES.experiments.detail(exp.id)}
                                >
                                  <ArrowRight className="mr-2 h-4 w-4" />
                                  View Detail
                                </Link>
                              </DropdownMenuItem>
                              <DropdownMenuItem
                                className="text-destructive"
                                onClick={() =>
                                  handleDelete(exp.id, exp.name)
                                }
                              >
                                <Trash2 className="mr-2 h-4 w-4" />
                                Delete
                              </DropdownMenuItem>
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
