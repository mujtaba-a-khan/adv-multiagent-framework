"use client";

import Link from "next/link";
import { formatDistanceToNow } from "date-fns";
import {
  ArrowRight,
  Beaker,
  FlaskConical,
  Plus,
  Server,
  Shield,
  Swords,
  Target,
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
import { useExperiments } from "@/hooks/use-experiments";
import { useStrategies } from "@/hooks/use-strategies";
import { useHealthCheck } from "@/hooks/use-targets";
import { ROUTES, CATEGORY_LABELS } from "@/lib/constants";

export default function DashboardPage() {
  const { data: expData, isLoading: expLoading } = useExperiments(0, 5);
  const { data: stratData, isLoading: stratLoading } = useStrategies();
  const { data: health } = useHealthCheck();

  const experiments = expData?.experiments ?? [];
  const totalExperiments = expData?.total ?? 0;
  const totalStrategies = stratData?.total ?? 0;

  return (
    <>
      <Header title="Dashboard" />

      <div className="flex flex-1 flex-col gap-6 p-6">
        {/* Stats row */}
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
          <StatsCard
            title="Experiments"
            value={expLoading ? null : totalExperiments}
            description="Total created"
            icon={<FlaskConical className="h-4 w-4 text-muted-foreground" />}
          />
          <StatsCard
            title="Attack Strategies"
            value={stratLoading ? null : totalStrategies}
            description="Available in registry"
            icon={<Swords className="h-4 w-4 text-muted-foreground" />}
          />
          <StatsCard
            title="Ollama Status"
            value={health ? (health.healthy ? "Online" : "Offline") : null}
            description={health?.provider ?? "Checking..."}
            icon={<Target className="h-4 w-4 text-muted-foreground" />}
            valueClassName={
              health?.healthy ? "text-emerald-500" : "text-red-500"
            }
          />
          <StatsCard
            title="Framework"
            value="v0.1"
            description="Multi-agent adversarial"
            icon={<Shield className="h-4 w-4 text-muted-foreground" />}
          />
        </div>

        <div className="grid gap-6 lg:grid-cols-3">
          {/* Recent experiments */}
          <Card className="lg:col-span-2">
            <CardHeader className="flex flex-row items-center justify-between">
              <div>
                <CardTitle>Recent Experiments</CardTitle>
                <CardDescription>
                  Latest adversarial testing runs
                </CardDescription>
              </div>
              <Button asChild size="sm">
                <Link href={ROUTES.experiments.new}>
                  <Plus className="mr-2 h-4 w-4" />
                  New Experiment
                </Link>
              </Button>
            </CardHeader>
            <CardContent>
              {expLoading ? (
                <div className="space-y-4">
                  {Array.from({ length: 3 }).map((_, i) => (
                    <Skeleton key={i} className="h-16 w-full" />
                  ))}
                </div>
              ) : experiments.length === 0 ? (
                <EmptyState />
              ) : (
                <div className="space-y-3">
                  {experiments.map((exp) => (
                    <Link
                      key={exp.id}
                      href={ROUTES.experiments.detail(exp.id)}
                      className="flex items-center gap-4 rounded-lg border p-4 transition-colors hover:bg-muted/50"
                    >
                      <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-lg bg-primary/10">
                        <FlaskConical className="h-5 w-5 text-primary" />
                      </div>
                      <div className="flex-1 space-y-1">
                        <div className="flex items-center gap-2">
                          <p className="text-sm font-medium leading-none">
                            {exp.name}
                          </p>
                          <Badge
                            variant="outline"
                            className="text-[10px] uppercase tracking-wider"
                          >
                            {CATEGORY_LABELS[exp.strategy_name] ??
                              exp.strategy_name}
                          </Badge>
                        </div>
                        <p className="text-xs text-muted-foreground line-clamp-1">
                          {exp.attack_objective}
                        </p>
                      </div>
                      <div className="flex flex-col items-end gap-1">
                        <span className="text-xs text-muted-foreground">
                          {formatDistanceToNow(new Date(exp.created_at), {
                            addSuffix: true,
                          })}
                        </span>
                        <span className="text-[10px] text-muted-foreground">
                          {exp.target_model}
                        </span>
                      </div>
                      <ArrowRight className="h-4 w-4 text-muted-foreground" />
                    </Link>
                  ))}
                </div>
              )}
              {experiments.length > 0 && (
                <div className="mt-4 text-center">
                  <Button variant="ghost" size="sm" asChild>
                    <Link href={ROUTES.experiments.list}>
                      View all experiments
                      <ArrowRight className="ml-2 h-4 w-4" />
                    </Link>
                  </Button>
                </div>
              )}
            </CardContent>
          </Card>

          {/* Quick actions + strategy preview */}
          <div className="flex flex-col gap-6">
            <Card>
              <CardHeader>
                <CardTitle>Quick Actions</CardTitle>
              </CardHeader>
              <CardContent className="grid gap-2">
                <Button className="w-full justify-start" asChild>
                  <Link href={ROUTES.experiments.new}>
                    <Plus className="mr-2 h-4 w-4" />
                    New Experiment
                  </Link>
                </Button>
                <Button
                  variant="outline"
                  className="w-full justify-start"
                  asChild
                >
                  <Link href={ROUTES.strategies}>
                    <Swords className="mr-2 h-4 w-4" />
                    Browse Strategies
                  </Link>
                </Button>
                <Button
                  variant="outline"
                  className="w-full justify-start"
                  asChild
                >
                  <Link href={ROUTES.targets}>
                    <Server className="mr-2 h-4 w-4" />
                    Manage Models
                  </Link>
                </Button>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Strategy Categories</CardTitle>
                <CardDescription>
                  {totalStrategies} strategies across 5 categories
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  {Object.entries(CATEGORY_LABELS).map(([key, label]) => (
                    <div key={key} className="flex items-center gap-3">
                      <div className="h-2 w-2 rounded-full bg-primary" />
                      <span className="text-sm">{label}</span>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          </div>
        </div>
      </div>
    </>
  );
}

function StatsCard({
  title,
  value,
  description,
  icon,
  valueClassName,
}: {
  title: string;
  value: string | number | null;
  description: string;
  icon: React.ReactNode;
  valueClassName?: string;
}) {
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
          <div className={`text-2xl font-bold ${valueClassName ?? ""}`}>
            {value}
          </div>
        )}
        <p className="text-xs text-muted-foreground">{description}</p>
      </CardContent>
    </Card>
  );
}

function EmptyState() {
  return (
    <div className="flex flex-col items-center justify-center py-12 text-center">
      <div className="flex h-12 w-12 items-center justify-center rounded-full bg-muted">
        <Beaker className="h-6 w-6 text-muted-foreground" />
      </div>
      <h3 className="mt-4 text-sm font-medium">No experiments yet</h3>
      <p className="mt-1 text-sm text-muted-foreground">
        Get started by creating your first adversarial experiment.
      </p>
      <Button className="mt-4" size="sm" asChild>
        <Link href={ROUTES.experiments.new}>
          <Plus className="mr-2 h-4 w-4" />
          Create Experiment
        </Link>
      </Button>
    </div>
  );
}
