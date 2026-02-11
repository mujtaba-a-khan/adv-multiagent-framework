"use client";

import {
  CheckCircle2,
  Cpu,
  Loader2,
  RefreshCw,
  Server,
  XCircle,
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
import { useModels, useHealthCheck } from "@/hooks/use-targets";

export default function TargetsPage() {
  const {
    data: modelsData,
    isLoading: modelsLoading,
    refetch: refetchModels,
    isFetching: modelsFetching,
  } = useModels();
  const { data: health, isLoading: healthLoading } = useHealthCheck();

  const models = modelsData?.models ?? [];

  return (
    <>
      <Header title="Targets & Models" />

      <div className="flex flex-1 flex-col gap-6 p-6">
        <div>
          <h2 className="text-2xl font-bold tracking-tight">
            Targets & Models
          </h2>
          <p className="text-muted-foreground">
            Manage LLM providers and browse available models.
          </p>
        </div>

        {/* Provider health */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Server className="h-5 w-5" />
              Ollama Provider
            </CardTitle>
            <CardDescription>
              Primary local LLM provider for open-source models
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="flex items-center gap-4">
              {healthLoading && <Skeleton className="h-6 w-20" />}
              {!healthLoading && health?.healthy && (
                <Badge
                  variant="outline"
                  className="border-emerald-500/20 bg-emerald-500/10 text-emerald-500"
                >
                  <CheckCircle2 className="mr-1 h-3 w-3" />
                  Online
                </Badge>
              )}
              {!healthLoading && !health?.healthy && (
                <Badge
                  variant="outline"
                  className="border-red-500/20 bg-red-500/10 text-red-500"
                >
                  <XCircle className="mr-1 h-3 w-3" />
                  Offline
                </Badge>
              )}
              <span className="text-sm text-muted-foreground">
                {health?.provider ?? "ollama"} &mdash;{" "}
                {models.length} model{models.length !== 1 && "s"} available
              </span>
            </div>
          </CardContent>
        </Card>

        {/* Model list */}
        <Card>
          <CardHeader className="flex flex-row items-center justify-between">
            <div>
              <CardTitle>Available Models</CardTitle>
              <CardDescription>
                Models detected from the Ollama instance
              </CardDescription>
            </div>
            <Button
              variant="outline"
              size="sm"
              onClick={() => refetchModels()}
              disabled={modelsFetching}
            >
              {modelsFetching ? (
                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
              ) : (
                <RefreshCw className="mr-2 h-4 w-4" />
              )}
              Refresh
            </Button>
          </CardHeader>
          <CardContent>
            {modelsLoading && (
              <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
                {["s1", "s2", "s3", "s4", "s5", "s6"].map((id) => (
                  <Skeleton key={id} className="h-16" />
                ))}
              </div>
            )}
            {!modelsLoading && models.length === 0 && (
              <div className="flex flex-col items-center justify-center py-12 text-center">
                <Cpu className="h-10 w-10 text-muted-foreground" />
                <h3 className="mt-4 text-sm font-medium">No models found</h3>
                <p className="mt-1 text-sm text-muted-foreground">
                  Ensure Ollama is running and has models pulled.
                </p>
                <code className="mt-3 rounded bg-muted px-3 py-1.5 text-xs">
                  ollama pull llama3:8b
                </code>
              </div>
            )}
            {!modelsLoading && models.length > 0 && (
              <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
                {models.map((model) => (
                  <ModelCard key={model} name={model} />
                ))}
              </div>
            )}
          </CardContent>
        </Card>

        {/* Optional providers info */}
        <Card>
          <CardHeader>
            <CardTitle>Other Providers</CardTitle>
            <CardDescription>
              Commercial API providers can be configured in Settings.
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid gap-3 sm:grid-cols-3">
              <ProviderInfo
                name="OpenAI"
                desc="GPT-4, GPT-3.5"
                status="optional"
              />
              <ProviderInfo
                name="Anthropic"
                desc="Claude models"
                status="optional"
              />
              <ProviderInfo
                name="Google"
                desc="Gemini models"
                status="optional"
              />
            </div>
          </CardContent>
        </Card>
      </div>
    </>
  );
}

function ModelCard({ name }: Readonly<{ name: string }>) {
  const family = name.split(":")[0] ?? name;
  const tag = name.includes(":") ? name.split(":")[1] : null;

  return (
    <div className="flex items-center gap-3 rounded-lg border p-3 transition-colors hover:bg-muted/50">
      <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-md bg-primary/10">
        <Cpu className="h-4 w-4 text-primary" />
      </div>
      <div className="flex-1 min-w-0">
        <p className="text-sm font-medium truncate">{family}</p>
        {tag && (
          <p className="text-xs text-muted-foreground font-mono">{tag}</p>
        )}
      </div>
    </div>
  );
}

function ProviderInfo({
  name,
  desc,
  status,
}: Readonly<{
  name: string;
  desc: string;
  status: "connected" | "optional";
}>) {
  return (
    <div className="flex items-center gap-3 rounded-lg border p-3">
      <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-md bg-muted">
        <Server className="h-4 w-4 text-muted-foreground" />
      </div>
      <div className="flex-1">
        <p className="text-sm font-medium">{name}</p>
        <p className="text-xs text-muted-foreground">{desc}</p>
      </div>
      <Badge variant="outline" className="text-[10px]">
        {status}
      </Badge>
    </div>
  );
}
