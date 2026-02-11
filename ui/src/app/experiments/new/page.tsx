"use client";

function getStepColor(isActive: boolean, isDone: boolean): string {
  if (isActive) return "bg-primary text-primary-foreground";
  if (isDone) return "bg-primary/10 text-primary hover:bg-primary/20";
  return "bg-muted text-muted-foreground";
}

import { useRouter } from "next/navigation";
import { useState } from "react";
import {
  ArrowLeft,
  ArrowRight,
  Check,
  FlaskConical,
  Loader2,
  Target,
  Users,
} from "lucide-react";
import { toast } from "sonner";
import { Header } from "@/components/layout/header";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Separator } from "@/components/ui/separator";
import { Textarea } from "@/components/ui/textarea";
import { useCreateExperiment } from "@/hooks/use-experiments";
import { useModels } from "@/hooks/use-targets";
import { ROUTES } from "@/lib/constants";
import type { CreateExperimentRequest } from "@/lib/types";

const STEPS = [
  { label: "Objective", icon: FlaskConical },
  { label: "Target", icon: Target },
  { label: "Agents", icon: Users },
  { label: "Review", icon: Check },
];

const DEFAULT_FORM: CreateExperimentRequest = {
  name: "",
  description: null,
  target_model: "",
  target_provider: "ollama",
  target_system_prompt: null,
  attacker_model: "phi4-reasoning:14b",
  analyzer_model: "phi4-reasoning:14b",
  defender_model: "qwen3:8b",
  attack_objective: "",
};

export default function NewExperimentPage() {
  const router = useRouter();
  const [step, setStep] = useState(0);
  const [form, setForm] = useState<CreateExperimentRequest>(DEFAULT_FORM);

  const { data: modelsData } = useModels();
  const createMutation = useCreateExperiment();

  const models = modelsData?.models ?? [];

  const set = <K extends keyof CreateExperimentRequest>(
    key: K,
    value: CreateExperimentRequest[K],
  ) => setForm((prev) => ({ ...prev, [key]: value }));

  const canNext = (): boolean => {
    switch (step) {
      case 0:
        return form.name.trim().length > 0 && form.attack_objective.trim().length > 0;
      case 1:
        return form.target_model.trim().length > 0;
      case 2:
        return true;
      case 3:
        return true;
      default:
        return false;
    }
  };

  const handleSubmit = () => {
    createMutation.mutate(form, {
      onSuccess: (exp) => {
        toast.success("Experiment created");
        router.push(ROUTES.experiments.detail(exp.id));
      },
      onError: () => toast.error("Failed to create experiment"),
    });
  };

  return (
    <>
      <Header title="New Experiment" />

      <div className="flex flex-1 flex-col gap-6 p-6">
        {/* Stepper */}
        <nav className="flex items-center justify-center gap-2">
          {STEPS.map((s, i) => {
            const Icon = s.icon;
            const isActive = i === step;
            const isDone = i < step;
            return (
              <div key={s.label} className="flex items-center gap-2">
                {i > 0 && (
                  <Separator
                    className={`w-8 ${isDone ? "bg-primary" : "bg-border"}`}
                  />
                )}
                <button
                  onClick={() => i < step && setStep(i)}
                  disabled={i > step}
                  className={`flex items-center gap-2 rounded-full px-3 py-1.5 text-xs font-medium transition-colors ${getStepColor(isActive, isDone)}`}
                >
                  <Icon className="h-3.5 w-3.5" />
                  <span className="hidden sm:inline">{s.label}</span>
                  <span className="sm:hidden">{i + 1}</span>
                </button>
              </div>
            );
          })}
        </nav>

        {/* Step content */}
        <div className="mx-auto w-full max-w-2xl">
          {step === 0 && (
            <StepCard
              title="Define Objective"
              description="Name your experiment and describe the attack objective."
            >
              <div className="space-y-4">
                <div className="space-y-2">
                  <Label htmlFor="name">Experiment Name</Label>
                  <Input
                    id="name"
                    placeholder="e.g., Llama3 Safety Experiment"
                    value={form.name}
                    onChange={(e) => set("name", e.target.value)}
                  />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="description">
                    Description{" "}
                    <span className="text-muted-foreground">(optional)</span>
                  </Label>
                  <Input
                    id="description"
                    placeholder="Brief description of the experiment..."
                    value={form.description ?? ""}
                    onChange={(e) =>
                      set("description", e.target.value || null)
                    }
                  />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="objective">Attack Objective</Label>
                  <Textarea
                    id="objective"
                    rows={4}
                    placeholder="Describe what the attacker should try to elicit from the target model..."
                    value={form.attack_objective}
                    onChange={(e) => set("attack_objective", e.target.value)}
                  />
                  <p className="text-xs text-muted-foreground">
                    This is the adversarial goal the attacker agent will
                    pursue.
                  </p>
                </div>
              </div>
            </StepCard>
          )}

          {step === 1 && (
            <StepCard
              title="Select Target"
              description="Choose which LLM to test for vulnerabilities."
            >
              <div className="space-y-4">
                <div className="space-y-2">
                  <Label>Target Model</Label>
                  <Select
                    value={form.target_model}
                    onValueChange={(v) => set("target_model", v)}
                  >
                    <SelectTrigger>
                      <SelectValue placeholder="Select a model..." />
                    </SelectTrigger>
                    <SelectContent>
                      {models.length === 0 ? (
                        <SelectItem value="_none" disabled>
                          No models found — is Ollama running?
                        </SelectItem>
                      ) : (
                        models.map((m) => (
                          <SelectItem key={m} value={m}>
                            {m}
                          </SelectItem>
                        ))
                      )}
                    </SelectContent>
                  </Select>
                </div>
                <div className="space-y-2">
                  <Label>Provider</Label>
                  <Select
                    value={form.target_provider}
                    onValueChange={(v) => set("target_provider", v)}
                  >
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="ollama">Ollama (Local)</SelectItem>
                      <SelectItem value="openai">OpenAI</SelectItem>
                      <SelectItem value="anthropic">Anthropic</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                <div className="space-y-2">
                  <Label htmlFor="system_prompt">
                    Target System Prompt{" "}
                    <span className="text-muted-foreground">(optional)</span>
                  </Label>
                  <Textarea
                    id="system_prompt"
                    rows={3}
                    placeholder="Custom system prompt for the target model..."
                    value={form.target_system_prompt ?? ""}
                    onChange={(e) =>
                      set("target_system_prompt", e.target.value || null)
                    }
                  />
                </div>
              </div>
            </StepCard>
          )}

          {step === 2 && (
            <StepCard
              title="Assign Agent Models"
              description="Choose which LLM powers each agent role."
            >
              <div className="space-y-4">
                <AgentModelSelector
                  label="Attacker Model"
                  description="Generates adversarial prompts to jailbreak the target."
                  value={form.attacker_model ?? ""}
                  models={models}
                  onChange={(v) => set("attacker_model", v)}
                />
                <Separator />
                <AgentModelSelector
                  label="Analyzer / Judge Model"
                  description="Evaluates whether the target response is a jailbreak."
                  value={form.analyzer_model ?? ""}
                  models={models}
                  onChange={(v) => set("analyzer_model", v)}
                />
                <Separator />
                <AgentModelSelector
                  label="Defender Model"
                  description="Generates guardrails when jailbreaks are detected."
                  value={form.defender_model ?? ""}
                  models={models}
                  onChange={(v) => set("defender_model", v)}
                />
                <Separator />
                <div>
                  <p className="text-xs font-medium text-muted-foreground mb-2">
                    Presets
                  </p>
                  <div className="flex gap-2">
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => {
                        set("attacker_model", "phi4-reasoning:14b");
                        set("analyzer_model", "phi4-reasoning:14b");
                        set("defender_model", "qwen3:8b");
                      }}
                    >
                      Optimized
                    </Button>
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => {
                        const model = form.target_model || models[0] || "";
                        set("attacker_model", model);
                        set("analyzer_model", model);
                        set("defender_model", model);
                      }}
                    >
                      Balanced (same model)
                    </Button>
                  </div>
                </div>
              </div>
            </StepCard>
          )}

          {step === 3 && (
            <StepCard
              title="Review & Launch"
              description="Verify all settings before creating the experiment."
            >
              <div className="space-y-4">
                <ReviewRow label="Name" value={form.name} />
                <ReviewRow
                  label="Objective"
                  value={form.attack_objective}
                  truncate
                />
                <Separator />
                <ReviewRow label="Target Model" value={form.target_model} />
                <ReviewRow label="Provider" value={form.target_provider ?? "ollama"} />
                <Separator />
                <ReviewRow
                  label="Attacker Model"
                  value={form.attacker_model ?? ""}
                />
                <ReviewRow
                  label="Analyzer Model"
                  value={form.analyzer_model ?? ""}
                />
                <ReviewRow
                  label="Defender Model"
                  value={form.defender_model ?? ""}
                />
              </div>
            </StepCard>
          )}

          {/* Navigation buttons */}
          <div className="mt-6 flex justify-between">
            <Button
              variant="outline"
              onClick={() => setStep((s) => s - 1)}
              disabled={step === 0}
            >
              <ArrowLeft className="mr-2 h-4 w-4" />
              Back
            </Button>

            {step < STEPS.length - 1 ? (
              <Button
                onClick={() => setStep((s) => s + 1)}
                disabled={!canNext()}
              >
                Next
                <ArrowRight className="ml-2 h-4 w-4" />
              </Button>
            ) : (
              <Button
                onClick={handleSubmit}
                disabled={createMutation.isPending}
              >
                {createMutation.isPending ? (
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                ) : (
                  <FlaskConical className="mr-2 h-4 w-4" />
                )}
                Create Experiment
              </Button>
            )}
          </div>
        </div>
      </div>
    </>
  );
}

function StepCard({
  title,
  description,
  children,
}: Readonly<{
  title: string;
  description: string;
  children: React.ReactNode;
}>) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>{title}</CardTitle>
        <CardDescription>{description}</CardDescription>
      </CardHeader>
      <CardContent>{children}</CardContent>
    </Card>
  );
}

function AgentModelSelector({
  label,
  description,
  value,
  models,
  onChange,
}: Readonly<{
  label: string;
  description: string;
  value: string;
  models: string[];
  onChange: (val: string) => void;
}>) {
  return (
    <div className="space-y-2">
      <Label>{label}</Label>
      <p className="text-xs text-muted-foreground">{description}</p>
      <Select value={value} onValueChange={onChange}>
        <SelectTrigger>
          <SelectValue placeholder="Select model..." />
        </SelectTrigger>
        <SelectContent>
          {models.map((m) => (
            <SelectItem key={m} value={m}>
              {m}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
    </div>
  );
}

function ReviewRow({
  label,
  value,
  truncate,
}: Readonly<{
  label: string;
  value: string;
  truncate?: boolean;
}>) {
  return (
    <div className="flex items-start justify-between gap-4">
      <span className="text-sm text-muted-foreground">{label}</span>
      <span
        className={`text-sm font-medium text-right ${truncate ? "line-clamp-2 max-w-sm" : ""}`}
      >
        {value}
      </span>
    </div>
  );
}
