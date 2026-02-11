"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import {
  Brain,
  Check,
  Loader2,
  MessageSquare,
  Send,
  Shield,
  Terminal,
} from "lucide-react";
import { toast } from "sonner";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Switch } from "@/components/ui/switch";
import { Textarea } from "@/components/ui/textarea";
import { ROUTES } from "@/lib/constants";
import {
  useCreatePlaygroundConversation,
  useSendPlaygroundMessage,
} from "@/hooks/use-playground";
import { useModels } from "@/hooks/use-targets";
import { useDefenses } from "@/hooks/use-defenses";
import type { DefenseSelectionItem } from "@/lib/types";

function generateTitle(text: string): string {
  const maxLen = 50;
  if (text.length <= maxLen) return text;
  const cut = text.slice(0, maxLen);
  const lastSpace = cut.lastIndexOf(" ");
  return (lastSpace > 20 ? cut.slice(0, lastSpace) : cut) + "...";
}

export default function PlaygroundNewChatPage() {
  const router = useRouter();
  const [prompt, setPrompt] = useState("");
  const [targetModel, setTargetModel] = useState("");
  const [analyzerModel, setAnalyzerModel] = useState("");
  const [systemPrompt, setSystemPrompt] = useState("");
  const [selectedDefenses, setSelectedDefenses] = useState<string[]>([]);
  const [isSending, setIsSending] = useState(false);

  const { data: modelsData } = useModels();
  const { data: defensesData } = useDefenses();
  const createMutation = useCreatePlaygroundConversation();
  const sendMutation = useSendPlaygroundMessage();

  async function handleSend() {
    if (!prompt.trim() || !targetModel || isSending) return;

    setIsSending(true);
    const text = prompt.trim();
    let convId: string | null = null;

    // Step 1: Create conversation
    try {
      const defenses: DefenseSelectionItem[] = selectedDefenses.map(
        (name) => ({ name }),
      );
      const conv = await createMutation.mutateAsync({
        title: generateTitle(text),
        target_model: targetModel,
        analyzer_model: analyzerModel || undefined,
        system_prompt: systemPrompt.trim() || undefined,
        active_defenses: defenses.length > 0 ? defenses : undefined,
      });
      convId = conv.id;
    } catch {
      toast.error("Failed to create conversation");
      setIsSending(false);
      return;
    }

    // Step 2: Send first message
    try {
      await sendMutation.mutateAsync({
        conversationId: convId,
        prompt: text,
      });
    } catch {
      toast.error("Message failed to send — try again from the chat");
    }

    // Step 3: Redirect after send completes
    router.push(ROUTES.playground.detail(convId));
  }

  function handleKeyDown(e: React.KeyboardEvent) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  }

  function toggleDefense(checked: boolean, name: string) {
    setSelectedDefenses((prev) =>
      checked ? [...prev, name] : prev.filter((n) => n !== name),
    );
  }

  const canSend = prompt.trim().length > 0 && targetModel && !isSending;

  return (
    <div className="flex flex-1 flex-col">
      {/* Center area */}
      <div className="flex flex-1 items-center justify-center">
        {isSending ? (
          <div className="flex flex-col items-center text-center">
            <Loader2 className="h-10 w-10 animate-spin text-primary" />
            <p className="mt-4 text-sm font-medium">
              Sending to {targetModel}...
            </p>
            <p className="mt-1 text-xs text-muted-foreground">
              Waiting for target response and analysis
            </p>
          </div>
        ) : (
          <div className="flex flex-col items-center text-center">
            <div className="flex h-16 w-16 items-center justify-center rounded-2xl bg-primary/10">
              <MessageSquare className="h-8 w-8 text-primary" />
            </div>
            <h2 className="mt-6 text-xl font-semibold tracking-tight">
              Playground
            </h2>
            <p className="mt-2 max-w-sm text-sm text-muted-foreground">
              Test LLM defenses with manual red-teaming. Select a target model
              and start probing.
            </p>
          </div>
        )}
      </div>

      {/* Input area */}
      <div className="shrink-0 border-t p-4">
        <div className="mx-auto max-w-3xl space-y-3">
          {/* Config toolbar */}
          <div className="flex flex-wrap items-center gap-2">
            {/* Target model selector */}
            <Select value={targetModel} onValueChange={setTargetModel}>
              <SelectTrigger className="h-8 w-auto min-w-[180px] gap-2 text-xs">
                <Terminal className="h-3.5 w-3.5 shrink-0 text-muted-foreground" />
                <SelectValue placeholder="Select target model..." />
              </SelectTrigger>
              <SelectContent>
                {(modelsData?.models ?? []).map((m) => (
                  <SelectItem key={m} value={m}>
                    {m}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>

            {/* Analyzer model selector */}
            <Select value={analyzerModel} onValueChange={setAnalyzerModel}>
              <SelectTrigger className="h-8 w-auto min-w-[180px] gap-2 text-xs">
                <Brain className="h-3.5 w-3.5 shrink-0 text-muted-foreground" />
                <SelectValue placeholder="Analyzer (default)" />
              </SelectTrigger>
              <SelectContent>
                {(modelsData?.models ?? []).map((m) => (
                  <SelectItem key={m} value={m}>
                    {m}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>

            {/* Defenses popover */}
            <Popover>
              <PopoverTrigger asChild>
                <Button
                  variant="outline"
                  size="sm"
                  className="h-8 gap-1.5 text-xs"
                >
                  <Shield className="h-3.5 w-3.5" />
                  {selectedDefenses.length > 0
                    ? `Defenses (${selectedDefenses.length})`
                    : "Defenses"}
                </Button>
              </PopoverTrigger>
              <PopoverContent className="w-72" align="start">
                <p className="mb-3 text-xs font-medium">Active Defenses</p>
                {defensesData?.defenses.length === 0 ? (
                  <p className="text-xs text-muted-foreground">
                    No defenses available
                  </p>
                ) : (
                  <div className="space-y-2">
                    {defensesData?.defenses.map((d) => (
                      <div
                        key={d.name}
                        className="flex items-center justify-between rounded-md border px-2.5 py-1.5"
                      >
                        <span className="text-xs font-medium">{d.name}</span>
                        <Switch
                          checked={selectedDefenses.includes(d.name)}
                          onCheckedChange={(checked) =>
                            toggleDefense(checked, d.name)
                          }
                        />
                      </div>
                    ))}
                  </div>
                )}
              </PopoverContent>
            </Popover>

            {/* System prompt popover */}
            <Popover>
              <PopoverTrigger asChild>
                <Button
                  variant="outline"
                  size="sm"
                  className="h-8 gap-1.5 text-xs"
                >
                  {systemPrompt.trim() ? (
                    <>
                      System
                      <Check className="h-3 w-3 text-emerald-500" />
                    </>
                  ) : (
                    "System"
                  )}
                </Button>
              </PopoverTrigger>
              <PopoverContent className="w-80" align="start">
                <p className="mb-2 text-xs font-medium">System Prompt</p>
                <Textarea
                  value={systemPrompt}
                  onChange={(e) => setSystemPrompt(e.target.value)}
                  placeholder="You are a helpful assistant..."
                  rows={4}
                  className="text-xs"
                />
              </PopoverContent>
            </Popover>

            {/* Selected indicators */}
            {selectedDefenses.length > 0 && (
              <div className="flex flex-wrap gap-1">
                {selectedDefenses.map((name) => (
                  <Badge
                    key={name}
                    variant="outline"
                    className="h-5 px-1.5 text-[10px]"
                  >
                    {name}
                  </Badge>
                ))}
              </div>
            )}
          </div>

          {/* Textarea + send */}
          <div className="flex gap-2">
            <Textarea
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder={
                targetModel
                  ? "Type your attack prompt..."
                  : "Select a target model first..."
              }
              rows={2}
              disabled={!targetModel || isSending}
              className="min-h-[60px] resize-none"
            />
            <Button
              size="icon"
              onClick={handleSend}
              disabled={!canSend}
              className="shrink-0 self-end"
            >
              <Send className="h-4 w-4" />
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
}
