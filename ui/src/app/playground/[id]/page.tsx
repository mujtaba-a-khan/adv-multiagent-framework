"use client";

import { useParams } from "next/navigation";
import { useCallback, useEffect, useRef, useState } from "react";
import {
  ArrowDown,
  Brain,
  Check,
  ChevronDown,
  Pencil,
  Send,
  Shield,
  Target,
  Terminal,
  User,
} from "lucide-react";
import { toast } from "sonner";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import { Input } from "@/components/ui/input";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import { Separator } from "@/components/ui/separator";
import { Skeleton } from "@/components/ui/skeleton";
import { Switch } from "@/components/ui/switch";
import { Textarea } from "@/components/ui/textarea";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { VERDICT_BG } from "@/lib/constants";
import {
  usePlaygroundConversation,
  usePlaygroundMessages,
  useSendPlaygroundMessage,
  useUpdatePlaygroundConversation,
} from "@/hooks/use-playground";
import { useDefenses } from "@/hooks/use-defenses";
import { usePGWSStore } from "@/stores/pg-ws-store";
import type {
  PlaygroundMessage,
  DefenseSelectionItem,
} from "@/lib/types";

const SCROLL_THRESHOLD = 150;

export default function PlaygroundChatPage() {
  const params = useParams<{ id: string }>();
  const conversationId = params.id;

  const { data: conversation, isLoading: convLoading } =
    usePlaygroundConversation(conversationId);
  const { data: messagesData, isLoading: msgsLoading } =
    usePlaygroundMessages(conversationId);
  const sendMutation = useSendPlaygroundMessage();
  const updateMutation = useUpdatePlaygroundConversation();
  const { data: defensesData } = useDefenses();
  const {
    connect: wsConnect,
    disconnect: wsDisconnect,
    phase: wsPhase,
  } = usePGWSStore();

  const [prompt, setPrompt] = useState("");
  const [systemPrompt, setSystemPrompt] = useState<string | null>(null);
  const [editingTitle, setEditingTitle] = useState(false);
  const [titleDraft, setTitleDraft] = useState("");
  const chatRef = useRef<HTMLDivElement>(null);
  const [isNearBottom, setIsNearBottom] = useState(true);
  const [showJumpBtn, setShowJumpBtn] = useState(false);

  // Connect WebSocket
  useEffect(() => {
    if (conversationId) {
      wsConnect(conversationId);
      return () => wsDisconnect();
    }
  }, [conversationId, wsConnect, wsDisconnect]);

  // Sync system prompt from conversation on load
  useEffect(() => {
    if (conversation && systemPrompt === null) {
      setSystemPrompt(conversation.system_prompt ?? "");
    }
  }, [conversation, systemPrompt]);

  const messages = messagesData?.messages ?? [];

  // Scroll tracking
  const handleScroll = useCallback(() => {
    const el = chatRef.current;
    if (!el) return;
    const nearBottom =
      el.scrollHeight - el.scrollTop - el.clientHeight < SCROLL_THRESHOLD;
    setIsNearBottom(nearBottom);
    setShowJumpBtn(!nearBottom);
  }, []);

  // Auto-scroll on new messages
  useEffect(() => {
    if (isNearBottom && chatRef.current) {
      chatRef.current.scrollTo({
        top: chatRef.current.scrollHeight,
        behavior: "smooth",
      });
    }
  }, [messages.length, sendMutation.isPending, isNearBottom]);

  const jumpToLatest = useCallback(() => {
    chatRef.current?.scrollTo({
      top: chatRef.current.scrollHeight,
      behavior: "smooth",
    });
  }, []);

  async function handleSend() {
    if (!prompt.trim() || sendMutation.isPending) return;
    const text = prompt.trim();
    setPrompt("");
    try {
      await sendMutation.mutateAsync({
        conversationId,
        prompt: text,
      });
    } catch {
      toast.error("Failed to send message");
      setPrompt(text);
    }
  }

  function handleKeyDown(e: React.KeyboardEvent) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  }

  // Toggle a defense on/off
  function toggleDefense(defenseName: string, enabled: boolean) {
    if (!conversation) return;
    const current: DefenseSelectionItem[] =
      conversation.active_defenses ?? [];
    const updated = enabled
      ? [...current, { name: defenseName }]
      : current.filter((d) => d.name !== defenseName);
    updateMutation.mutate(
      { id: conversationId, data: { active_defenses: updated } },
      {
        onSuccess: () => toast.success("Defenses updated"),
        onError: () => toast.error("Failed to update defenses"),
      },
    );
  }

  // Save system prompt on blur
  function handleSystemPromptBlur() {
    if (!conversation) return;
    const newVal = (systemPrompt ?? "").trim() || null;
    if (newVal !== conversation.system_prompt) {
      updateMutation.mutate(
        { id: conversationId, data: { system_prompt: newVal } },
        {
          onError: () => toast.error("Failed to update system prompt"),
        },
      );
    }
  }

  // Inline title edit
  function startEditTitle() {
    if (!conversation) return;
    setTitleDraft(conversation.title);
    setEditingTitle(true);
  }

  function saveTitle() {
    setEditingTitle(false);
    if (!conversation || !titleDraft.trim()) return;
    if (titleDraft.trim() !== conversation.title) {
      updateMutation.mutate(
        { id: conversationId, data: { title: titleDraft.trim() } },
        {
          onError: () => toast.error("Failed to update title"),
        },
      );
    }
  }

  const activeDefenseNames = new Set(
    (conversation?.active_defenses ?? []).map((d) => d.name),
  );
  const asr =
    conversation && conversation.total_messages > 0
      ? (
          (conversation.total_jailbreaks / conversation.total_messages) *
          100
        ).toFixed(1)
      : "0.0";

  const isLoading = convLoading || msgsLoading;

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      {/* Header bar */}
      <div className="flex shrink-0 items-center gap-3 border-b px-4 py-2.5">
        {/* Title */}
        <div className="min-w-0 flex-1">
          {editingTitle ? (
            <Input
              value={titleDraft}
              onChange={(e) => setTitleDraft(e.target.value)}
              onBlur={saveTitle}
              onKeyDown={(e) => {
                if (e.key === "Enter") saveTitle();
                if (e.key === "Escape") setEditingTitle(false);
              }}
              className="h-7 text-sm font-medium"
              autoFocus
            />
          ) : (
            <button
              type="button"
              onClick={startEditTitle}
              className="group flex items-center gap-1.5 text-left"
            >
              <h3 className="truncate text-sm font-medium">
                {conversation?.title ?? "..."}
              </h3>
              <Pencil className="h-3 w-3 shrink-0 text-muted-foreground/50 transition-colors group-hover:text-muted-foreground" />
            </button>
          )}
        </div>

        {/* Model badges */}
        {conversation && (
          <div className="hidden shrink-0 items-center gap-1.5 sm:flex">
            <code className="rounded bg-muted px-1.5 py-0.5 text-[11px]">
              {conversation.target_model}
            </code>
            <code className="rounded bg-muted px-1.5 py-0.5 text-[11px] text-muted-foreground">
              <Brain className="mr-0.5 inline h-3 w-3" />
              {conversation.analyzer_model}
            </code>
          </div>
        )}

        {/* Stats */}
        {conversation && conversation.total_messages > 0 && (
          <TooltipProvider>
            <div className="hidden items-center gap-2 text-xs text-muted-foreground md:flex">
              <span>{conversation.total_messages} msgs</span>
              {conversation.total_jailbreaks > 0 && (
                <Tooltip>
                  <TooltipTrigger>
                    <span className="font-medium text-red-500">
                      {conversation.total_jailbreaks} jailbreaks
                    </span>
                  </TooltipTrigger>
                  <TooltipContent>ASR: {asr}%</TooltipContent>
                </Tooltip>
              )}
              {conversation.total_blocked > 0 && (
                <span className="text-amber-500">
                  {conversation.total_blocked} blocked
                </span>
              )}
            </div>
          </TooltipProvider>
        )}
      </div>

      {/* Scrollable chat area */}
      <div className="relative min-h-0 flex-1">
        <div
          ref={chatRef}
          onScroll={handleScroll}
          className="h-full overflow-y-auto p-4"
        >
          <div className="mx-auto max-w-3xl space-y-4 pb-4">
            {isLoading && (
              <div className="space-y-4">
                {["a", "b", "c"].map((k) => (
                  <Skeleton key={k} className="h-24 w-full rounded-lg" />
                ))}
              </div>
            )}

            {!isLoading && messages.length === 0 && (
              <div className="flex flex-col items-center justify-center py-16 text-center">
                <Target className="h-10 w-10 text-muted-foreground" />
                <h3 className="mt-4 text-sm font-medium">No messages yet</h3>
                <p className="mt-1 text-sm text-muted-foreground">
                  Type a prompt below to start testing the target model.
                </p>
              </div>
            )}

            {!isLoading &&
              messages.map((msg) => (
                <MessagePair
                  key={msg.id}
                  message={msg}
                />
              ))}

            {/* Pending indicator while sending */}
            {sendMutation.isPending && (
              <div className="space-y-3">
                <MessageBubble role="user">
                  <p className="whitespace-pre-wrap break-words text-sm [overflow-wrap:anywhere]">
                    {sendMutation.variables?.prompt ?? ""}
                  </p>
                </MessageBubble>
                <MessageBubble role="target">
                  <div className="flex items-center gap-2">
                    <BouncingDots />
                    {wsPhase && (
                      <span className="text-xs text-muted-foreground">
                        {wsPhase === "started" && "Processing..."}
                        {wsPhase === "target_calling" &&
                          "Calling target..."}
                        {wsPhase === "target_responded" &&
                          "Response received"}
                        {wsPhase === "analyzing" && "Analyzing..."}
                      </span>
                    )}
                  </div>
                </MessageBubble>
              </div>
            )}
          </div>
        </div>

        {/* Jump to latest */}
        {showJumpBtn && (
          <button
            type="button"
            onClick={jumpToLatest}
            className="absolute bottom-4 left-1/2 z-10 flex -translate-x-1/2 items-center gap-1.5 rounded-full border bg-background/95 px-3 py-1.5 text-xs font-medium shadow-lg backdrop-blur transition-colors hover:bg-accent"
          >
            <ArrowDown className="h-3 w-3" />
            Jump to latest
          </button>
        )}
      </div>

      {/* Input area with config toolbar */}
      <div className="shrink-0 border-t p-4">
        <div className="mx-auto max-w-3xl space-y-3">
          {/* Config toolbar */}
          <div className="flex flex-wrap items-center gap-2">
            {/* Target model — read-only in active chat */}
            <div className="flex h-8 items-center gap-1.5 rounded-md border bg-muted/50 px-2.5 text-xs">
              <Terminal className="h-3.5 w-3.5 text-muted-foreground" />
              <span>{conversation?.target_model ?? "..."}</span>
            </div>

            {/* Analyzer model — read-only */}
            <div className="flex h-8 items-center gap-1.5 rounded-md border bg-muted/50 px-2.5 text-xs">
              <Brain className="h-3.5 w-3.5 text-muted-foreground" />
              <span>{conversation?.analyzer_model ?? "..."}</span>
            </div>

            {/* Defenses popover */}
            <Popover>
              <PopoverTrigger asChild>
                <Button
                  variant="outline"
                  size="sm"
                  className="h-8 gap-1.5 text-xs"
                >
                  <Shield className="h-3.5 w-3.5" />
                  {activeDefenseNames.size > 0
                    ? `Defenses (${activeDefenseNames.size})`
                    : "Defenses"}
                </Button>
              </PopoverTrigger>
              <PopoverContent className="w-72" align="start">
                <p className="mb-3 text-xs font-medium">Active Defenses</p>
                <div className="space-y-2">
                  {defensesData?.defenses.map((d) => (
                    <div
                      key={d.name}
                      className="flex items-center justify-between rounded-md border px-2.5 py-1.5"
                    >
                      <span className="text-xs font-medium">{d.name}</span>
                      <Switch
                        checked={activeDefenseNames.has(d.name)}
                        onCheckedChange={(checked) =>
                          toggleDefense(d.name, checked)
                        }
                        disabled={updateMutation.isPending}
                      />
                    </div>
                  ))}
                </div>
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
                  {(systemPrompt ?? "").trim() ? (
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
                  value={systemPrompt ?? ""}
                  onChange={(e) => setSystemPrompt(e.target.value)}
                  onBlur={handleSystemPromptBlur}
                  placeholder="No system prompt"
                  rows={4}
                  className="text-xs"
                />
              </PopoverContent>
            </Popover>
          </div>

          {/* Textarea + send */}
          <div className="flex gap-2">
            <Textarea
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Type your attack prompt..."
              rows={2}
              disabled={sendMutation.isPending}
              className="min-h-[60px] resize-none"
            />
            <Button
              size="icon"
              onClick={handleSend}
              disabled={!prompt.trim() || sendMutation.isPending}
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

/* ── Helpers ──────────────────────────────────────────────── */

function BouncingDots() {
  return (
    <span className="inline-flex items-center gap-1">
      <span className="h-1.5 w-1.5 animate-bounce rounded-full bg-blue-500/60 [animation-delay:0ms]" />
      <span className="h-1.5 w-1.5 animate-bounce rounded-full bg-blue-500/60 [animation-delay:150ms]" />
      <span className="h-1.5 w-1.5 animate-bounce rounded-full bg-blue-500/60 [animation-delay:300ms]" />
    </span>
  );
}

function MessageBubble({
  role,
  blocked,
  children,
}: {
  role: "user" | "target";
  blocked?: boolean;
  children: React.ReactNode;
}) {
  const isUser = role === "user";
  const Icon = isUser ? User : Target;

  let borderColor: string;
  let labelColor: string;
  let label: string;

  if (isUser) {
    borderColor = "border-red-500/10 bg-red-500/5";
    labelColor = "text-red-500";
    label = "You";
  } else if (blocked) {
    borderColor = "border-amber-500/10 bg-amber-500/5";
    labelColor = "text-amber-500";
    label = "Target (Blocked)";
  } else {
    borderColor = "border-blue-500/10 bg-blue-500/5";
    labelColor = "text-blue-500";
    label = "Target";
  }

  const iconColor = isUser
    ? "bg-red-500/10 text-red-500"
    : "bg-blue-500/10 text-blue-500";

  return (
    <div className="flex gap-3">
      <div
        className={`flex h-7 w-7 shrink-0 items-center justify-center rounded-full ${iconColor}`}
      >
        <Icon className="h-3.5 w-3.5" />
      </div>
      <div
        className={`min-w-0 flex-1 rounded-lg border p-3 ${borderColor}`}
      >
        <p className={`mb-1 text-xs font-medium ${labelColor}`}>{label}</p>
        {children}
      </div>
    </div>
  );
}

function MessagePair({
  message: msg,
}: {
  message: PlaygroundMessage;
}) {
  const hasDefenses = (msg.defenses_applied ?? []).length > 0;
  const defended =
    hasDefenses &&
    (msg.target_blocked || msg.judge_verdict === "refused");

  return (
    <div className="space-y-3">
      {/* Message number + verdict */}
      <div className="flex flex-wrap items-center gap-2">
        <span className="font-mono text-xs text-muted-foreground">
          #{msg.message_number}
        </span>
        <Badge
          variant="outline"
          className={`text-[10px] ${VERDICT_BG[msg.judge_verdict] ?? ""}`}
        >
          {msg.judge_verdict}
          {msg.judge_confidence > 0 &&
            ` (${(msg.judge_confidence * 100).toFixed(0)}%)`}
        </Badge>
        {defended && (
          <Badge
            variant="outline"
            className="border-blue-500/20 bg-blue-500/10 text-[10px] text-blue-500"
          >
            <Shield className="mr-1 h-2.5 w-2.5" />
            {msg.blocked_by_defense
              ? `Blocked by ${msg.blocked_by_defense}`
              : "Defended"}
          </Badge>
        )}
        {msg.vulnerability_category && (
          <Badge variant="outline" className="text-[10px]">
            {msg.vulnerability_category}
          </Badge>
        )}
      </div>

      {/* User prompt */}
      <MessageBubble role="user">
        <p className="whitespace-pre-wrap break-words text-sm [overflow-wrap:anywhere]">
          {msg.user_prompt}
        </p>
      </MessageBubble>

      {/* Target response */}
      <MessageBubble role="target" blocked={msg.target_blocked}>
        <p className="whitespace-pre-wrap break-words text-sm [overflow-wrap:anywhere]">
          {msg.target_response}
        </p>
        {msg.target_blocked && msg.raw_target_response && (
          <RawResponseCollapsible
            rawResponse={msg.raw_target_response}
          />
        )}
        {msg.target_blocked && msg.block_reason && (
          <p className="mt-2 text-xs text-amber-500">
            Reason: {msg.block_reason}
          </p>
        )}
      </MessageBubble>

      {/* Scores */}
      {(msg.severity_score !== null ||
        msg.specificity_score !== null) && (
        <div className="ml-10 flex flex-wrap gap-3 text-xs text-muted-foreground">
          {msg.severity_score !== null && (
            <span>Severity: {msg.severity_score}/10</span>
          )}
          {msg.specificity_score !== null && (
            <span>Specificity: {msg.specificity_score}/10</span>
          )}
          {msg.attack_technique && (
            <span>Technique: {msg.attack_technique}</span>
          )}
        </div>
      )}

      {/* Latency */}
      <div className="ml-10 flex gap-3 text-xs text-muted-foreground">
        <span>
          Target: {(msg.target_latency_ms / 1000).toFixed(1)}s
        </span>
        <span>
          Analysis: {(msg.analyzer_latency_ms / 1000).toFixed(1)}s
        </span>
      </div>

      <Separator className="my-1" />
    </div>
  );
}

function RawResponseCollapsible({
  rawResponse,
}: {
  rawResponse: string;
}) {
  const [open, setOpen] = useState(false);

  return (
    <Collapsible open={open} onOpenChange={setOpen} className="mt-3">
      <CollapsibleTrigger asChild>
        <button
          type="button"
          className="flex items-center gap-1.5 rounded-md border border-amber-500/20 bg-amber-500/5 px-2.5 py-1.5 text-xs font-medium text-amber-600 transition-colors hover:bg-amber-500/10 dark:text-amber-400"
        >
          <ChevronDown
            className={`h-3 w-3 transition-transform ${open ? "rotate-180" : ""}`}
          />
          {open ? "Hide Original Response" : "View Original Response"}
        </button>
      </CollapsibleTrigger>
      <CollapsibleContent>
        <div className="mt-2 rounded-md border border-dashed border-amber-500/20 bg-amber-500/5 p-3">
          <p className="mb-1 text-[10px] font-medium uppercase tracking-wide text-amber-600/70 dark:text-amber-400/70">
            Original LLM Output (Pre-Defense)
          </p>
          <p className="whitespace-pre-wrap break-words text-sm text-muted-foreground [overflow-wrap:anywhere]">
            {rawResponse}
          </p>
        </div>
      </CollapsibleContent>
    </Collapsible>
  );
}
