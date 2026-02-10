"use client";

import { useState } from "react";
import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { formatDistanceToNow } from "date-fns";
import {
  MessageSquare,
  PanelRightOpen,
  Plus,
  Search,
  Trash2,
} from "lucide-react";
import { toast } from "sonner";
import { Header } from "@/components/layout/header";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  Sheet,
  SheetContent,
  SheetHeader,
  SheetTitle,
  SheetTrigger,
} from "@/components/ui/sheet";
import { Skeleton } from "@/components/ui/skeleton";
import { ROUTES, VERDICT_BG } from "@/lib/constants";
import {
  usePlaygroundConversations,
  useDeletePlaygroundConversation,
} from "@/hooks/use-playground";

export default function PlaygroundLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  const pathname = usePathname();
  const router = useRouter();
  const [search, setSearch] = useState("");

  const { data, isLoading } = usePlaygroundConversations();
  const deleteMutation = useDeletePlaygroundConversation();

  const conversations = data?.conversations ?? [];
  const filtered = conversations.filter(
    (c) =>
      c.title.toLowerCase().includes(search.toLowerCase()) ||
      c.target_model.toLowerCase().includes(search.toLowerCase()),
  );

  // Extract active conversation id from pathname
  const activeId = pathname.startsWith("/playground/")
    ? pathname.split("/")[2]
    : null;

  async function handleDelete(id: string) {
    try {
      await deleteMutation.mutateAsync(id);
      toast.success("Conversation deleted");
      if (activeId === id) {
        router.push(ROUTES.playground.list);
      }
    } catch {
      toast.error("Failed to delete conversation");
    }
  }

  const historyContent = (
    <div className="flex h-full flex-col">
      {/* New Chat button */}
      <div className="p-3">
        <Button
          variant="outline"
          className="w-full justify-start gap-2"
          asChild
        >
          <Link href={ROUTES.playground.list}>
            <Plus className="h-4 w-4" />
            New Chat
          </Link>
        </Button>
      </div>

      {/* Search */}
      <div className="px-3 pb-2">
        <div className="relative">
          <Search className="absolute left-2.5 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-muted-foreground" />
          <Input
            placeholder="Search..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            className="h-8 pl-8 text-xs"
          />
        </div>
      </div>

      {/* Conversation list */}
      <div className="flex-1 overflow-y-auto px-2 pb-2">
        {isLoading ? (
          <div className="space-y-2 px-1">
            {Array.from({ length: 4 }).map((_, i) => (
              <Skeleton key={i} className="h-16 w-full rounded-lg" />
            ))}
          </div>
        ) : filtered.length === 0 ? (
          <div className="flex flex-col items-center px-3 py-8 text-center">
            <MessageSquare className="h-6 w-6 text-muted-foreground" />
            <p className="mt-2 text-xs text-muted-foreground">
              {conversations.length === 0
                ? "No conversations yet"
                : "No matches"}
            </p>
          </div>
        ) : (
          <div className="space-y-1">
            {filtered.map((conv) => (
              <Link
                key={conv.id}
                href={ROUTES.playground.detail(conv.id)}
                className="group block"
              >
                <div
                  className={`relative rounded-lg border px-3 py-2.5 text-left transition-colors hover:bg-accent/50 ${
                    activeId === conv.id
                      ? "border-primary/40 bg-accent/30"
                      : "border-transparent"
                  }`}
                >
                  <p className="truncate text-sm font-medium">{conv.title}</p>
                  <div className="mt-1 flex items-center gap-1.5 text-[11px] text-muted-foreground">
                    <code className="rounded bg-muted px-1 py-0.5 text-[10px]">
                      {conv.target_model}
                    </code>
                    <span>&middot;</span>
                    <span>{conv.total_messages} msgs</span>
                  </div>
                  <div className="mt-1 flex items-center gap-1.5">
                    {conv.total_jailbreaks > 0 && (
                      <Badge
                        variant="outline"
                        className={`h-4 px-1 text-[9px] ${VERDICT_BG.jailbreak}`}
                      >
                        {conv.total_jailbreaks} jailbreak
                        {conv.total_jailbreaks !== 1 ? "s" : ""}
                      </Badge>
                    )}
                    <span className="text-[10px] text-muted-foreground">
                      {formatDistanceToNow(new Date(conv.updated_at), {
                        addSuffix: true,
                      })}
                    </span>
                  </div>

                  {/* Delete button */}
                  <button
                    type="button"
                    onClick={(e) => {
                      e.preventDefault();
                      e.stopPropagation();
                      handleDelete(conv.id);
                    }}
                    className="absolute right-2 top-2 rounded-md p-1 opacity-0 transition-opacity hover:bg-destructive/10 group-hover:opacity-100"
                  >
                    <Trash2 className="h-3 w-3 text-muted-foreground hover:text-destructive" />
                  </button>
                </div>
              </Link>
            ))}
          </div>
        )}
      </div>
    </div>
  );

  return (
    <div className="flex h-dvh flex-col overflow-hidden">
      <Header title="Playground" />

      <div className="flex min-h-0 flex-1">
        {/* Chat area (children) */}
        <div className="flex min-h-0 min-w-0 flex-1 flex-col">
          {children}
        </div>

        {/* Right history panel — desktop */}
        <div className="hidden w-72 shrink-0 border-l lg:block">
          {historyContent}
        </div>

        {/* Right history panel — mobile (Sheet) */}
        <div className="fixed right-4 top-3 z-50 lg:hidden">
          <Sheet>
            <SheetTrigger asChild>
              <Button variant="ghost" size="icon" className="h-8 w-8">
                <PanelRightOpen className="h-4 w-4" />
              </Button>
            </SheetTrigger>
            <SheetContent side="right" className="w-80 p-0">
              <SheetHeader className="border-b px-4 py-3">
                <SheetTitle className="text-sm">Chat History</SheetTitle>
              </SheetHeader>
              {historyContent}
            </SheetContent>
          </Sheet>
        </div>
      </div>
    </div>
  );
}
