"use client";

import { useCallback, useRef, useState } from "react";
import { formatDistanceToNow } from "date-fns";
import {
  AlertTriangle,
  Check,
  CircleDot,
  Database,
  Loader2,
  MoreHorizontal,
  Pencil,
  Plus,
  Power,
  PowerOff,
  Sparkles,
  Square,
  Trash2,
  Upload,
  X,
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
import { Checkbox } from "@/components/ui/checkbox";
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
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Skeleton } from "@/components/ui/skeleton";
import { Textarea } from "@/components/ui/textarea";
import {
  useDatasetPrompts,
  useDatasetStats,
  useAddDatasetPrompt,
  useUploadDataset,
  useUpdateDatasetPrompt,
  useDeleteDatasetPrompt,
  useDatasetSuggestions,
  useConfirmSuggestion,
  useDismissSuggestion,
  useGenerateHarmless,
  useLoadModel,
  useUnloadModel,
  useModelStatus,
} from "@/hooks/use-dataset";
import { useModels } from "@/hooks/use-targets";
import {
  PROMPT_CATEGORY_COLORS,
  PROMPT_SOURCE_LABELS,
} from "@/lib/constants";
import type { AbliterationPrompt, PromptCategory } from "@/lib/types";

// ── Pair computation ───────────────────────────────────────────

interface PromptPair {
  harmful: AbliterationPrompt;
  harmless: AbliterationPrompt;
}

function computePairs(prompts: AbliterationPrompt[]) {
  // Build reverse index: harmful_id → harmless prompt
  const harmlessByPairId = new Map<string, AbliterationPrompt>();
  for (const p of prompts) {
    if (p.category === "harmless" && p.pair_id) {
      harmlessByPairId.set(p.pair_id, p);
    }
  }

  const pairedIds = new Set<string>();
  const pairs: PromptPair[] = [];
  for (const p of prompts) {
    if (p.category === "harmful") {
      const harmless = harmlessByPairId.get(p.id);
      if (harmless) {
        pairs.push({ harmful: p, harmless });
        pairedIds.add(p.id);
        pairedIds.add(harmless.id);
      }
    }
  }

  const singles = prompts.filter((p) => !pairedIds.has(p.id));
  const unpairedHarmful = singles.filter(
    (p) => p.category === "harmful",
  );
  return { pairs, singles, unpairedHarmful };
}

// ── Main page ──────────────────────────────────────────────────

export default function DatasetPage() {
  // Filters
  const [categoryFilter, setCategoryFilter] = useState<string>("all");
  const [sourceFilter, setSourceFilter] = useState<string>("all");

  // Model lifecycle
  const [selectedModel, setSelectedModel] = useState<string>("");
  const [isStopping, setIsStopping] = useState(false);

  // Batch generation
  const [isGeneratingAll, setIsGeneratingAll] = useState(false);
  const [genProgress, setGenProgress] = useState<{
    current: number;
    total: number;
  } | null>(null);
  const shouldStopRef = useRef(false);

  // Per-prompt generation tracking
  const [generatingId, setGeneratingId] = useState<string | null>(null);

  // Add prompt dialog
  const [addOpen, setAddOpen] = useState(false);
  const [addText, setAddText] = useState("");
  const [addCategory, setAddCategory] = useState<PromptCategory>("harmful");
  const [addAutoGenerate, setAddAutoGenerate] = useState(true);

  // Upload dialog
  const [uploadOpen, setUploadOpen] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Edit dialog
  const [editOpen, setEditOpen] = useState(false);
  const [editPrompt, setEditPrompt] = useState<AbliterationPrompt | null>(
    null,
  );
  const [editText, setEditText] = useState("");
  const [editCategory, setEditCategory] =
    useState<PromptCategory>("harmful");

  // Suggestion editing
  const [editingSuggestionId, setEditingSuggestionId] = useState<
    string | null
  >(null);
  const [editingSuggestionText, setEditingSuggestionText] = useState("");

  // Queries
  const catParam = categoryFilter === "all" ? undefined : categoryFilter;
  const srcParam = sourceFilter === "all" ? undefined : sourceFilter;
  const { data: promptsData, isLoading } = useDatasetPrompts(
    catParam,
    srcParam,
  );
  const { data: stats } = useDatasetStats();
  const { data: suggestionsData } = useDatasetSuggestions();
  const { data: modelsData } = useModels();
  const { data: modelStatusData } = useModelStatus();

  // Mutations
  const addMutation = useAddDatasetPrompt();
  const uploadMutation = useUploadDataset();
  const updateMutation = useUpdateDatasetPrompt();
  const deleteMutation = useDeleteDatasetPrompt();
  const confirmMutation = useConfirmSuggestion();
  const dismissMutation = useDismissSuggestion();
  const generateMutation = useGenerateHarmless();
  const loadModelMutation = useLoadModel();
  const unloadModelMutation = useUnloadModel();

  const prompts = promptsData?.prompts ?? [];
  const suggestions = suggestionsData?.suggestions ?? [];
  const availableModels = modelsData?.models ?? [];

  // Derive model loaded state
  const runningModels = modelStatusData?.models ?? [];
  const isModelLoaded =
    !!selectedModel &&
    runningModels.some((m) => m.name === selectedModel);

  // Compute pairs
  const { pairs, singles, unpairedHarmful } = computePairs(prompts);

  // Any generation in progress (single or batch)
  const isAnyGenerating = isGeneratingAll || generatingId !== null;

  // ── Model lifecycle handlers ─────────────────────────────────

  const handleInitializeModel = () => {
    if (!selectedModel) return;
    loadModelMutation.mutate(selectedModel, {
      onSuccess: () => toast.success(`Model ${selectedModel} loaded`),
      onError: () => toast.error("Failed to load model"),
    });
  };

  const handleStopModel = () => {
    if (!selectedModel || isStopping) return;
    setIsStopping(true);
    unloadModelMutation.mutate(selectedModel, {
      onSuccess: () => {
        toast.success(`Model ${selectedModel} unloaded`);
        setIsStopping(false);
      },
      onError: () => {
        toast.error("Failed to unload model");
        setIsStopping(false);
      },
    });
  };

  // ── Batch generation ─────────────────────────────────────────

  const handleGenerateAll = useCallback(async () => {
    if (!selectedModel || unpairedHarmful.length === 0) return;
    setIsGeneratingAll(true);
    shouldStopRef.current = false;
    const total = unpairedHarmful.length;
    setGenProgress({ current: 0, total });

    for (let i = 0; i < unpairedHarmful.length; i++) {
      if (shouldStopRef.current) break;
      const p = unpairedHarmful[i];
      setGeneratingId(p.id);
      setGenProgress({ current: i, total });
      try {
        await generateMutation.mutateAsync({
          data: { harmful_prompt: p.text, pair_id: p.id },
          model: selectedModel,
        });
      } catch {
        toast.error(
          `Failed to generate counterpart for prompt ${i + 1}`,
        );
      }
    }

    setIsGeneratingAll(false);
    setGeneratingId(null);
    setGenProgress(null);
    if (shouldStopRef.current) {
      toast.info("Generation stopped");
    } else {
      toast.success("All counterparts generated");
    }
  }, [selectedModel, unpairedHarmful, generateMutation]);

  const handleStopGeneration = () => {
    shouldStopRef.current = true;
  };

  // ── Single generation ────────────────────────────────────────

  const handleGenerateCounterpart = (prompt: AbliterationPrompt) => {
    if (!selectedModel) {
      toast.error("Please select and initialize a model first");
      return;
    }
    setGeneratingId(prompt.id);
    generateMutation.mutate(
      {
        data: { harmful_prompt: prompt.text, pair_id: prompt.id },
        model: selectedModel,
      },
      {
        onSuccess: () => {
          toast.success("Harmless counterpart generated");
          setGeneratingId(null);
        },
        onError: () => {
          toast.error("Failed to generate counterpart");
          setGeneratingId(null);
        },
      },
    );
  };

  // ── CRUD handlers ────────────────────────────────────────────

  const handleAdd = () => {
    if (!addText.trim()) {
      toast.error("Prompt text is required");
      return;
    }
    const wantsAutoGen =
      addCategory === "harmful" && addAutoGenerate;
    if (wantsAutoGen && !isModelLoaded) {
      toast.error("Please initialize a model first");
      return;
    }
    addMutation.mutate(
      {
        data: {
          text: addText.trim(),
          category: addCategory,
          auto_generate_counterpart: wantsAutoGen,
        },
        generationModel: wantsAutoGen ? selectedModel : undefined,
      },
      {
        onSuccess: () => {
          toast.success(
            wantsAutoGen
              ? "Prompt added with harmless counterpart"
              : "Prompt added",
          );
          setAddOpen(false);
          setAddText("");
          setAddCategory("harmful");
          setAddAutoGenerate(true);
        },
        onError: () => toast.error("Failed to add prompt"),
      },
    );
  };

  const handleUpload = (file: File) => {
    uploadMutation.mutate(file, {
      onSuccess: (data) => {
        toast.success(`Uploaded ${data.total} prompts`);
        setUploadOpen(false);
      },
      onError: () => toast.error("Failed to upload file"),
    });
  };

  const handleEdit = () => {
    if (!editPrompt || !editText.trim()) return;
    updateMutation.mutate(
      {
        id: editPrompt.id,
        data: { text: editText.trim(), category: editCategory },
      },
      {
        onSuccess: () => {
          toast.success("Prompt updated");
          setEditOpen(false);
          setEditPrompt(null);
        },
        onError: () => toast.error("Failed to update prompt"),
      },
    );
  };

  const handleDelete = (id: string) => {
    deleteMutation.mutate(id, {
      onSuccess: () => toast.success("Prompt deleted"),
      onError: () => toast.error("Failed to delete prompt"),
    });
  };

  const handleConfirm = (id: string) => {
    if (!isModelLoaded) {
      toast.error("Please initialize a model first");
      return;
    }
    confirmMutation.mutate(
      {
        id,
        autoGenerateCounterpart: true,
        model: selectedModel,
      },
      {
        onSuccess: () =>
          toast.success("Suggestion confirmed with counterpart"),
        onError: () => toast.error("Failed to confirm suggestion"),
      },
    );
  };

  const handleConfirmWithoutGen = (id: string) => {
    confirmMutation.mutate(
      { id, autoGenerateCounterpart: false },
      {
        onSuccess: () => toast.success("Suggestion confirmed"),
        onError: () => toast.error("Failed to confirm suggestion"),
      },
    );
  };

  const handleDismiss = (id: string) => {
    dismissMutation.mutate(id, {
      onSuccess: () => toast.success("Suggestion dismissed"),
      onError: () => toast.error("Failed to dismiss suggestion"),
    });
  };

  const openEdit = (prompt: AbliterationPrompt) => {
    setEditPrompt(prompt);
    setEditText(prompt.text);
    setEditCategory(prompt.category);
    setEditOpen(true);
  };

  const handleEditSuggestion = (s: AbliterationPrompt) => {
    setEditingSuggestionId(s.id);
    setEditingSuggestionText(s.text);
  };

  const handleSaveAndConfirm = (id: string) => {
    const trimmed = editingSuggestionText.trim();
    if (!trimmed) {
      toast.error("Prompt text is required");
      return;
    }
    updateMutation.mutate(
      { id, data: { text: trimmed } },
      {
        onSuccess: () => {
          const wantGen = isModelLoaded;
          confirmMutation.mutate(
            {
              id,
              autoGenerateCounterpart: wantGen,
              model: wantGen ? selectedModel : undefined,
            },
            {
              onSuccess: () => {
                toast.success(
                  wantGen
                    ? "Suggestion updated & confirmed with counterpart"
                    : "Suggestion updated & confirmed",
                );
                setEditingSuggestionId(null);
              },
              onError: () =>
                toast.error("Failed to confirm suggestion"),
            },
          );
        },
        onError: () =>
          toast.error("Failed to update suggestion text"),
      },
    );
  };

  const handleCancelEditSuggestion = () => {
    setEditingSuggestionId(null);
    setEditingSuggestionText("");
  };

  return (
    <>
      <Header title="Abliteration Dataset" />

      <div className="flex flex-1 flex-col gap-6 p-6">
        {/* Page header */}
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-2xl font-bold tracking-tight">
              Abliteration Dataset
            </h2>
            <p className="text-muted-foreground">
              Manage harmful/harmless prompt pairs used for refusal
              direction computation during abliteration.
            </p>
          </div>

          <div className="flex items-center gap-2">
            <Dialog open={uploadOpen} onOpenChange={setUploadOpen}>
              <DialogTrigger asChild>
                <Button variant="outline">
                  <Upload className="mr-2 h-4 w-4" />
                  Upload
                </Button>
              </DialogTrigger>
              <DialogContent>
                <DialogHeader>
                  <DialogTitle>Upload JSONL File</DialogTitle>
                  <DialogDescription>
                    Upload a JSONL file with one JSON object per line.
                    Format:{" "}
                    <code className="rounded bg-muted px-1 py-0.5 text-xs">
                      {`{"harmful": "...", "harmless": "..."}`}
                    </code>
                  </DialogDescription>
                </DialogHeader>
                <div className="grid gap-4 py-4">
                  <input
                    ref={fileInputRef}
                    type="file"
                    accept=".jsonl,.json"
                    className="block w-full text-sm text-muted-foreground file:mr-4 file:rounded file:border-0 file:bg-muted file:px-4 file:py-2 file:text-sm file:font-medium hover:file:bg-muted/80"
                  />
                </div>
                <DialogFooter>
                  <Button
                    variant="outline"
                    onClick={() => setUploadOpen(false)}
                  >
                    Cancel
                  </Button>
                  <Button
                    onClick={() => {
                      const file =
                        fileInputRef.current?.files?.[0];
                      if (!file) {
                        toast.error("Select a file first");
                        return;
                      }
                      handleUpload(file);
                    }}
                    disabled={uploadMutation.isPending}
                  >
                    {uploadMutation.isPending
                      ? "Uploading..."
                      : "Upload"}
                  </Button>
                </DialogFooter>
              </DialogContent>
            </Dialog>

            <Dialog open={addOpen} onOpenChange={setAddOpen}>
              <DialogTrigger asChild>
                <Button>
                  <Plus className="mr-2 h-4 w-4" />
                  Add Prompt
                </Button>
              </DialogTrigger>
              <DialogContent>
                <DialogHeader>
                  <DialogTitle>Add Prompt</DialogTitle>
                  <DialogDescription>
                    Add a prompt to the abliteration dataset.
                  </DialogDescription>
                </DialogHeader>
                <div className="grid gap-4 py-4">
                  <div className="grid gap-2">
                    <Label>Category</Label>
                    <Select
                      value={addCategory}
                      onValueChange={(v) =>
                        setAddCategory(v as PromptCategory)
                      }
                    >
                      <SelectTrigger>
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="harmful">
                          Harmful
                        </SelectItem>
                        <SelectItem value="harmless">
                          Harmless
                        </SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                  <div className="grid gap-2">
                    <Label>Prompt Text</Label>
                    <Textarea
                      placeholder="Enter prompt text..."
                      value={addText}
                      onChange={(e) => setAddText(e.target.value)}
                      rows={4}
                    />
                  </div>
                  {addCategory === "harmful" && (
                    <div className="flex items-center gap-2">
                      <Checkbox
                        id="auto-generate"
                        checked={addAutoGenerate}
                        onCheckedChange={(v) =>
                          setAddAutoGenerate(v === true)
                        }
                      />
                      <Label
                        htmlFor="auto-generate"
                        className="text-sm font-normal"
                      >
                        Auto-generate harmless counterpart via LLM
                        {addAutoGenerate && !isModelLoaded && (
                          <span className="ml-1 text-xs text-amber-500">
                            (requires initialized model)
                          </span>
                        )}
                      </Label>
                    </div>
                  )}
                </div>
                <DialogFooter>
                  <Button
                    variant="outline"
                    onClick={() => setAddOpen(false)}
                  >
                    Cancel
                  </Button>
                  <Button
                    onClick={handleAdd}
                    disabled={addMutation.isPending}
                  >
                    {addMutation.isPending
                      ? "Adding..."
                      : "Add Prompt"}
                  </Button>
                </DialogFooter>
              </DialogContent>
            </Dialog>
          </div>
        </div>

        {/* Model control bar */}
        <Card>
          <CardContent className="pt-6">
            <div className="flex items-center gap-3">
              <Label className="shrink-0 text-sm font-medium">
                Generation Model
              </Label>
              <Select
                value={selectedModel}
                onValueChange={setSelectedModel}
                disabled={
                  loadModelMutation.isPending || isGeneratingAll
                }
              >
                <SelectTrigger className="w-[240px]">
                  <SelectValue placeholder="Select a model..." />
                </SelectTrigger>
                <SelectContent>
                  {availableModels.map((m) => (
                    <SelectItem key={m} value={m}>
                      {m}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>

              {isStopping && (
                <div className="ml-auto flex items-center gap-2 text-sm text-muted-foreground">
                  <Loader2 className="h-4 w-4 animate-spin" />
                  Releasing model...
                </div>
              )}
              {!isStopping && !isModelLoaded && (
                <Button
                  onClick={handleInitializeModel}
                  disabled={
                    !selectedModel ||
                    loadModelMutation.isPending
                  }
                  size="sm"
                >
                  {loadModelMutation.isPending ? (
                    <>
                      <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                      Initializing...
                    </>
                  ) : (
                    <>
                      <Power className="mr-2 h-4 w-4" />
                      Initialize
                    </>
                  )}
                </Button>
              )}
              {!isStopping && isModelLoaded && (
                <>
                  <div className="flex items-center gap-1.5 text-sm text-emerald-500">
                    <CircleDot className="h-3.5 w-3.5" />
                    Loaded
                  </div>

                  {/* Batch generation */}
                  {isGeneratingAll ? (
                    <div className="ml-auto flex items-center gap-2">
                      <div className="flex items-center gap-2 text-sm text-muted-foreground">
                        <Loader2 className="h-4 w-4 animate-spin" />
                        Generating{" "}
                        {genProgress
                          ? `${genProgress.current + 1}/${genProgress.total}`
                          : "..."}
                      </div>
                      <Button
                        size="sm"
                        variant="destructive"
                        onClick={handleStopGeneration}
                      >
                        <Square className="mr-2 h-3 w-3" />
                        Stop
                      </Button>
                    </div>
                  ) : (
                    unpairedHarmful.length > 0 && (
                      <Button
                        size="sm"
                        variant="outline"
                        onClick={handleGenerateAll}
                        disabled={isAnyGenerating}
                        className="ml-auto"
                      >
                        <Sparkles className="mr-2 h-4 w-4" />
                        Generate All (
                        {unpairedHarmful.length})
                      </Button>
                    )
                  )}

                  <Button
                    size="sm"
                    variant="ghost"
                    onClick={handleStopModel}
                    disabled={isGeneratingAll}
                    className="ml-auto"
                  >
                    <PowerOff className="mr-2 h-4 w-4" />
                    Stop Model
                  </Button>
                </>
              )}
            </div>
          </CardContent>
        </Card>

        {/* Stats card */}
        {stats && (
          <Card>
            <CardContent className="pt-6">
              <div className="flex items-center gap-8">
                <div className="flex items-center gap-2">
                  <div className="h-3 w-3 rounded-full bg-red-500" />
                  <span className="text-sm font-medium">
                    {stats.harmful_count} Harmful
                  </span>
                </div>
                <div className="flex items-center gap-2">
                  <div className="h-3 w-3 rounded-full bg-emerald-500" />
                  <span className="text-sm font-medium">
                    {stats.harmless_count} Harmless
                  </span>
                </div>
                <div className="text-sm text-muted-foreground">
                  {stats.total} total prompts
                </div>
              </div>
              {stats.warning && (
                <div className="mt-3 flex items-center gap-2 rounded-md border border-amber-500/20 bg-amber-500/10 px-3 py-2 text-sm text-amber-500">
                  <AlertTriangle className="h-4 w-4 shrink-0" />
                  {stats.warning}
                </div>
              )}
            </CardContent>
          </Card>
        )}

        {/* Pending suggestions */}
        {suggestions.length > 0 && (
          <Card>
            <CardHeader>
              <div className="flex items-center gap-2">
                <Database className="h-4 w-4 text-muted-foreground" />
                <span className="text-sm font-medium">
                  Pending Suggestions ({suggestions.length})
                </span>
              </div>
              <p className="text-xs text-muted-foreground">
                Auto-detected from sessions where the baseline was
                refused.
              </p>
            </CardHeader>
            <CardContent>
              <div className="space-y-2">
                {suggestions.map((s) => (
                  <div
                    key={s.id}
                    className="rounded-md border px-3 py-2"
                  >
                    {editingSuggestionId === s.id ? (
                      <div className="space-y-2">
                        <Textarea
                          value={editingSuggestionText}
                          onChange={(e) =>
                            setEditingSuggestionText(
                              e.target.value,
                            )
                          }
                          rows={3}
                          className="text-sm"
                        />
                        <div className="flex items-center justify-end gap-2">
                          <Button
                            size="sm"
                            variant="ghost"
                            onClick={
                              handleCancelEditSuggestion
                            }
                          >
                            Cancel
                          </Button>
                          <Button
                            size="sm"
                            onClick={() =>
                              handleSaveAndConfirm(s.id)
                            }
                            disabled={
                              updateMutation.isPending ||
                              confirmMutation.isPending
                            }
                          >
                            <Check className="mr-1 h-3 w-3" />
                            {updateMutation.isPending ||
                            confirmMutation.isPending
                              ? "Saving..."
                              : "Save & Confirm"}
                          </Button>
                        </div>
                      </div>
                    ) : (
                      <div className="flex items-center justify-between">
                        <p className="flex-1 truncate text-sm">
                          {s.text}
                        </p>
                        <div className="ml-4 flex shrink-0 items-center gap-2">
                          <Button
                            size="sm"
                            variant="ghost"
                            onClick={() =>
                              handleEditSuggestion(s)
                            }
                          >
                            <Pencil className="mr-1 h-3 w-3" />
                            Edit
                          </Button>
                          {isModelLoaded ? (
                            <Button
                              size="sm"
                              variant="outline"
                              onClick={() =>
                                handleConfirm(s.id)
                              }
                              disabled={
                                confirmMutation.isPending
                              }
                            >
                              <Check className="mr-1 h-3 w-3" />
                              Confirm
                            </Button>
                          ) : (
                            <Button
                              size="sm"
                              variant="outline"
                              onClick={() =>
                                handleConfirmWithoutGen(
                                  s.id,
                                )
                              }
                              disabled={
                                confirmMutation.isPending
                              }
                            >
                              <Check className="mr-1 h-3 w-3" />
                              Confirm
                            </Button>
                          )}
                          <Button
                            size="sm"
                            variant="ghost"
                            onClick={() =>
                              handleDismiss(s.id)
                            }
                            disabled={
                              dismissMutation.isPending
                            }
                          >
                            <X className="mr-1 h-3 w-3" />
                            Dismiss
                          </Button>
                        </div>
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        )}

        {/* Prompt cards */}
        <Card>
          <CardHeader>
            <div className="flex items-center gap-4">
              <Select
                value={categoryFilter}
                onValueChange={setCategoryFilter}
              >
                <SelectTrigger className="w-[140px]">
                  <SelectValue placeholder="Category" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">
                    All Categories
                  </SelectItem>
                  <SelectItem value="harmful">Harmful</SelectItem>
                  <SelectItem value="harmless">
                    Harmless
                  </SelectItem>
                </SelectContent>
              </Select>
              <Select
                value={sourceFilter}
                onValueChange={setSourceFilter}
              >
                <SelectTrigger className="w-[140px]">
                  <SelectValue placeholder="Source" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">All Sources</SelectItem>
                  <SelectItem value="manual">Manual</SelectItem>
                  <SelectItem value="upload">Upload</SelectItem>
                  <SelectItem value="session">Session</SelectItem>
                </SelectContent>
              </Select>
            </div>
          </CardHeader>
          <CardContent>
            {isLoading && (
              <div className="space-y-3">
                {["s1", "s2", "s3", "s4", "s5"].map((id) => (
                  <Skeleton key={id} className="h-24 w-full" />
                ))}
              </div>
            )}
            {!isLoading && prompts.length === 0 && (
              <div className="flex flex-col items-center justify-center py-16 text-center">
                <Database className="h-10 w-10 text-muted-foreground" />
                <h3 className="mt-4 text-sm font-medium">
                  No prompts yet
                </h3>
                <p className="mt-1 text-sm text-muted-foreground">
                  Add prompts manually or upload a JSONL file to
                  get started.
                </p>
              </div>
            )}
            {!isLoading && prompts.length > 0 && (
              <div className="space-y-3">
                {/* Paired cards */}
                {pairs.map((pair) => (
                  <PairedCard
                    key={pair.harmful.id}
                    pair={pair}
                    onEdit={openEdit}
                    onDelete={handleDelete}
                  />
                ))}

                {/* Unpaired cards */}
                {singles.map((p) => (
                  <UnpairedCard
                    key={p.id}
                    prompt={p}
                    isModelLoaded={isModelLoaded}
                    isGenerating={generatingId === p.id}
                    isAnyGenerating={isAnyGenerating}
                    onEdit={openEdit}
                    onDelete={handleDelete}
                    onGenerate={handleGenerateCounterpart}
                  />
                ))}
              </div>
            )}
          </CardContent>
        </Card>
      </div>

      {/* Edit dialog */}
      <Dialog open={editOpen} onOpenChange={setEditOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Edit Prompt</DialogTitle>
            <DialogDescription>
              Update the prompt text or category.
            </DialogDescription>
          </DialogHeader>
          <div className="grid gap-4 py-4">
            <div className="grid gap-2">
              <Label>Category</Label>
              <Select
                value={editCategory}
                onValueChange={(v) =>
                  setEditCategory(v as PromptCategory)
                }
              >
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="harmful">Harmful</SelectItem>
                  <SelectItem value="harmless">
                    Harmless
                  </SelectItem>
                </SelectContent>
              </Select>
            </div>
            <div className="grid gap-2">
              <Label>Prompt Text</Label>
              <Textarea
                value={editText}
                onChange={(e) => setEditText(e.target.value)}
                rows={4}
              />
            </div>
          </div>
          <DialogFooter>
            <Button
              variant="outline"
              onClick={() => setEditOpen(false)}
            >
              Cancel
            </Button>
            <Button
              onClick={handleEdit}
              disabled={updateMutation.isPending}
            >
              {updateMutation.isPending
                ? "Saving..."
                : "Save Changes"}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </>
  );
}

// ── Paired card component ──────────────────────────────────────

function PairedCard({
  pair,
  onEdit,
  onDelete,
}: Readonly<{
  pair: PromptPair;
  onEdit: (p: AbliterationPrompt) => void;
  onDelete: (id: string) => void;
}>) {
  return (
    <div className="overflow-hidden rounded-lg border">
      {/* Harmful half */}
      <PromptHalf
        prompt={pair.harmful}
        onEdit={onEdit}
        onDelete={onDelete}
      />
      {/* Divider */}
      <div className="border-t" />
      {/* Harmless half */}
      <PromptHalf
        prompt={pair.harmless}
        onEdit={onEdit}
        onDelete={onDelete}
      />
    </div>
  );
}

// ── Unpaired card component ────────────────────────────────────

function UnpairedCard({
  prompt,
  isModelLoaded,
  isGenerating,
  isAnyGenerating,
  onEdit,
  onDelete,
  onGenerate,
}: Readonly<{
  prompt: AbliterationPrompt;
  isModelLoaded: boolean;
  isGenerating: boolean;
  isAnyGenerating: boolean;
  onEdit: (p: AbliterationPrompt) => void;
  onDelete: (id: string) => void;
  onGenerate: (p: AbliterationPrompt) => void;
}>) {
  const showGenerate =
    prompt.category === "harmful" && isModelLoaded;

  return (
    <div className="overflow-hidden rounded-lg border">
      <PromptHalf
        prompt={prompt}
        onEdit={onEdit}
        onDelete={onDelete}
      />
      {showGenerate && (
        <div className="flex items-center justify-end border-t bg-muted/30 px-4 py-2">
          <Button
            size="sm"
            variant="outline"
            onClick={() => onGenerate(prompt)}
            disabled={isGenerating || isAnyGenerating}
          >
            {isGenerating ? (
              <>
                <Loader2 className="mr-2 h-3 w-3 animate-spin" />
                Generating...
              </>
            ) : (
              <>
                <Sparkles className="mr-2 h-3 w-3" />
                Generate Counterpart
              </>
            )}
          </Button>
        </div>
      )}
    </div>
  );
}

// ── Shared prompt half component ───────────────────────────────

function PromptHalf({
  prompt,
  onEdit,
  onDelete,
}: Readonly<{
  prompt: AbliterationPrompt;
  onEdit: (p: AbliterationPrompt) => void;
  onDelete: (id: string) => void;
}>) {
  const isHarmful = prompt.category === "harmful";

  return (
    <div className="group flex items-start justify-between gap-3 px-4 py-3">
      <div className="min-w-0 flex-1 space-y-1">
        <div className="flex items-center gap-2">
          <Badge
            variant="outline"
            className={
              PROMPT_CATEGORY_COLORS[prompt.category] ?? ""
            }
          >
            {prompt.category}
          </Badge>
          <span className="text-xs text-muted-foreground">
            {PROMPT_SOURCE_LABELS[prompt.source] ?? prompt.source}
          </span>
          <span className="text-xs text-muted-foreground">
            {formatDistanceToNow(new Date(prompt.created_at), {
              addSuffix: true,
            })}
          </span>
        </div>
        <p
          className={`text-sm leading-relaxed ${
            isHarmful
              ? "text-foreground"
              : "text-muted-foreground"
          }`}
        >
          {prompt.text}
        </p>
      </div>

      <DropdownMenu>
        <DropdownMenuTrigger asChild>
          <Button
            variant="ghost"
            size="icon"
            className="h-8 w-8 shrink-0 opacity-0 group-hover:opacity-100"
          >
            <MoreHorizontal className="h-4 w-4" />
          </Button>
        </DropdownMenuTrigger>
        <DropdownMenuContent align="end">
          <DropdownMenuItem onClick={() => onEdit(prompt)}>
            <Pencil className="mr-2 h-4 w-4" />
            Edit
          </DropdownMenuItem>
          <DropdownMenuItem
            className="text-destructive"
            onClick={() => onDelete(prompt.id)}
          >
            <Trash2 className="mr-2 h-4 w-4" />
            Delete
          </DropdownMenuItem>
        </DropdownMenuContent>
      </DropdownMenu>
    </div>
  );
}
