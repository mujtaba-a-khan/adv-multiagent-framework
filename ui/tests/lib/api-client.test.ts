import { describe, it, expect, vi, beforeEach } from "vitest";

// Mock constants before importing api-client
vi.mock("@/lib/constants", () => ({
  API_V1: "http://test-api/api/v1",
}));

const mockFetch = vi.fn();
globalThis.fetch = mockFetch;

// Dynamic import so mocks are in place
const api = await import("@/lib/api-client");

function jsonResponse(data: unknown, status = 200) {
  return {
    ok: status >= 200 && status < 300,
    status,
    json: () => Promise.resolve(data),
    text: () => Promise.resolve(JSON.stringify(data)),
  };
}

function emptyResponse(status = 204) {
  return { ok: true, status, json: () => Promise.resolve(undefined), text: () => Promise.resolve("") };
}

function errorResponse(status: number, body = "Error") {
  return { ok: false, status, text: () => Promise.resolve(body) };
}

describe("ApiError", () => {
  it("has correct name and status", () => {
    const err = new api.ApiError(404, "Not found");
    expect(err.name).toBe("ApiError");
    expect(err.status).toBe(404);
    expect(err.message).toBe("Not found");
  });
});

describe("api-client request layer", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("throws ApiError on non-2xx response", async () => {
    mockFetch.mockResolvedValue(errorResponse(500, "Internal Server Error"));
    await expect(api.listExperiments()).rejects.toThrow(api.ApiError);
  });

  it("returns undefined for 204 responses", async () => {
    mockFetch.mockResolvedValue(emptyResponse(204));
    const result = await api.deleteExperiment("id-1");
    expect(result).toBeUndefined();
  });

  it("sets Content-Type header", async () => {
    mockFetch.mockResolvedValue(jsonResponse({ items: [], total: 0 }));
    await api.listExperiments();
    expect(mockFetch).toHaveBeenCalledWith(
      expect.any(String),
      expect.objectContaining({
        headers: expect.objectContaining({ "Content-Type": "application/json" }),
      }),
    );
  });
});

describe("Experiments", () => {
  beforeEach(() => vi.clearAllMocks());

  it("createExperiment sends POST", async () => {
    const exp = { id: "e1", name: "Test" };
    mockFetch.mockResolvedValue(jsonResponse(exp));
    const result = await api.createExperiment({ name: "Test", objective: "obj", target_model: "m" });
    expect(result).toEqual(exp);
    expect(mockFetch).toHaveBeenCalledWith(
      "http://test-api/api/v1/experiments",
      expect.objectContaining({ method: "POST" }),
    );
  });

  it("listExperiments sends GET with pagination", async () => {
    mockFetch.mockResolvedValue(jsonResponse({ items: [], total: 0 }));
    await api.listExperiments(10, 25);
    expect(mockFetch).toHaveBeenCalledWith(
      "http://test-api/api/v1/experiments?offset=10&limit=25",
      expect.any(Object),
    );
  });

  it("getExperiment fetches by id", async () => {
    mockFetch.mockResolvedValue(jsonResponse({ id: "e1" }));
    await api.getExperiment("e1");
    expect(mockFetch).toHaveBeenCalledWith(
      "http://test-api/api/v1/experiments/e1",
      expect.any(Object),
    );
  });

  it("updateExperiment sends PATCH", async () => {
    mockFetch.mockResolvedValue(jsonResponse({ id: "e1" }));
    await api.updateExperiment("e1", { name: "Updated" });
    expect(mockFetch).toHaveBeenCalledWith(
      "http://test-api/api/v1/experiments/e1",
      expect.objectContaining({ method: "PATCH" }),
    );
  });

  it("deleteExperiment sends DELETE", async () => {
    mockFetch.mockResolvedValue(emptyResponse());
    await api.deleteExperiment("e1");
    expect(mockFetch).toHaveBeenCalledWith(
      "http://test-api/api/v1/experiments/e1",
      expect.objectContaining({ method: "DELETE" }),
    );
  });
});

describe("Sessions", () => {
  beforeEach(() => vi.clearAllMocks());

  it("createSession sends POST", async () => {
    mockFetch.mockResolvedValue(jsonResponse({ id: "s1" }));
    await api.createSession("e1", { strategy: "pair", max_turns: 5 });
    expect(mockFetch).toHaveBeenCalledWith(
      "http://test-api/api/v1/experiments/e1/sessions",
      expect.objectContaining({ method: "POST" }),
    );
  });

  it("createSession without data", async () => {
    mockFetch.mockResolvedValue(jsonResponse({ id: "s1" }));
    await api.createSession("e1");
    const call = mockFetch.mock.calls[0];
    expect(call[1].body).toBeUndefined();
  });

  it("listSessions with pagination", async () => {
    mockFetch.mockResolvedValue(jsonResponse({ items: [], total: 0 }));
    await api.listSessions("e1", 0, 10);
    expect(mockFetch).toHaveBeenCalledWith(
      "http://test-api/api/v1/experiments/e1/sessions?offset=0&limit=10",
      expect.any(Object),
    );
  });

  it("getSession fetches specific session", async () => {
    mockFetch.mockResolvedValue(jsonResponse({ id: "s1" }));
    await api.getSession("e1", "s1");
    expect(mockFetch).toHaveBeenCalledWith(
      "http://test-api/api/v1/experiments/e1/sessions/s1",
      expect.any(Object),
    );
  });

  it("startSession sends POST to start endpoint", async () => {
    mockFetch.mockResolvedValue(jsonResponse({ id: "s1", status: "running" }));
    await api.startSession("e1", "s1");
    expect(mockFetch).toHaveBeenCalledWith(
      "http://test-api/api/v1/experiments/e1/sessions/s1/start",
      expect.objectContaining({ method: "POST" }),
    );
  });

  it("deleteSession sends DELETE", async () => {
    mockFetch.mockResolvedValue(emptyResponse());
    await api.deleteSession("e1", "s1");
    expect(mockFetch).toHaveBeenCalledWith(
      "http://test-api/api/v1/experiments/e1/sessions/s1",
      expect.objectContaining({ method: "DELETE" }),
    );
  });
});

describe("Turns", () => {
  beforeEach(() => vi.clearAllMocks());

  it("listTurns with pagination", async () => {
    mockFetch.mockResolvedValue(jsonResponse({ items: [], total: 0 }));
    await api.listTurns("s1", 0, 50);
    expect(mockFetch).toHaveBeenCalledWith(
      "http://test-api/api/v1/sessions/s1/turns?offset=0&limit=50",
      expect.any(Object),
    );
  });

  it("getLatestTurn", async () => {
    mockFetch.mockResolvedValue(jsonResponse({ turn_number: 1 }));
    await api.getLatestTurn("s1");
    expect(mockFetch).toHaveBeenCalledWith(
      "http://test-api/api/v1/sessions/s1/turns/latest",
      expect.any(Object),
    );
  });
});

describe("Strategies", () => {
  beforeEach(() => vi.clearAllMocks());

  it("listStrategies", async () => {
    mockFetch.mockResolvedValue(jsonResponse({ strategies: [] }));
    await api.listStrategies();
    expect(mockFetch).toHaveBeenCalledWith(
      "http://test-api/api/v1/strategies",
      expect.any(Object),
    );
  });

  it("getStrategy by name", async () => {
    mockFetch.mockResolvedValue(jsonResponse({ name: "pair" }));
    await api.getStrategy("pair");
    expect(mockFetch).toHaveBeenCalledWith(
      "http://test-api/api/v1/strategies/pair",
      expect.any(Object),
    );
  });
});

describe("Targets", () => {
  beforeEach(() => vi.clearAllMocks());

  it("listModels", async () => {
    mockFetch.mockResolvedValue(jsonResponse({ models: ["llama3:8b"] }));
    await api.listModels();
    expect(mockFetch).toHaveBeenCalledWith(
      "http://test-api/api/v1/targets/models",
      expect.any(Object),
    );
  });

  it("checkHealth", async () => {
    mockFetch.mockResolvedValue(jsonResponse({ status: "ok" }));
    await api.checkHealth();
    expect(mockFetch).toHaveBeenCalledWith(
      "http://test-api/api/v1/targets/health",
      expect.any(Object),
    );
  });
});

describe("Defenses", () => {
  beforeEach(() => vi.clearAllMocks());

  it("listDefenses", async () => {
    mockFetch.mockResolvedValue(jsonResponse({ defenses: [] }));
    await api.listDefenses();
    expect(mockFetch).toHaveBeenCalledWith(
      "http://test-api/api/v1/defenses",
      expect.any(Object),
    );
  });
});

describe("Comparisons", () => {
  beforeEach(() => vi.clearAllMocks());

  it("compareSessions sends correct query params", async () => {
    mockFetch.mockResolvedValue(jsonResponse({}));
    await api.compareSessions("e1", "attack-s", "defense-s");
    expect(mockFetch).toHaveBeenCalledWith(
      expect.stringContaining("attack_session_id=attack-s&defense_session_id=defense-s"),
      expect.any(Object),
    );
  });
});

describe("Reports", () => {
  beforeEach(() => vi.clearAllMocks());

  it("getSessionReport", async () => {
    mockFetch.mockResolvedValue(jsonResponse({ session_id: "s1" }));
    await api.getSessionReport("e1", "s1");
    expect(mockFetch).toHaveBeenCalledWith(
      "http://test-api/api/v1/reports/e1/sessions/s1",
      expect.any(Object),
    );
  });

  it("exportReportJson", async () => {
    mockFetch.mockResolvedValue(jsonResponse({}));
    await api.exportReportJson("e1", "s1");
    expect(mockFetch).toHaveBeenCalledWith(
      expect.stringContaining("export?format=json"),
      expect.any(Object),
    );
  });
});

describe("Fine-Tuning", () => {
  beforeEach(() => vi.clearAllMocks());

  it("createFineTuningJob", async () => {
    mockFetch.mockResolvedValue(jsonResponse({ id: "j1" }));
    await api.createFineTuningJob({ job_type: "abliterate", base_model: "m" });
    expect(mockFetch).toHaveBeenCalledWith(
      "http://test-api/api/v1/finetuning/jobs",
      expect.objectContaining({ method: "POST" }),
    );
  });

  it("listFineTuningJobs with optional status filter", async () => {
    mockFetch.mockResolvedValue(jsonResponse({ items: [], total: 0 }));
    await api.listFineTuningJobs(0, 10, "completed");
    expect(mockFetch).toHaveBeenCalledWith(
      expect.stringContaining("status=completed"),
      expect.any(Object),
    );
  });

  it("listFineTuningJobs without status filter", async () => {
    mockFetch.mockResolvedValue(jsonResponse({ items: [], total: 0 }));
    await api.listFineTuningJobs(0, 10);
    expect(mockFetch).toHaveBeenCalledWith(
      expect.not.stringContaining("status="),
      expect.any(Object),
    );
  });

  it("getFineTuningJob", async () => {
    mockFetch.mockResolvedValue(jsonResponse({ id: "j1" }));
    await api.getFineTuningJob("j1");
    expect(mockFetch).toHaveBeenCalledWith(
      "http://test-api/api/v1/finetuning/jobs/j1",
      expect.any(Object),
    );
  });

  it("startFineTuningJob", async () => {
    mockFetch.mockResolvedValue(jsonResponse({ id: "j1" }));
    await api.startFineTuningJob("j1");
    expect(mockFetch).toHaveBeenCalledWith(
      "http://test-api/api/v1/finetuning/jobs/j1/start",
      expect.objectContaining({ method: "POST" }),
    );
  });

  it("cancelFineTuningJob", async () => {
    mockFetch.mockResolvedValue(jsonResponse({ status: "cancelled" }));
    await api.cancelFineTuningJob("j1");
    expect(mockFetch).toHaveBeenCalledWith(
      "http://test-api/api/v1/finetuning/jobs/j1/cancel",
      expect.objectContaining({ method: "POST" }),
    );
  });

  it("deleteFineTuningJob", async () => {
    mockFetch.mockResolvedValue(emptyResponse());
    await api.deleteFineTuningJob("j1");
    expect(mockFetch).toHaveBeenCalledWith(
      "http://test-api/api/v1/finetuning/jobs/j1",
      expect.objectContaining({ method: "DELETE" }),
    );
  });

  it("listCustomModels", async () => {
    mockFetch.mockResolvedValue(jsonResponse({ models: [] }));
    await api.listCustomModels();
    expect(mockFetch).toHaveBeenCalledWith(
      "http://test-api/api/v1/finetuning/models",
      expect.any(Object),
    );
  });

  it("deleteOllamaModel encodes name", async () => {
    mockFetch.mockResolvedValue(emptyResponse());
    await api.deleteOllamaModel("my-model:latest");
    expect(mockFetch).toHaveBeenCalledWith(
      expect.stringContaining(encodeURIComponent("my-model:latest")),
      expect.objectContaining({ method: "DELETE" }),
    );
  });
});

describe("Disk Management", () => {
  beforeEach(() => vi.clearAllMocks());

  it("getDiskStatus", async () => {
    mockFetch.mockResolvedValue(jsonResponse({ disk_total_gb: 100 }));
    await api.getDiskStatus();
    expect(mockFetch).toHaveBeenCalledWith(
      "http://test-api/api/v1/finetuning/disk-status",
      expect.any(Object),
    );
  });

  it("cleanupOrphans", async () => {
    mockFetch.mockResolvedValue(jsonResponse({ freed_gb: 1.5 }));
    await api.cleanupOrphans();
    expect(mockFetch).toHaveBeenCalledWith(
      "http://test-api/api/v1/finetuning/cleanup-orphans",
      expect.objectContaining({ method: "POST" }),
    );
  });
});

describe("Abliteration Dataset", () => {
  beforeEach(() => vi.clearAllMocks());

  it("listDatasetPrompts with optional filters", async () => {
    mockFetch.mockResolvedValue(jsonResponse({ items: [], total: 0 }));
    await api.listDatasetPrompts(0, 50, "harmful", "manual");
    expect(mockFetch).toHaveBeenCalledWith(
      expect.stringContaining("category=harmful"),
      expect.any(Object),
    );
    expect(mockFetch).toHaveBeenCalledWith(
      expect.stringContaining("source=manual"),
      expect.any(Object),
    );
  });

  it("getDatasetStats", async () => {
    mockFetch.mockResolvedValue(jsonResponse({ total: 100 }));
    await api.getDatasetStats();
    expect(mockFetch).toHaveBeenCalledWith(
      "http://test-api/api/v1/finetuning/dataset/prompts/stats",
      expect.any(Object),
    );
  });

  it("addDatasetPrompt with generation model", async () => {
    mockFetch.mockResolvedValue(jsonResponse({ id: "p1" }));
    await api.addDatasetPrompt({ text: "test", category: "harmful" }, "llama3:8b");
    expect(mockFetch).toHaveBeenCalledWith(
      expect.stringContaining("generation_model="),
      expect.objectContaining({ method: "POST" }),
    );
  });

  it("uploadDatasetFile uses FormData", async () => {
    mockFetch.mockResolvedValue(jsonResponse({ items: [] }));
    const file = new File(["content"], "test.jsonl", { type: "application/json" });
    await api.uploadDatasetFile(file);
    const call = mockFetch.mock.calls[0];
    expect(call[1].method).toBe("POST");
    expect(call[1].body).toBeInstanceOf(FormData);
  });

  it("uploadDatasetFile throws ApiError on failure", async () => {
    mockFetch.mockResolvedValue(errorResponse(400, "Bad file"));
    const file = new File(["bad"], "test.jsonl");
    await expect(api.uploadDatasetFile(file)).rejects.toThrow(api.ApiError);
  });

  it("updateDatasetPrompt", async () => {
    mockFetch.mockResolvedValue(jsonResponse({ id: "p1" }));
    await api.updateDatasetPrompt("p1", { text: "updated" });
    expect(mockFetch).toHaveBeenCalledWith(
      "http://test-api/api/v1/finetuning/dataset/prompts/p1",
      expect.objectContaining({ method: "PUT" }),
    );
  });

  it("deleteDatasetPrompt", async () => {
    mockFetch.mockResolvedValue(emptyResponse());
    await api.deleteDatasetPrompt("p1");
    expect(mockFetch).toHaveBeenCalledWith(
      "http://test-api/api/v1/finetuning/dataset/prompts/p1",
      expect.objectContaining({ method: "DELETE" }),
    );
  });

  it("generateHarmlessCounterpart with model", async () => {
    mockFetch.mockResolvedValue(jsonResponse({ id: "p2" }));
    await api.generateHarmlessCounterpart({ prompt_id: "p1" }, "llama3:8b");
    expect(mockFetch).toHaveBeenCalledWith(
      expect.stringContaining("model="),
      expect.objectContaining({ method: "POST" }),
    );
  });

  it("listDatasetSuggestions", async () => {
    mockFetch.mockResolvedValue(jsonResponse({ suggestions: [] }));
    await api.listDatasetSuggestions();
    expect(mockFetch).toHaveBeenCalledWith(
      "http://test-api/api/v1/finetuning/dataset/prompts/suggestions",
      expect.any(Object),
    );
  });

  it("confirmDatasetSuggestion", async () => {
    mockFetch.mockResolvedValue(jsonResponse({ id: "p1" }));
    await api.confirmDatasetSuggestion("p1", true, "llama3:8b");
    expect(mockFetch).toHaveBeenCalledWith(
      expect.stringContaining("confirm"),
      expect.objectContaining({ method: "POST" }),
    );
  });

  it("dismissDatasetSuggestion", async () => {
    mockFetch.mockResolvedValue(emptyResponse());
    await api.dismissDatasetSuggestion("p1");
    expect(mockFetch).toHaveBeenCalledWith(
      expect.stringContaining("dismiss"),
      expect.objectContaining({ method: "POST" }),
    );
  });
});

describe("Dataset Model Lifecycle", () => {
  beforeEach(() => vi.clearAllMocks());

  it("loadDatasetModel", async () => {
    mockFetch.mockResolvedValue(emptyResponse());
    await api.loadDatasetModel("llama3:8b");
    expect(mockFetch).toHaveBeenCalledWith(
      "http://test-api/api/v1/finetuning/dataset/model/load",
      expect.objectContaining({ method: "POST" }),
    );
  });

  it("unloadDatasetModel", async () => {
    mockFetch.mockResolvedValue(emptyResponse());
    await api.unloadDatasetModel("llama3:8b");
    expect(mockFetch).toHaveBeenCalledWith(
      "http://test-api/api/v1/finetuning/dataset/model/unload",
      expect.objectContaining({ method: "POST" }),
    );
  });

  it("getModelStatus", async () => {
    mockFetch.mockResolvedValue(jsonResponse({ models: [] }));
    await api.getModelStatus();
    expect(mockFetch).toHaveBeenCalledWith(
      "http://test-api/api/v1/finetuning/dataset/model/status",
      expect.any(Object),
    );
  });
});

describe("Playground", () => {
  beforeEach(() => vi.clearAllMocks());

  it("createPlaygroundConversation", async () => {
    mockFetch.mockResolvedValue(jsonResponse({ id: "c1" }));
    await api.createPlaygroundConversation({ title: "Test", target_model: "m" });
    expect(mockFetch).toHaveBeenCalledWith(
      "http://test-api/api/v1/playground/conversations",
      expect.objectContaining({ method: "POST" }),
    );
  });

  it("listPlaygroundConversations", async () => {
    mockFetch.mockResolvedValue(jsonResponse({ items: [], total: 0 }));
    await api.listPlaygroundConversations(0, 20);
    expect(mockFetch).toHaveBeenCalledWith(
      "http://test-api/api/v1/playground/conversations?offset=0&limit=20",
      expect.any(Object),
    );
  });

  it("getPlaygroundConversation", async () => {
    mockFetch.mockResolvedValue(jsonResponse({ id: "c1" }));
    await api.getPlaygroundConversation("c1");
    expect(mockFetch).toHaveBeenCalledWith(
      "http://test-api/api/v1/playground/conversations/c1",
      expect.any(Object),
    );
  });

  it("updatePlaygroundConversation", async () => {
    mockFetch.mockResolvedValue(jsonResponse({ id: "c1" }));
    await api.updatePlaygroundConversation("c1", { title: "Updated" });
    expect(mockFetch).toHaveBeenCalledWith(
      "http://test-api/api/v1/playground/conversations/c1",
      expect.objectContaining({ method: "PATCH" }),
    );
  });

  it("deletePlaygroundConversation", async () => {
    mockFetch.mockResolvedValue(emptyResponse());
    await api.deletePlaygroundConversation("c1");
    expect(mockFetch).toHaveBeenCalledWith(
      "http://test-api/api/v1/playground/conversations/c1",
      expect.objectContaining({ method: "DELETE" }),
    );
  });

  it("listPlaygroundMessages", async () => {
    mockFetch.mockResolvedValue(jsonResponse({ items: [], total: 0 }));
    await api.listPlaygroundMessages("c1", 0, 50);
    expect(mockFetch).toHaveBeenCalledWith(
      "http://test-api/api/v1/playground/conversations/c1/messages?offset=0&limit=50",
      expect.any(Object),
    );
  });

  it("sendPlaygroundMessage", async () => {
    mockFetch.mockResolvedValue(jsonResponse({ id: "m1" }));
    await api.sendPlaygroundMessage("c1", "Hello");
    expect(mockFetch).toHaveBeenCalledWith(
      "http://test-api/api/v1/playground/conversations/c1/messages",
      expect.objectContaining({ method: "POST" }),
    );
  });
});
