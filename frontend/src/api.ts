import type {
  AdminOverview,
  EvaluationReport,
  LoginResponse,
  MetricsResponse,
  TaskConversation,
  TaskDetail,
  TaskRequest,
  TaskSummary,
  AuthIdentity,
} from "./types";

const API_BASE = (import.meta.env.VITE_API_BASE_URL || "/api/v1").replace(/\/$/, "");
const TOKEN_KEY = "civicnexus_admin_token";
const USER_TOKEN_KEY = "civicnexus_user_token";
const USER_ID_KEY = "civicnexus_auth_user_id";
const USER_NAME_KEY = "civicnexus_auth_display_name";
const USERNAME_KEY = "civicnexus_auth_username";
const ADMIN_NAME_KEY = "civicnexus_admin_display_name";

function profile(role: "admin" | "user"): AuthIdentity | null {
  const storage = sessionStorage;
  const token = storage.getItem(role === "admin" ? TOKEN_KEY : USER_TOKEN_KEY);
  if (!token) return null;
  const displayName = storage.getItem(role === "admin" ? ADMIN_NAME_KEY : USER_NAME_KEY);
  return {
    role,
    user_id: role === "user" ? storage.getItem(USER_ID_KEY) || undefined : undefined,
    username: role === "user" ? storage.getItem(USERNAME_KEY) || undefined : undefined,
    display_name: displayName || (role === "admin" ? "管理员" : "用户"),
  };
}

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
  ) {
    super(message);
  }
}

export const auth = {
  get token(): string {
    return sessionStorage.getItem(TOKEN_KEY) || "";
  },
  get identity(): AuthIdentity | null {
    return profile("admin");
  },
  set(token: string, info: Partial<LoginResponse> = {}): void {
    sessionStorage.setItem(TOKEN_KEY, token);
    sessionStorage.setItem(ADMIN_NAME_KEY, info.display_name || info.username || "管理员");
  },
  clear(): void {
    sessionStorage.removeItem(TOKEN_KEY);
    sessionStorage.removeItem(ADMIN_NAME_KEY);
  },
};

export const userAuth = {
  get token(): string {
    return sessionStorage.getItem(USER_TOKEN_KEY) || "";
  },
  get identity(): AuthIdentity | null {
    return profile("user");
  },
  set(token: string, info: Partial<LoginResponse> = {}): void {
    sessionStorage.setItem(USER_TOKEN_KEY, token);
    if (info.user_id) sessionStorage.setItem(USER_ID_KEY, info.user_id);
    if (info.username) sessionStorage.setItem(USERNAME_KEY, info.username);
    sessionStorage.setItem(USER_NAME_KEY, info.display_name || info.username || info.user_id || "用户");
  },
  clear(): void {
    sessionStorage.removeItem(USER_TOKEN_KEY);
    sessionStorage.removeItem(USER_ID_KEY);
    sessionStorage.removeItem(USER_NAME_KEY);
    sessionStorage.removeItem(USERNAME_KEY);
  },
};

type AuthScope = "admin" | "user" | "none";

async function request<T>(path: string, init: RequestInit = {}, scope: AuthScope | boolean = "user"): Promise<T> {
  const headers = new Headers(init.headers);
  headers.set("Accept", "application/json");
  if (init.body) headers.set("Content-Type", "application/json");
  const resolvedScope: AuthScope = scope === true ? "admin" : scope === false ? "none" : scope;
  const token = resolvedScope === "admin" ? auth.token : resolvedScope === "user" ? userAuth.token : "";
  if (token) headers.set("Authorization", `Bearer ${token}`);

  const response = await fetch(`${API_BASE}${path}`, { ...init, headers });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    const detail = payload.detail || payload.message || `请求失败（HTTP ${response.status}）`;
    throw new ApiError(String(detail), response.status);
  }
  return payload as T;
}

async function stream<T>(path: string, init: RequestInit, onEvent: (event: string, data: any) => void, scope: AuthScope | boolean = "user"): Promise<T> {
  const headers = new Headers(init.headers);
  headers.set("Accept", "text/event-stream");
  headers.set("Content-Type", "application/json");
  const resolvedScope: AuthScope = scope === true ? "admin" : scope === false ? "none" : scope;
  const token = resolvedScope === "admin" ? auth.token : resolvedScope === "user" ? userAuth.token : "";
  if (token) headers.set("Authorization", `Bearer ${token}`);
  const response = await fetch(`${API_BASE}${path}`, { ...init, headers });
  if (!response.ok || !response.body) {
    const payload = await response.json().catch(() => ({}));
    throw new ApiError(String(payload.detail || `请求失败（HTTP ${response.status}）`), response.status);
  }
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let completed: T | undefined;
  while (true) {
    const chunk = await reader.read();
    buffer += decoder.decode(chunk.value || new Uint8Array(), { stream: !chunk.done });
    const blocks = buffer.split("\n\n");
    buffer = blocks.pop() || "";
    for (const block of blocks) {
      const event = block.match(/^event: (.+)$/m)?.[1] || "message";
      const line = block.match(/^data: (.+)$/m)?.[1];
      if (!line) continue;
      const data = JSON.parse(line);
      if (event === "error") throw new ApiError(data.message || "请求失败", response.status || 500);
      if (event === "complete") completed = data as T;
      onEvent(event, data);
    }
    if (chunk.done) break;
  }
  if (completed === undefined) throw new ApiError("流式响应未完成", 502);
  return completed;
}

export const api = {
  createTask: (body: TaskRequest) =>
    request<TaskConversation>("/tasks", { method: "POST", body: JSON.stringify(body) }),

  streamTask: (body: TaskRequest, onEvent: (event: string, data: any) => void) =>
    stream<TaskConversation>("/tasks/stream", { method: "POST", body: JSON.stringify(body) }, onEvent),

  sendMessage: (taskId: string, body: TaskRequest) =>
    request<TaskConversation>(`/tasks/${encodeURIComponent(taskId)}/messages`, {
      method: "POST",
      body: JSON.stringify(body),
    }),

  streamMessage: (taskId: string, body: TaskRequest, onEvent: (event: string, data: any) => void) =>
    stream<TaskConversation>(`/tasks/${encodeURIComponent(taskId)}/messages/stream`, {
      method: "POST",
      body: JSON.stringify(body),
    }, onEvent),

  getMessages: (taskId: string, userId?: string, sessionId?: string) => {
    const query = userId && sessionId ? `?user_id=${encodeURIComponent(userId)}&session_id=${encodeURIComponent(sessionId)}` : "";
    return request<TaskConversation>(`/tasks/${encodeURIComponent(taskId)}/messages${query}`);
  },

  // One form and one endpoint; the API returns the role after authentication.
  login: (username: string, password: string) =>
    request<LoginResponse>("/auth/login", {
      method: "POST",
      body: JSON.stringify({ username, password }),
    }, "none"),

  register: (username: string, password: string, displayName?: string) =>
    request<LoginResponse>("/auth/register", {
      method: "POST",
      body: JSON.stringify({
        username,
        password,
        ...(displayName?.trim() ? { display_name: displayName.trim() } : {}),
      }),
    }, "none"),

  // Kept for callers that need to explicitly target the user credential store.
  userLogin: (username: string, password: string) =>
    request<LoginResponse>("/auth/login", {
      method: "POST",
      body: JSON.stringify({ username, password }),
    }, "none"),

  getOverview: () => request<AdminOverview>("/admin/overview", {}, true),

  getTasks: () => request<TaskSummary[] | { tasks: TaskSummary[] }>("/admin/tasks", {}, true),

  getTask: (taskId: string) =>
    request<TaskDetail>(`/admin/tasks/${encodeURIComponent(taskId)}`, {}, true),

  getMetrics: () => request<MetricsResponse>("/admin/metrics", {}, true),

  getEvaluation: () => request<EvaluationReport>("/admin/evaluation/report", {}, true),
};
