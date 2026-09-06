export type TaskStatus =
  | "RECEIVED"
  | "INTAKE"
  | "WAITING_FOR_USER"
  | "CASE_ANALYSIS"
  | "ROUTING"
  | "REVIEWING"
  | "WAITING_FOR_HUMAN"
  | "CREATE_LOCAL_CASE"
  | "COMPLETED"
  | "CANCELLED"
  | "FAILED";

export interface ChatMessage {
  role: "user" | "assistant" | "system";
  content: string;
  created_at?: string;
}

export interface TaskConversation {
  task_id: string;
  status?: TaskStatus;
  messages: ChatMessage[];
  missing_fields?: string[];
  assistant_message?: string;
  message_zh?: string;
  reply?: string;
}

export interface TaskRequest {
  user_id: string;
  session_id: string;
  message: string;
}

export interface LoginResponse {
  access_token?: string;
  token?: string;
  token_type?: string;
  role?: "admin" | "user" | string;
  user_id?: string;
  display_name?: string;
  username?: string;
}

export interface AuthIdentity {
  role: "admin" | "user";
  user_id?: string;
  display_name: string;
  username?: string;
}

export interface AdminOverview {
  total_tasks?: number;
  active_tasks?: number;
  completed_tasks?: number;
  failed_tasks?: number;
  completion_rate?: number;
  [key: string]: unknown;
}

export interface TaskSummary {
  task_id: string;
  status: TaskStatus;
  user_message?: string;
  current_agent?: string;
  updated_at?: string;
  created_at?: string;
}

export interface TraceEvent {
  sequence?: number;
  timestamp?: string;
  agent?: string;
  agent_name?: string;
  from_status?: TaskStatus | string;
  to_status?: TaskStatus | string;
  action?: string;
  tool_name?: string;
  skill_name?: string;
  summary?: string;
  [key: string]: unknown;
}

export interface MemoryRecord {
  key?: string;
  value?: unknown;
  namespace?: string;
  importance?: number;
  expires_at?: string;
  [key: string]: unknown;
}

export interface SkillRecord {
  name?: string;
  version?: string;
  enabled?: boolean;
  selected?: boolean;
  tool?: string;
  [key: string]: unknown;
}

export interface TaskDetail extends TaskSummary {
  messages?: ChatMessage[];
  trace?: TraceEvent[];
  events?: TraceEvent[];
  context?: Record<string, unknown>;
  memory?: MemoryRecord[];
  memories?: MemoryRecord[];
  skills?: SkillRecord[];
}

export interface MetricItem {
  name: string;
  label?: string;
  value: number | string;
  unit?: string;
  target?: number | string;
  passed?: boolean;
}

export interface MetricsResponse {
  metrics?: MetricItem[];
  [key: string]: unknown;
}

export interface EvaluationReport extends MetricsResponse {
  generated_at?: string;
  sample_count?: number;
  passed?: boolean;
}
