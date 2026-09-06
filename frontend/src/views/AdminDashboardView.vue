<script setup lang="ts">
import { computed, onMounted, onUnmounted, ref } from "vue";
import { useRouter } from "vue-router";

import { ApiError, api, auth } from "../api";
import type {
  AdminOverview,
  EvaluationReport,
  MetricItem,
  MetricsResponse,
  TaskDetail,
  TaskStatus,
  TaskSummary,
  TraceEvent,
} from "../types";

const router = useRouter();
const overview = ref<AdminOverview>({});
const tasks = ref<TaskSummary[]>([]);
const detail = ref<TaskDetail | null>(null);
const metrics = ref<MetricItem[]>([]);
const evaluation = ref<EvaluationReport>({});
const selectedTaskId = ref("");
const loading = ref(true);
const detailLoading = ref(false);
const error = ref("");

const pipeline: TaskStatus[] = [
  "RECEIVED",
  "INTAKE",
  "CASE_ANALYSIS",
  "ROUTING",
  "REVIEWING",
  "CREATE_LOCAL_CASE",
  "COMPLETED",
];

const overviewCards = computed(() => [
  { label: "任务总数", value: overview.value.total_tasks ?? tasks.value.length },
  { label: "运行中", value: overview.value.active_tasks ?? countStatus("active") },
  { label: "已完成", value: overview.value.completed_tasks ?? countStatus("completed") },
  { label: "失败", value: overview.value.failed_tasks ?? countStatus("failed") },
]);

const trace = computed<TraceEvent[]>(() => detail.value?.trace || detail.value?.events || []);
const memories = computed(() => detail.value?.memory || detail.value?.memories || []);

function countStatus(group: "active" | "completed" | "failed"): number {
  if (group === "completed") return tasks.value.filter((item) => item.status === "COMPLETED").length;
  if (group === "failed") return tasks.value.filter((item) => item.status === "FAILED").length;
  return tasks.value.filter((item) => !["COMPLETED", "FAILED", "CANCELLED"].includes(item.status)).length;
}

function taskItems(payload: TaskSummary[] | { tasks: TaskSummary[] }): TaskSummary[] {
  return Array.isArray(payload) ? payload : payload.tasks || [];
}

function metricItems(payload: MetricsResponse): MetricItem[] {
  if (Array.isArray(payload.metrics)) return payload.metrics;
  return Object.entries(payload)
    .filter(([, value]) => typeof value === "number" || typeof value === "string")
    .map(([name, value]) => ({ name, value: value as number | string }));
}

function readableName(name: string): string {
  const labels: Record<string, string> = {
    valid_transition_rate: "合法状态转移率",
    completion_rate: "任务完成率",
    checkpoint_resume_equivalence: "断点恢复一致率",
    compression_ratio: "上下文压缩率",
    critical_fact_recall: "关键事实保留率",
    tool_success_rate: "工具调用成功率",
    memory_recall_at_3: "记忆 Recall@3",
    skill_selection_accuracy: "Skill 选择准确率",
  };
  return labels[name] || name.replaceAll("_", " ");
}

function formatValue(value: number | string, unit?: string): string {
  if (typeof value === "number" && !unit && value >= 0 && value <= 1) {
    return `${(value * 100).toFixed(1)}%`;
  }
  return `${value}${unit || ""}`;
}

function shortId(value: string): string {
  return value.length > 13 ? `${value.slice(0, 8)}…${value.slice(-4)}` : value;
}

function eventAgent(event: TraceEvent): string {
  return String(event.agent_name || event.agent || event.skill_name || "Runtime");
}

function eventAction(event: TraceEvent): string {
  return String(event.summary || event.action || event.tool_name || "状态更新");
}

function pretty(value: unknown): string {
  return JSON.stringify(value ?? {}, null, 2);
}

function logout(): void {
  auth.clear();
  router.replace("/login");
}

async function handleError(reason: unknown): Promise<void> {
  if (reason instanceof ApiError && reason.status === 401) {
    logout();
    return;
  }
  error.value = reason instanceof Error ? reason.message : "数据加载失败。";
}

async function loadTask(taskId: string): Promise<void> {
  selectedTaskId.value = taskId;
  detailLoading.value = true;
  try {
    detail.value = await api.getTask(taskId);
  } catch (reason) {
    await handleError(reason);
  } finally {
    detailLoading.value = false;
  }
}

async function loadDashboard(): Promise<void> {
  loading.value = true;
  error.value = "";
  try {
    const [overviewData, taskData, metricData, reportData] = await Promise.all([
      api.getOverview(),
      api.getTasks(),
      api.getMetrics(),
      api.getEvaluation(),
    ]);
    overview.value = overviewData;
    tasks.value = taskItems(taskData);
    metrics.value = metricItems(metricData);
    evaluation.value = reportData;
    if (tasks.value.length) await loadTask(selectedTaskId.value || tasks.value[0].task_id);
  } catch (reason) {
    await handleError(reason);
  } finally {
    loading.value = false;
  }
}

onMounted(loadDashboard);

let refreshTimer: number | undefined;
onMounted(() => {
  refreshTimer = window.setInterval(() => {
    if (!loading.value && !detailLoading.value) void loadDashboard();
  }, 5000);
});
onUnmounted(() => {
  if (refreshTimer !== undefined) window.clearInterval(refreshTimer);
});
</script>

<template>
  <div class="admin-layout">
    <aside class="admin-sidebar">
      <div class="admin-brand"><span class="brand-mark">CN</span><strong>CivicNexus</strong></div>
      <nav aria-label="管理员导航">
        <a class="nav-item active" href="#overview">运行总览</a>
        <a class="nav-item" href="#tasks">任务与 Trace</a>
        <a class="nav-item" href="#evaluation">评测报告</a>
      </nav>
      <button class="logout-button" type="button" @click="logout">退出登录</button>
    </aside>

    <main class="admin-main">
      <header class="admin-header">
        <div><p class="eyebrow">ADMIN CONSOLE</p><h1>Agent 运行监控</h1></div>
        <button class="secondary-button" type="button" :disabled="loading" @click="loadDashboard">
          {{ loading ? "加载中…" : "刷新数据" }}
        </button>
      </header>

      <div v-if="error" class="error-banner page-error" role="alert">{{ error }}</div>

      <section id="overview" class="metric-grid" aria-label="运行总览">
        <article v-for="card in overviewCards" :key="card.label" class="summary-card">
          <span>{{ card.label }}</span><strong>{{ card.value }}</strong>
        </article>
      </section>

      <section class="panel metrics-panel">
        <div class="panel-heading"><div><p class="eyebrow">LIVE METRICS</p><h2>核心质量指标</h2></div></div>
        <div v-if="metrics.length" class="quality-grid">
          <article v-for="metric in metrics" :key="metric.name" class="quality-item">
            <div><span>{{ metric.label || readableName(metric.name) }}</span><small v-if="metric.target">目标 {{ metric.target }}</small></div>
            <strong :class="{ passed: metric.passed === true, failed: metric.passed === false }">
              {{ formatValue(metric.value, metric.unit) }}
            </strong>
          </article>
        </div>
        <p v-else class="empty-state">暂无运行指标。</p>
      </section>

      <section id="tasks" class="workspace-grid">
        <article class="panel task-list-panel">
          <div class="panel-heading"><div><p class="eyebrow">TASKS</p><h2>最近任务</h2></div><span>{{ tasks.length }}</span></div>
          <div class="task-list">
            <button
              v-for="task in tasks"
              :key="task.task_id"
              class="task-row"
              :class="{ selected: selectedTaskId === task.task_id }"
              type="button"
              @click="loadTask(task.task_id)"
            >
              <span><strong>{{ shortId(task.task_id) }}</strong><small>{{ task.user_message || task.current_agent || "城市公共服务事项" }}</small></span>
              <em :class="`status-${task.status.toLowerCase()}`">{{ task.status }}</em>
            </button>
            <p v-if="!tasks.length" class="empty-state">暂无任务记录。</p>
          </div>
        </article>

        <article class="panel detail-panel">
          <div class="panel-heading"><div><p class="eyebrow">TRACE</p><h2>执行详情</h2></div><span v-if="detail">{{ detail.status }}</span></div>
          <div v-if="detailLoading" class="empty-state">正在加载任务详情…</div>
          <template v-else-if="detail">
            <div class="pipeline" aria-label="状态流水线">
              <div v-for="state in pipeline" :key="state" :class="{ reached: pipeline.indexOf(state) <= pipeline.indexOf(detail.status) }">
                <i></i><span>{{ state }}</span>
              </div>
            </div>
            <div class="trace-list">
              <article v-for="(event, index) in trace" :key="event.sequence ?? index" class="trace-row">
                <span class="trace-index">{{ event.sequence ?? index + 1 }}</span>
                <div><strong>{{ eventAgent(event) }}</strong><p>{{ eventAction(event) }}</p></div>
                <code>{{ event.from_status || "—" }} → {{ event.to_status || "—" }}</code>
              </article>
              <p v-if="!trace.length" class="empty-state">该任务暂无 Trace。</p>
            </div>
          </template>
          <p v-else class="empty-state">选择任务后查看执行详情。</p>
        </article>
      </section>

      <section v-if="detail" class="inspect-grid">
        <article class="panel inspect-card"><div class="panel-heading"><h2>Context</h2></div><pre>{{ pretty(detail.context) }}</pre></article>
        <article class="panel inspect-card"><div class="panel-heading"><h2>Memory</h2><span>{{ memories.length }}</span></div><pre>{{ pretty(memories) }}</pre></article>
        <article class="panel inspect-card"><div class="panel-heading"><h2>Dynamic Skills</h2><span>{{ detail.skills?.length || 0 }}</span></div><pre>{{ pretty(detail.skills) }}</pre></article>
      </section>

      <section id="evaluation" class="panel report-panel">
        <div class="panel-heading">
          <div><p class="eyebrow">OFFLINE EVALUATION</p><h2>可复现评测报告</h2></div>
          <span v-if="evaluation.sample_count">样本 {{ evaluation.sample_count }}</span>
        </div>
        <pre>{{ pretty(evaluation) }}</pre>
      </section>
    </main>
  </div>
</template>
