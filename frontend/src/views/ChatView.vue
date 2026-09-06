<script setup lang="ts">
import { computed, nextTick, onMounted, ref } from "vue";

import { api, auth, userAuth } from "../api";
import type { AuthIdentity, ChatMessage, TaskConversation, TaskRequest } from "../types";

const TASK_KEY = "civicnexus_task_id";
const SESSION_KEY = "civicnexus_session_id";
const USER_KEY = "civicnexus_user_id";

const taskId = ref(sessionStorage.getItem(TASK_KEY) || "");
const input = ref("");
const pending = ref(false);
const processingLabel = ref("正在思考并处理事项…");
const error = ref("");
const messageList = ref<HTMLElement>();
const messages = ref<ChatMessage[]>([]);

const guestUserId = persistentId(localStorage, USER_KEY, "user");
const sessionId = persistentId(sessionStorage, SESSION_KEY, "session");
const identity = ref<AuthIdentity | null>(userAuth.identity || auth.identity);
const accountMenuOpen = ref(false);
const currentUserId = computed(() => identity.value?.role === "user" && identity.value.user_id ? identity.value.user_id : guestUserId);

function initials(name: string): string {
  return name.trim().slice(0, 1).toUpperCase() || "U";
}

function persistentId(storage: Storage, key: string, prefix: string): string {
  const existing = storage.getItem(key);
  if (existing) return existing;
  const value = `${prefix}-${crypto.randomUUID()}`;
  storage.setItem(key, value);
  return value;
}

function applyConversation(payload: TaskConversation): void {
  if (payload.task_id) {
    taskId.value = payload.task_id;
    sessionStorage.setItem(TASK_KEY, payload.task_id);
  }
  const visibleMessages = (payload.messages || []).filter((item) => item.role === "user" || item.role === "assistant");
  if (visibleMessages.length) {
    messages.value = visibleMessages;
    return;
  }

  const reply = payload.assistant_message || payload.reply || payload.message_zh;
  if (reply) messages.value.push({ role: "assistant", content: reply });
}

async function scrollToBottom(): Promise<void> {
  await nextTick();
  if (messageList.value) messageList.value.scrollTop = messageList.value.scrollHeight;
}

async function send(): Promise<void> {
  const content = input.value.trim();
  if (!content || pending.value) return;

  input.value = "";
  error.value = "";
  processingLabel.value = "正在接收并整理事项…";
  messages.value.push({ role: "user", content });
  const streamedReply: ChatMessage = { role: "assistant", content: "" };
  pending.value = true;
  await scrollToBottom();

  const body: TaskRequest = { user_id: currentUserId.value, session_id: sessionId, message: content };
  try {
    const receive = (event: string, data: { content?: string; state?: string }) => {
      if (event === "status") {
        processingLabel.value = data.state === "started"
          ? "正在接收并整理事项…"
          : "正在思考并处理事项…";
      }
      if (event === "delta") {
        processingLabel.value = "正在准备回复…";
        if (!messages.value.includes(streamedReply)) messages.value.push(streamedReply);
        streamedReply.content += data.content || "";
        void scrollToBottom();
      }
    };
    const response = taskId.value
      ? await api.streamMessage(taskId.value, body, receive)
      : await api.streamTask(body, receive);
    applyConversation(response);
  } catch {
    if (!streamedReply.content) messages.value = messages.value.filter((item) => item !== streamedReply);
    // Keep provider/database details out of the public chat surface.
    error.value = "暂时无法回复，请稍后再试。";
  } finally {
    pending.value = false;
    await scrollToBottom();
  }
}

function logout(): void {
  userAuth.clear();
  auth.clear();
  identity.value = null;
  accountMenuOpen.value = false;
  resetChat();
}

function closeAccountMenu(): void {
  accountMenuOpen.value = false;
}

function resetChat(): void {
  if (pending.value) return;
  taskId.value = "";
  input.value = "";
  processingLabel.value = "正在思考并处理事项…";
  error.value = "";
  sessionStorage.removeItem(TASK_KEY);
  messages.value = [];
}

onMounted(async () => {
  if (!taskId.value) return;
  try {
    applyConversation(await api.getMessages(taskId.value, currentUserId.value, sessionId));
    await scrollToBottom();
  } catch {
    sessionStorage.removeItem(TASK_KEY);
    taskId.value = "";
  }
});
</script>

<template>
  <main class="chat-page">
    <aside class="chat-sidebar" aria-label="会话导航">
      <div class="sidebar-brand"><span class="brand-mark">CN</span><strong>CivicNexus</strong></div>
      <button class="new-chat-button" type="button" @click="resetChat">＋ 新对话</button>
      <div class="sidebar-account">
        <template v-if="identity">
          <button
            class="account-button"
            type="button"
            :aria-expanded="accountMenuOpen"
            aria-haspopup="menu"
            @click="accountMenuOpen = !accountMenuOpen"
          >
            <span class="account-avatar">{{ initials(identity.display_name) }}</span>
            <span class="account-copy"><strong>{{ identity.display_name }}</strong><small>{{ identity.role === "admin" ? "管理员" : "普通用户" }}</small></span>
            <span class="account-chevron" aria-hidden="true">⌄</span>
          </button>
          <div v-if="accountMenuOpen" class="account-menu" role="menu">
            <RouterLink v-if="identity.role === 'admin'" to="/admin" role="menuitem" @click="closeAccountMenu">管理控制台</RouterLink>
            <button type="button" role="menuitem" @click="logout">退出登录</button>
          </div>
        </template>
        <RouterLink v-else class="guest-account" to="/login">
          <span class="account-avatar guest">↗</span>
          <span class="account-copy"><strong>登录账户</strong><small>登录以保存对话</small></span>
        </RouterLink>
      </div>
    </aside>
    <section class="chat-shell" aria-label="城市公共服务对话">
      <header class="chat-header">
        <h1>CivicNexus</h1>
        <button v-if="taskId" class="text-button" type="button" @click="resetChat">新对话</button>
      </header>

      <div ref="messageList" class="messages" aria-live="polite">
        <article
          v-for="(message, index) in messages"
          :key="`${index}-${message.role}`"
          class="message-row"
          :class="message.role"
        >
          <div class="avatar">{{ message.role === "user" ? "我" : "CN" }}</div>
          <div class="message-bubble">{{ message.content }}</div>
        </article>
      </div>

      <div v-if="error" class="error-banner" role="alert">{{ error }}</div>
      <div v-if="pending" class="processing-strip" role="status" aria-live="polite">
        <span class="processing-indicator" aria-hidden="true"><i></i><i></i><i></i></span>
        <span>{{ processingLabel }}</span>
      </div>

      <form class="composer" @submit.prevent="send">
        <textarea
          v-model="input"
          rows="2"
          maxlength="500"
          aria-label="输入问题"
          @keydown.enter.exact.prevent="send"
        ></textarea>
        <button class="primary-button send-button" type="submit" :disabled="pending || !input.trim()">
          发送
        </button>
      </form>
    </section>
  </main>
</template>
