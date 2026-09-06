<script setup lang="ts">
import { ref } from "vue";
import { useRouter } from "vue-router";

import { api, auth, userAuth } from "../api";

const router = useRouter();
const username = ref("");
const displayName = ref("");
const password = ref("");
const passwordConfirmation = ref("");
const pending = ref(false);
const error = ref("");

function validate(): string {
  if (username.value.trim().length < 3) return "用户名至少需要 3 个字符。";
  if (password.value.length < 6) return "密码至少需要 6 个字符。";
  if (password.value !== passwordConfirmation.value) return "两次输入的密码不一致。";
  return "";
}

async function register(): Promise<void> {
  if (pending.value) return;
  error.value = validate();
  if (error.value) return;

  pending.value = true;
  try {
    const response = await api.register(
      username.value.trim(),
      password.value,
      displayName.value,
    );
    const token = response.access_token || response.token;
    if (!token) throw new Error("注册响应中缺少访问令牌。");

    auth.clear();
    userAuth.clear();
    userAuth.set(token, {
      ...response,
      username: response.username || username.value.trim(),
    });
    await router.replace("/");
  } catch (reason) {
    error.value = reason instanceof Error ? reason.message : "注册失败。";
  } finally {
    pending.value = false;
  }
}
</script>

<template>
  <main class="login-page user-login-page">
    <section class="login-card">
      <div class="brand-mark large">CN</div>
      <p class="eyebrow">CIVICNEXUS</p>
      <h1>创建账户</h1>
      <p class="muted">注册后即可保存和继续你的公共服务事项</p>

      <form @submit.prevent="register">
        <label>
          用户名
          <input
            v-model="username"
            autocomplete="username"
            minlength="3"
            maxlength="128"
            required
            placeholder="请输入用户名"
          />
        </label>
        <label>
          显示名称（可选）
          <input
            v-model="displayName"
            autocomplete="nickname"
            maxlength="128"
            placeholder="例如：小王"
          />
        </label>
        <label>
          密码
          <input
            v-model="password"
            type="password"
            autocomplete="new-password"
            minlength="6"
            maxlength="256"
            required
            placeholder="至少 6 个字符"
          />
        </label>
        <label>
          确认密码
          <input
            v-model="passwordConfirmation"
            type="password"
            autocomplete="new-password"
            minlength="6"
            required
            placeholder="再次输入密码"
          />
        </label>
        <div v-if="error" class="error-banner" role="alert">{{ error }}</div>
        <button class="primary-button login-button" type="submit" :disabled="pending">
          {{ pending ? "正在创建…" : "注册" }}
        </button>
      </form>

      <RouterLink class="back-link" to="/login">已有账户？登录</RouterLink>
    </section>
  </main>
</template>
