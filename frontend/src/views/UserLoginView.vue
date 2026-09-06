<script setup lang="ts">
import { ref } from "vue";
import { useRouter } from "vue-router";

import { api, auth, userAuth } from "../api";

const router = useRouter();
const username = ref("");
const password = ref("");
const pending = ref(false);
const error = ref("");

async function login(): Promise<void> {
  if (!username.value.trim() || !password.value || pending.value) return;
  pending.value = true;
  error.value = "";
  try {
    const response = await api.login(username.value.trim(), password.value);
    const token = response.access_token || response.token;
    if (!token) throw new Error("登录响应中缺少访问令牌。");
    const identity = { ...response, username: response.username || username.value.trim() };
    auth.clear();
    userAuth.clear();
    if (response.role === "admin") {
      auth.set(token, identity);
      await router.replace("/admin");
    } else {
      userAuth.set(token, identity);
      await router.replace("/");
    }
  } catch (reason) {
    error.value = reason instanceof Error ? reason.message : "登录失败。";
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
      <h1>登录 CivicNexus</h1>
      <p class="muted">登录后可在不同设备继续你的公共服务事项</p>

      <form @submit.prevent="login">
        <label>
          用户名
          <input v-model="username" autocomplete="username" placeholder="请输入用户名" />
        </label>
        <label>
          密码
          <input v-model="password" type="password" autocomplete="current-password" placeholder="请输入密码" />
        </label>
        <div v-if="error" class="error-banner" role="alert">{{ error }}</div>
        <button class="primary-button login-button" type="submit" :disabled="pending">
          {{ pending ? "正在验证…" : "登录" }}
        </button>
      </form>

      <div class="login-options">
        <RouterLink class="secondary-button register-button" to="/register">注册账户</RouterLink>
        <RouterLink class="back-link" to="/">暂不登录，以访客继续</RouterLink>
      </div>
    </section>
  </main>
</template>
