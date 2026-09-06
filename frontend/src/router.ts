import { createRouter, createWebHistory } from "vue-router";

import { auth, userAuth } from "./api";
import AdminDashboardView from "./views/AdminDashboardView.vue";
import ChatView from "./views/ChatView.vue";
import RegisterView from "./views/RegisterView.vue";
import UserLoginView from "./views/UserLoginView.vue";

const router = createRouter({
  history: createWebHistory(),
  routes: [
    { path: "/", component: ChatView },
    { path: "/login", component: UserLoginView },
    { path: "/register", component: RegisterView },
    { path: "/user/login", redirect: "/login" },
    // Keep old bookmarks working without exposing a separate role-specific page.
    { path: "/admin/login", redirect: "/login" },
    { path: "/admin", component: AdminDashboardView, meta: { requiresAdmin: true } },
    { path: "/:pathMatch(.*)*", redirect: "/" },
  ],
});

router.beforeEach((to) => {
  if (to.meta.requiresAdmin && !auth.token) return "/login";
  if (to.path === "/login" || to.path === "/register" || to.path === "/user/login") {
    if (auth.token) return "/admin";
    if (userAuth.token) return "/";
  }
  return true;
});

export default router;
