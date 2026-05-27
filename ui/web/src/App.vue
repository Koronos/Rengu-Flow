<template>
  <el-config-provider>
  <el-container class="app-shell">
    <el-aside v-if="!isMobile" width="220px" class="app-aside hide-mobile">
      <div class="app-brand">Renga Flow</div>
      <nav class="app-nav">
        <el-menu :default-active="activeMenu" class="app-menu app-menu--main" router>
          <el-menu-item index="/datasets">
            <el-icon><Files /></el-icon>
            <span>Datasets</span>
          </el-menu-item>
          <el-menu-item index="/configs">
            <el-icon><Setting /></el-icon>
            <span>Configs</span>
          </el-menu-item>
          <el-menu-item index="/runs">
            <el-icon><VideoPlay /></el-icon>
            <span>Runs</span>
          </el-menu-item>
        </el-menu>
        <div class="app-menu-bottom">
          <el-menu :default-active="activeMenu" class="app-menu app-menu--footer" router>
            <el-menu-item index="/docs">
              <el-icon><Document /></el-icon>
              <span>Docs</span>
            </el-menu-item>
          </el-menu>
        </div>
      </nav>
    </el-aside>

    <el-container direction="vertical">
      <el-header class="app-header">
        <el-button
          v-if="isMobile"
          :icon="Menu"
          circle
          @click="drawerOpen = true"
        />
        <span class="app-header-title hide-on-narrow">{{ pageTitle }}</span>
        <HostStatsBar class="host-stats-header" />
      </el-header>

      <el-main class="app-main">
        <RouterView />
      </el-main>
    </el-container>

    <el-drawer
      v-model="drawerOpen"
      direction="ltr"
      size="260px"
      title="Renga Flow"
    >
      <nav class="app-nav app-nav--drawer">
        <el-menu
          :default-active="activeMenu"
          class="app-menu app-menu--main"
          router
          @select="drawerOpen = false"
        >
          <el-menu-item index="/datasets">
            <el-icon><Files /></el-icon>
            <span>Datasets</span>
          </el-menu-item>
          <el-menu-item index="/configs">
            <el-icon><Setting /></el-icon>
            <span>Configs</span>
          </el-menu-item>
          <el-menu-item index="/runs">
            <el-icon><VideoPlay /></el-icon>
            <span>Runs</span>
          </el-menu-item>
        </el-menu>
        <div class="app-menu-bottom">
          <el-menu
            :default-active="activeMenu"
            class="app-menu app-menu--footer"
            router
            @select="drawerOpen = false"
          >
            <el-menu-item index="/docs">
              <el-icon><Document /></el-icon>
              <span>Docs</span>
            </el-menu-item>
          </el-menu>
        </div>
      </nav>
    </el-drawer>
  </el-container>
  </el-config-provider>
</template>

<script setup>
import { computed, ref } from "vue";
import { useRoute } from "vue-router";
import { Document, Files, Menu, Setting, VideoPlay } from "@element-plus/icons-vue";
import { useBreakpoint } from "./composables/useBreakpoint";
import HostStatsBar from "./components/HostStatsBar.vue";

const route = useRoute();
const { isMobile } = useBreakpoint();
const drawerOpen = ref(false);

const activeMenu = computed(() => {
  if (route.name === "docs") return "/docs";
  if (route.name?.startsWith("configs-")) return "/configs";
  if (route.name?.startsWith("datasets-")) return "/datasets";
  if (route.name === "jobs" || route.name === "job-detail" || route.name === "run-detail") return "/runs";
  return "/configs";
});

const pageTitle = computed(() => {
  const names = {
    docs: "Docs",
    jobs: "Runs",
    "job-detail": "Run detail",
    "configs-list": "Configs",
    "configs-new": "New config",
    "configs-detail": "Config",
    "datasets-list": "Datasets",
    "datasets-new": "New dataset",
    "datasets-detail": "Dataset",
    "run-detail": "Run detail",
  };
  return names[route.name] || "Renga Flow";
});
</script>

<style scoped>
@media (max-width: 900px) {
  .hide-on-narrow {
    display: none;
  }
}
</style>
