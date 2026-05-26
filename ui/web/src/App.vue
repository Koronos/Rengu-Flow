<template>
  <el-config-provider>
  <el-container class="app-shell">
    <el-aside v-if="!isMobile" width="220px" class="app-aside hide-mobile">
      <div class="app-brand">Renga Flow</div>
      <el-menu
        :default-active="activeMenu"
        class="app-menu"
        router
      >
        <el-menu-item index="/datasets">
          <el-icon><Files /></el-icon>
          <span>Datasets</span>
        </el-menu-item>
        <el-menu-item index="/configs">
          <el-icon><Setting /></el-icon>
          <span>Configs</span>
        </el-menu-item>
        <el-menu-item index="/jobs">
          <el-icon><VideoPlay /></el-icon>
          <span>Jobs</span>
        </el-menu-item>
        <el-menu-item index="/runs">
          <el-icon><FolderOpened /></el-icon>
          <span>Output runs</span>
        </el-menu-item>
      </el-menu>
    </el-aside>

    <el-container direction="vertical">
      <el-header class="app-header">
        <el-button
          v-if="isMobile"
          :icon="Menu"
          circle
          @click="drawerOpen = true"
        />
        <span class="app-brand-mobile hide-on-narrow">{{ pageTitle }}</span>
        <HostStatsBar />
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
      <el-menu
        :default-active="activeMenu"
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
        <el-menu-item index="/jobs">
          <el-icon><VideoPlay /></el-icon>
          <span>Jobs</span>
        </el-menu-item>
        <el-menu-item index="/runs">
          <el-icon><FolderOpened /></el-icon>
          <span>Output runs</span>
        </el-menu-item>
      </el-menu>
    </el-drawer>
  </el-container>
  </el-config-provider>
</template>

<script setup>
import { computed, ref } from "vue";
import { useRoute } from "vue-router";
import { Files, FolderOpened, Menu, Setting, VideoPlay } from "@element-plus/icons-vue";
import { useBreakpoint } from "./composables/useBreakpoint";
import HostStatsBar from "./components/HostStatsBar.vue";

const route = useRoute();
const { isMobile } = useBreakpoint();
const drawerOpen = ref(false);

const activeMenu = computed(() => {
  if (route.name === "configs") return "/configs";
  if (route.name === "datasets") return "/datasets";
  if (route.name === "jobs" || route.name === "job-detail") return "/jobs";
  return "/runs";
});

const pageTitle = computed(() => {
  const names = {
    jobs: "Jobs",
    "job-detail": "Job detail",
    configs: "Configs",
    datasets: "Datasets",
    runs: "Output runs",
    "run-detail": "Run detail",
  };
  return names[route.name] || "Renga Flow";
});
</script>

<style scoped>
.app-brand-mobile {
  font-weight: 600;
  font-size: 1rem;
  flex-shrink: 0;
}
@media (max-width: 900px) {
  .hide-on-narrow {
    display: none;
  }
}
</style>
