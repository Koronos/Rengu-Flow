<template>
  <el-config-provider>
  <el-container class="app-shell">
    <el-aside v-if="!isMobile" width="220px" class="app-aside hide-mobile">
      <div class="app-brand">Rengu</div>
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
            <el-menu-item v-if="maintenanceNav" index="/maintenance">
              <el-icon><Tools /></el-icon>
              <span>Maintenance</span>
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
      title="Rengu"
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
            <el-menu-item v-if="maintenanceNav" index="/maintenance">
              <el-icon><Tools /></el-icon>
              <span>Maintenance</span>
            </el-menu-item>
          </el-menu>
        </div>
      </nav>
    </el-drawer>
  </el-container>
  <DatasetGalleryHost />
  <DatasetImageViewerHost />
  </el-config-provider>
</template>

<script setup lang="ts">
import { computed, onMounted, ref } from "vue";
import { useRoute } from "vue-router";
import { Document, Files, Menu, Setting, Tools, VideoPlay } from "@element-plus/icons-vue";
import { api } from "./api";
import { useBreakpoint } from "./composables/useBreakpoint";
import DatasetGalleryHost from "./components/DatasetGalleryHost.vue";
import DatasetImageViewerHost from "./components/DatasetImageViewerHost.vue";
import HostStatsBar from "./components/HostStatsBar.vue";

const route = useRoute();
const { isMobile } = useBreakpoint();
const drawerOpen = ref(false);
const maintenanceNav = ref(false);

onMounted(async () => {
  try {
    const r = await api.maintenanceEnabled();
    maintenanceNav.value = r.enabled;
  } catch {
    maintenanceNav.value = false;
  }
});

const routeName = computed(() =>
  typeof route.name === "string" ? route.name : ""
);

const activeMenu = computed(() => {
  const name = routeName.value;
  if (name === "docs") return "/docs";
  if (name === "maintenance") return "/maintenance";
  if (name.startsWith("configs-")) return "/configs";
  if (name.startsWith("datasets-")) return "/datasets";
  if (name === "jobs" || name === "job-detail" || name === "run-detail") return "/runs";
  return "/configs";
});

const pageTitle = computed(() => {
  const names: Record<string, string> = {
    docs: "Docs",
    maintenance: "Maintenance",
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
  return names[routeName.value] || "Rengu";
});
</script>

<style scoped>
@media (max-width: 900px) {
  .hide-on-narrow {
    display: none;
  }
}
</style>
