<template>
  <el-config-provider>
  <el-container class="app-shell">
    <el-aside v-if="!isMobile" width="220px" class="app-aside hide-mobile">
      <router-link to="/" class="app-brand">
        <img src="/icon.svg" class="app-brand__icon" alt="" width="24" height="24" />
        <span>Rengu Flow UI</span>
      </router-link>
      <nav class="app-nav">
        <el-menu :default-active="activeMenu" class="app-menu app-menu--main" router>
          <el-menu-item index="/runs">
            <el-icon><VideoPlay /></el-icon>
            <span>Runs</span>
          </el-menu-item>
          <el-menu-item index="/compare">
            <el-icon><TrendCharts /></el-icon>
            <span>Compare</span>
          </el-menu-item>
          <el-menu-item index="/datasets">
            <el-icon><Files /></el-icon>
            <span>Datasets</span>
          </el-menu-item>
          <el-menu-item index="/prep">
            <el-icon><MagicStick /></el-icon>
            <span>Studio</span>
          </el-menu-item>
          <el-menu-item index="/toolbox">
            <el-icon><Tools /></el-icon>
            <span>Toolbox</span>
          </el-menu-item>
        </el-menu>
        <div v-if="versionLabel" class="app-brand-version">
          <span class="app-brand-version__label" :title="versionTitle">{{ versionLabel }}</span>
          <a
            class="app-brand-version__gh"
            :href="REPO_URL"
            target="_blank"
            rel="noopener noreferrer"
            title="View on GitHub"
            aria-label="View on GitHub"
          >
            <svg viewBox="0 0 16 16" width="15" height="15" aria-hidden="true" fill="currentColor">
              <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.65-.89-3.65-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82a7.65 7.65 0 012-.27c.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.01 8.01 0 0016 8c0-4.42-3.58-8-8-8z"/>
            </svg>
          </a>
        </div>
        <div class="app-menu-bottom">
          <ThemeToggle />
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

    <el-drawer v-model="drawerOpen" direction="ltr" size="260px">
      <template #header>
        <span class="app-brand app-brand--drawer">
          <img src="/icon.svg" class="app-brand__icon" alt="" width="22" height="22" />
          <span>Rengu Flow UI</span>
        </span>
      </template>
      <nav class="app-nav app-nav--drawer">
        <el-menu
          :default-active="activeMenu"
          class="app-menu app-menu--main"
          router
          @select="drawerOpen = false"
        >
          <el-menu-item index="/runs">
            <el-icon><VideoPlay /></el-icon>
            <span>Runs</span>
          </el-menu-item>
          <el-menu-item index="/compare">
            <el-icon><TrendCharts /></el-icon>
            <span>Compare</span>
          </el-menu-item>
          <el-menu-item index="/datasets">
            <el-icon><Files /></el-icon>
            <span>Datasets</span>
          </el-menu-item>
          <el-menu-item index="/prep">
            <el-icon><MagicStick /></el-icon>
            <span>Studio</span>
          </el-menu-item>
          <el-menu-item index="/toolbox">
            <el-icon><Tools /></el-icon>
            <span>Toolbox</span>
          </el-menu-item>
        </el-menu>
        <div class="app-menu-bottom">
          <ThemeToggle />
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
  <DatasetFormModalHost />
  </el-config-provider>
</template>

<script setup lang="ts">
import { computed, onMounted, ref } from "vue";
import { useRoute } from "vue-router";
import { Document, Files, MagicStick, Menu, Tools, TrendCharts, VideoPlay } from "@element-plus/icons-vue";
import { api } from "./api";
import { useBreakpoint } from "./composables/useBreakpoint";
import DatasetGalleryHost from "./components/DatasetGalleryHost.vue";
import DatasetImageViewerHost from "./components/DatasetImageViewerHost.vue";
import DatasetFormModalHost from "./components/DatasetFormModalHost.vue";
import HostStatsBar from "./components/HostStatsBar.vue";
import ThemeToggle from "./components/ThemeToggle.vue";

const route = useRoute();
const { isMobile } = useBreakpoint();
const drawerOpen = ref(false);
const maintenanceNav = ref(false);
const versionLabel = ref("");
const versionTitle = ref("");
const REPO_URL = "https://github.com/Koronos/Rengu-Flow";

onMounted(async () => {
  try {
    const r = await api.maintenanceEnabled();
    maintenanceNav.value = r.enabled;
  } catch {
    maintenanceNav.value = false;
  }
  try {
    const v = await api.version();
    versionLabel.value = v.commit ? `v${v.version} · ${v.commit}` : `v${v.version}`;
    versionTitle.value = v.kaon
      ? `rengu-flow ${v.version}${v.commit ? ` (${v.commit})` : ""} · kaon ${v.kaon}`
      : `rengu-flow ${v.version}${v.commit ? ` (${v.commit})` : ""}`;
  } catch {
    versionLabel.value = "";
  }
});

const routeName = computed(() =>
  typeof route.name === "string" ? route.name : ""
);

const activeMenu = computed(() => {
  const name = routeName.value;
  if (name === "docs") return "/docs";
  if (name === "maintenance") return "/maintenance";
  if (name === "run-comparison") return "/compare";
  if (name.startsWith("datasets-")) return "/datasets";
  if (name.startsWith("prep-")) return "/prep";
  if (name.startsWith("toolbox")) return "/toolbox";
  return "/runs";
});

const pageTitle = computed(() => {
  const names: Record<string, string> = {
    docs: "Docs",
    maintenance: "Maintenance",
    jobs: "Runs",
    "job-detail": "Run detail",
    "run-new": "New run",
    "run-edit": "Edit run",
    "run-continue": "Continue training",
    "datasets-list": "Datasets",
    "run-comparison": "Compare runs",
    "run-detail": "Run detail",
    "prep-jobs": "Dataset Studio",
    "prep-new": "Dataset Studio · New job",
    "prep-tags": "Dataset Studio · Tag editor",
    toolbox: "Toolbox",
    "toolbox-new": "Toolbox · New tool",
    "toolbox-edit": "Toolbox · Edit tool",
  };
  return names[routeName.value] || "Rengu Flow UI";
});
</script>

<style scoped>
.app-brand-version {
  /* Pinned to the bottom, just above the Docs/Maintenance panel. */
  margin-top: auto;
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 8px 16px;
  font-size: 11px;
  line-height: 1.4;
  color: var(--el-text-color-secondary);
}
.app-brand-version__label {
  min-width: 0;
  opacity: 0.75;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
.app-brand-version__gh {
  display: inline-flex;
  flex-shrink: 0;
  color: var(--el-text-color-secondary);
  opacity: 0.75;
  transition: opacity 0.15s ease, color 0.15s ease;
}
.app-brand-version__gh:hover {
  opacity: 1;
  color: var(--el-color-primary);
}

@media (max-width: 900px) {
  .hide-on-narrow {
    display: none;
  }
}
</style>
