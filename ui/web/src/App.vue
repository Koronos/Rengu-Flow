<template>
  <el-config-provider>
  <el-container class="app-shell">
    <el-aside v-if="!isMobile" width="220px" class="app-aside hide-mobile">
      <router-link to="/" class="app-brand">Rengu Flow UI</router-link>
      <nav class="app-nav">
        <el-menu :default-active="activeMenu" class="app-menu app-menu--main" router>
          <el-menu-item index="/runs">
            <el-icon><VideoPlay /></el-icon>
            <span>Runs</span>
          </el-menu-item>
          <el-menu-item index="/datasets">
            <el-icon><Files /></el-icon>
            <span>Datasets</span>
          </el-menu-item>
        </el-menu>
        <div
          v-if="versionLabel"
          class="app-brand-version"
          :title="versionTitle"
        >
          {{ versionLabel }}
        </div>
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
      title="Rengu Flow UI"
    >
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
          <el-menu-item index="/datasets">
            <el-icon><Files /></el-icon>
            <span>Datasets</span>
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
  <DatasetFormModalHost />
  </el-config-provider>
</template>

<script setup lang="ts">
import { computed, onMounted, ref } from "vue";
import { useRoute } from "vue-router";
import { Document, Files, Menu, Tools, VideoPlay } from "@element-plus/icons-vue";
import { api } from "./api";
import { useBreakpoint } from "./composables/useBreakpoint";
import DatasetGalleryHost from "./components/DatasetGalleryHost.vue";
import DatasetImageViewerHost from "./components/DatasetImageViewerHost.vue";
import DatasetFormModalHost from "./components/DatasetFormModalHost.vue";
import HostStatsBar from "./components/HostStatsBar.vue";

const route = useRoute();
const { isMobile } = useBreakpoint();
const drawerOpen = ref(false);
const maintenanceNav = ref(false);
const versionLabel = ref("");
const versionTitle = ref("");

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
    versionTitle.value = v.koptim
      ? `rengu-flow ${v.version}${v.commit ? ` (${v.commit})` : ""} · koptim ${v.koptim}`
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
  if (name.startsWith("datasets-")) return "/datasets";
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
    "run-detail": "Run detail",
  };
  return names[routeName.value] || "Rengu Flow UI";
});
</script>

<style scoped>
.app-brand-version {
  /* Pinned to the bottom, just above the Docs/Maintenance panel. */
  margin-top: auto;
  padding: 8px 16px;
  font-size: 11px;
  line-height: 1.4;
  color: var(--el-text-color-secondary);
  opacity: 0.75;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

@media (max-width: 900px) {
  .hide-on-narrow {
    display: none;
  }
}
</style>
