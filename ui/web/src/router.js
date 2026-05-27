import { createRouter, createWebHistory } from "vue-router";
import ConfigEditorView from "./views/ConfigEditorView.vue";
import ConfigsListView from "./views/ConfigsListView.vue";
import DatasetsListView from "./views/DatasetsListView.vue";
import DatasetEditorView from "./views/DatasetEditorView.vue";
import DocsView from "./views/DocsView.vue";
import JobsView from "./views/JobsView.vue";
import RunDetailView from "./views/RunDetailView.vue";

function configsListLegacyRedirect(to) {
  const { config, new: isNew, continue_run, pick, ...rest } = to.query;
  if (isNew === "1") {
    return {
      name: "configs-new",
      query: continue_run ? { continue_run, ...rest } : rest,
    };
  }
  if (typeof continue_run === "string" && continue_run) {
    return { name: "configs-new", query: { continue_run, ...rest } };
  }
  if (typeof config === "string" && config) {
    return {
      name: "configs-detail",
      params: { configId: config },
      query: pick ? { pick, ...rest } : rest,
    };
  }
  return true;
}

const router = createRouter({
  history: createWebHistory(),
  routes: [
    { path: "/", redirect: "/configs" },
    { path: "/docs", name: "docs", component: DocsView },
    { path: "/runs/jobs/:id", name: "job-detail", component: RunDetailView, props: { mode: "job" } },
    { path: "/runs", name: "jobs", component: JobsView },
    {
      path: "/runs/:name",
      name: "run-detail",
      component: RunDetailView,
      props: (route) => ({ mode: "fs", name: route.params.name }),
    },
    { path: "/jobs/:id", redirect: (to) => `/runs/jobs/${to.params.id}` },
    { path: "/jobs", redirect: "/runs" },
    {
      path: "/configs",
      name: "configs-list",
      component: ConfigsListView,
      beforeEnter: configsListLegacyRedirect,
    },
    { path: "/configs/new", name: "configs-new", component: ConfigEditorView },
    {
      path: "/configs/:configId",
      name: "configs-detail",
      component: ConfigEditorView,
      props: true,
    },
    { path: "/datasets", name: "datasets-list", component: DatasetsListView },
    { path: "/datasets/new", name: "datasets-new", component: DatasetEditorView },
    {
      path: "/datasets/:datasetId",
      name: "datasets-detail",
      component: DatasetEditorView,
      props: true,
    },
  ],
});

export default router;
