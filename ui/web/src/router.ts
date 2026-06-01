import { createRouter, createWebHistory } from "vue-router";
import DatasetsListView from "./views/DatasetsListView.vue";
import DatasetEditorView from "./views/DatasetEditorView.vue";
import DocsView from "./views/DocsView.vue";
import MaintenanceView from "./views/MaintenanceView.vue";
import JobsView from "./views/JobsView.vue";
import RunDetailView from "./views/RunDetailView.vue";

const router = createRouter({
  history: createWebHistory(),
  routes: [
    { path: "/", redirect: "/runs" },
    { path: "/docs", name: "docs", component: DocsView },
    { path: "/maintenance", name: "maintenance", component: MaintenanceView },
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
