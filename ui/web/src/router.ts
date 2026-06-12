import { createRouter, createWebHistory } from "vue-router";
import DatasetsListView from "./views/DatasetsListView.vue";
import DocsView from "./views/DocsView.vue";
import MaintenanceView from "./views/MaintenanceView.vue";
import JobsView from "./views/JobsView.vue";
import RunComparisonView from "./views/RunComparisonView.vue";
import RunDetailView from "./views/RunDetailView.vue";
import RunFormView from "./views/RunFormView.vue";
import TagEditorView from "./views/TagEditorView.vue";
import PrepJobsView from "./views/PrepJobsView.vue";
import PrepJobFormView from "./views/PrepJobFormView.vue";

const router = createRouter({
  history: createWebHistory(),
  routes: [
    { path: "/", redirect: "/runs" },
    { path: "/docs", name: "docs", component: DocsView },
    { path: "/maintenance", name: "maintenance", component: MaintenanceView },
    { path: "/runs/new", name: "run-new", component: RunFormView },
    { path: "/runs/jobs/:id/edit", name: "run-edit", component: RunFormView },
    { path: "/runs/jobs/:id/continue", name: "run-continue", component: RunFormView },
    { path: "/runs/jobs/:id", name: "job-detail", component: RunDetailView, props: { mode: "job" } },
    { path: "/runs", name: "jobs", component: JobsView },
    { path: "/compare", name: "run-comparison", component: RunComparisonView },
    {
      path: "/runs/:name",
      name: "run-detail",
      component: RunDetailView,
      props: (route) => ({ mode: "fs", name: route.params.name }),
    },
    { path: "/jobs/:id", redirect: (to) => `/runs/jobs/${to.params.id}` },
    { path: "/jobs", redirect: "/runs" },
    { path: "/datasets", name: "datasets-list", component: DatasetsListView },
    { path: "/prep", name: "prep-jobs", component: PrepJobsView },
    { path: "/prep/new/:stage", name: "prep-new", component: PrepJobFormView },
    { path: "/prep/tags", name: "prep-tags", component: TagEditorView },
  ],
});

export default router;
