import { createRouter, createWebHistory } from "vue-router";

// Lazy-loaded views: each becomes its own chunk so the initial load ships only the shell + the
// landing route, not every view (and their heavy deps — uPlot for compare, marked/dompurify for
// docs, the tag editor, etc.). The rest load on navigation. RunDetailView and RunFormView back
// several routes, so their import is shared (one promise) to avoid duplicating the chunk.
const RunDetailView = () => import("./views/RunDetailView.vue");
const RunFormView = () => import("./views/RunFormView.vue");

const router = createRouter({
  history: createWebHistory(),
  routes: [
    { path: "/", redirect: "/runs" },
    { path: "/docs", name: "docs", component: () => import("./views/DocsView.vue") },
    {
      path: "/maintenance",
      name: "maintenance",
      component: () => import("./views/MaintenanceView.vue"),
    },
    {
      path: "/settings",
      name: "settings",
      component: () => import("./views/SettingsView.vue"),
    },
    { path: "/runs/new", name: "run-new", component: RunFormView },
    { path: "/runs/jobs/:id/edit", name: "run-edit", component: RunFormView },
    { path: "/runs/jobs/:id/continue", name: "run-continue", component: RunFormView },
    { path: "/runs/jobs/:id", name: "job-detail", component: RunDetailView, props: { mode: "job" } },
    { path: "/runs", name: "jobs", component: () => import("./views/JobsView.vue") },
    {
      path: "/compare",
      name: "run-comparison",
      component: () => import("./views/RunComparisonView.vue"),
    },
    {
      path: "/runs/:name",
      name: "run-detail",
      component: RunDetailView,
      props: (route) => ({ mode: "fs", name: route.params.name }),
    },
    { path: "/jobs/:id", redirect: (to) => `/runs/jobs/${to.params.id}` },
    { path: "/jobs", redirect: "/runs" },
    {
      path: "/datasets",
      name: "datasets-list",
      component: () => import("./views/DatasetsListView.vue"),
    },
    { path: "/prep", name: "prep-jobs", component: () => import("./views/PrepJobsView.vue") },
    {
      path: "/prep/new/:stage",
      name: "prep-new",
      component: () => import("./views/PrepJobFormView.vue"),
    },
    { path: "/prep/tags", name: "prep-tags", component: () => import("./views/TagEditorView.vue") },
    { path: "/toolbox", name: "toolbox", component: () => import("./views/ToolboxView.vue") },
    {
      path: "/toolbox/new",
      name: "toolbox-new",
      component: () => import("./views/ToolboxToolFormView.vue"),
    },
    {
      path: "/toolbox/:id/edit",
      name: "toolbox-edit",
      component: () => import("./views/ToolboxToolFormView.vue"),
    },
  ],
});

export default router;
