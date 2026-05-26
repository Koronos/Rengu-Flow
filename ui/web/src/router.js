import { createRouter, createWebHistory } from "vue-router";
import ConfigsView from "./views/ConfigsView.vue";
import DatasetsView from "./views/DatasetsView.vue";
import JobsView from "./views/JobsView.vue";
import FsRunsView from "./views/FsRunsView.vue";
import RunDetailView from "./views/RunDetailView.vue";

const router = createRouter({
  history: createWebHistory(),
  routes: [
    { path: "/", redirect: "/configs" },
    { path: "/jobs", name: "jobs", component: JobsView },
    { path: "/jobs/:id", name: "job-detail", component: RunDetailView, props: { mode: "job" } },
    { path: "/configs", name: "configs", component: ConfigsView },
    { path: "/datasets", name: "datasets", component: DatasetsView },
    { path: "/runs", name: "runs", component: FsRunsView },
    {
      path: "/runs/:name",
      name: "run-detail",
      component: RunDetailView,
      props: (route) => ({ mode: "fs", name: route.params.name }),
    },
  ],
});

export default router;
