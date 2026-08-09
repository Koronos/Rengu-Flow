<!--
  The `train` node's config: a single field, `job_id`.

  **This is a picker, not an editor.** The node fires a run that is already registered — one the
  user configured on the Runs page, pointing at a dataset TOML that already names the folder the
  workflow is treating. No config is synthesized, no TOML rewritten, no path injected: every prep
  stage writes in place, so the folder the chain prepared is the folder the run already consumes.

  Only a `new` draft or a `pending` run can be fired (`workflow_nodes._run_train`): a draft is
  promoted to pending first, then the run is put at the front of the queue and the queue is
  started. Anything already running or finished is refused, so those rows are offered disabled
  rather than silently failing at execution time.

  And the outcome is deliberately understated: the step is `done` when the run is **queued**, which
  is not the same thing as trained. That is why the result reads `Queued run #123 ->` with a link,
  never a bare green check.
-->
<template>
  <div class="train-node-form">
    <el-alert
      type="info"
      :closable="false"
      show-icon
      class="train-node-form__banner"
      title="This step queues a run you already registered and starts the queue. It is done as soon as the run is queued — training itself is watched on the run's own page."
    />

    <el-form label-position="top" :disabled="disabled">
      <el-form-item label="Run" required>
        <el-select
          v-model="jobId"
          filterable
          :loading="loading"
          placeholder="Pick a saved or queued run"
          class="w-full"
        >
          <el-option
            v-for="job in jobs"
            :key="job.id"
            :label="optionLabel(job)"
            :value="job.id"
            :disabled="!isEligible(job)"
          >
            <span class="opt-title">{{ optionLabel(job) }}</span>
            <span class="opt-note">{{ isEligible(job) ? job.state : `${job.state} — cannot be fired` }}</span>
          </el-option>
        </el-select>
        <el-text v-if="loadError" size="small" type="danger" class="hint-text">
          {{ loadError }}
        </el-text>
        <el-text v-else-if="!loading && !eligibleCount" size="small" type="warning" class="hint-text">
          No saved or queued run to fire. Register one on the Runs page — a run that is already
          training or finished cannot be re-fired from here.
        </el-text>
        <el-text v-else size="small" type="info" class="hint-text">
          Saved drafts and queued runs only. The run goes to the front of the queue; if something
          is already training it waits its turn rather than preempting it.
        </el-text>
      </el-form-item>
    </el-form>

    <el-card v-if="queuedJobId != null" shadow="never" class="train-node-form__queued">
      <router-link :to="`/runs/jobs/${queuedJobId}`" class="train-node-form__link">
        Queued run #{{ queuedJobId }} &rarr;
      </router-link>
      <el-text size="small" type="info" class="hint-text">
        Queued, not trained. Open the run to see whether it actually started and how it is going.
      </el-text>
    </el-card>
  </div>
</template>

<script setup lang="ts">
import { computed, onMounted, ref } from "vue";
import type { PropType } from "vue";
import { api } from "../../../api";
import { formatError } from "../../../lib/formatError";
import type { JobRecord } from "../../../types/api";

const config = defineModel<Record<string, unknown>>({ required: true });

defineProps({
  /** `state.nodes[id].result.job_id` — the run this node actually queued, once it has run. */
  queuedJobId: { type: [Number, String] as PropType<number | string | null>, default: null },
  /** Read-only: the runner owns the workflow while it runs. */
  disabled: { type: Boolean, default: false },
});

/** The only two states `_run_train` accepts; everything else is refused at execution. */
const FIREABLE_STATES = ["new", "pending"];

const jobs = ref<JobRecord[]>([]);
const loading = ref(false);
const loadError = ref("");

const jobId = computed<string | number | null>({
  get: () => {
    const raw = config.value.job_id;
    return typeof raw === "string" || typeof raw === "number" ? raw : null;
  },
  set: (value) => {
    config.value = { ...config.value, job_id: value };
  },
});

function isEligible(job: JobRecord): boolean {
  return FIREABLE_STATES.includes(String(job.state));
}

function optionLabel(job: JobRecord): string {
  return `#${job.id} — ${job.run_name || "unnamed run"}`;
}

const eligibleCount = computed(() => jobs.value.filter(isEligible).length);

onMounted(async () => {
  loading.value = true;
  try {
    const result = await api.listJobs();
    jobs.value = result.jobs ?? [];
  } catch (e) {
    loadError.value = formatError(e);
  } finally {
    loading.value = false;
  }
});
</script>

<style scoped>
.train-node-form__banner {
  margin-bottom: 12px;
}
.train-node-form__queued {
  margin-top: 12px;
}
.train-node-form__link {
  font-weight: 600;
  color: var(--el-color-primary);
  text-decoration: none;
}
.train-node-form__link:hover {
  text-decoration: underline;
}
.hint-text {
  display: block;
  margin-top: 4px;
  line-height: 1.45;
}
.opt-title {
  margin-right: 12px;
}
.opt-note {
  float: right;
  color: var(--el-text-color-secondary);
  font-size: 12px;
}
</style>
