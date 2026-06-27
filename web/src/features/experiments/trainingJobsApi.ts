import { API_BASE_URL } from "@/shared/config/api";

import type { JobsListResponse, JobStatus, TrainingJob } from "./types";
import { JOB_LIST_PARAMS } from "./jobListParams";

type TrainingJobSummaryPayload = {
  jobId: string;
  status: JobStatus;
  runName?: string | null;
  modelArchitecture?: string | null;
  datasetId?: string | null;
  startedAt?: string | null;
  finishedAt?: string | null;
  computeBackend?: string | null;
};

type TrainingJobsListPayload = {
  jobs: TrainingJobSummaryPayload[];
  total: number;
};

function inferDatasetSource(datasetId: string): TrainingJob["datasetSource"] {
  return datasetId.startsWith("/") || datasetId.startsWith(".") ? "local" : "huggingface";
}

export function mapTrainingJobSummary(summary: TrainingJobSummaryPayload): TrainingJob {
  const modelArchitecture = summary.modelArchitecture || "unknown";
  const datasetId = summary.datasetId || "unknown";
  return {
    id: summary.jobId,
    name: summary.runName || `${modelArchitecture} ${summary.jobId.slice(0, JOB_LIST_PARAMS.jobIdPreviewLength)}`,
    status: summary.status,
    modelArchitecture,
    datasetId,
    datasetSource: inferDatasetSource(datasetId),
    computeBackend: summary.computeBackend || "local",
    startedAt: summary.startedAt || new Date(0).toISOString(),
    finishedAt: summary.finishedAt || undefined,
  };
}

export async function fetchTrainingJobs(page: number, pageSize: number): Promise<JobsListResponse> {
  const response = await fetch(`${API_BASE_URL}/training/jobs?limit=${pageSize}`);
  if (!response.ok) throw new Error("Failed to fetch jobs");
  const payload: TrainingJobsListPayload = await response.json();
  return {
    jobs: payload.jobs.map(mapTrainingJobSummary),
    total: payload.total,
    page,
    pageSize,
  };
}
