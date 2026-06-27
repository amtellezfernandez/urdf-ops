import { describe, expect, it } from "vitest";

import { filterExperimentJobs, getRunningExperimentJobs } from "./useExperimentStore";
import type { JobFilters, TrainingJob } from "./types";

const baseJob: TrainingJob = {
  id: "job-1",
  name: "Pick place",
  status: "completed",
  modelArchitecture: "act",
  datasetId: "lerobot/pusht",
  datasetSource: "huggingface",
  computeBackend: "local",
  startedAt: "2026-06-27T10:00:00.000Z",
};

const defaultFilters: JobFilters = {
  status: "all",
};

describe("experiment job helpers", () => {
  it("filters jobs by search query without mutating the source list", () => {
    const jobs: TrainingJob[] = [
      baseJob,
      {
        ...baseJob,
        id: "job-2",
        name: "Lift cube",
        datasetId: "/tmp/urdf-studio-teleop-replays/lift-cube",
        datasetSource: "local",
      },
    ];

    const filtered = filterExperimentJobs(jobs, {
      ...defaultFilters,
      searchQuery: "lift",
    });

    expect(filtered).toHaveLength(1);
    expect(filtered[0].id).toBe("job-2");
    expect(jobs).toHaveLength(2);
  });

  it("returns pending, queued, and running jobs as active", () => {
    const jobs: TrainingJob[] = [
      { ...baseJob, id: "job-pending", status: "pending" },
      { ...baseJob, id: "job-queued", status: "queued" },
      { ...baseJob, id: "job-running", status: "running" },
      { ...baseJob, id: "job-failed", status: "failed" },
    ];

    expect(getRunningExperimentJobs(jobs).map((job) => job.id)).toEqual([
      "job-pending",
      "job-queued",
      "job-running",
    ]);
  });
});
