/**
 * ExperimentDashboard - Main dashboard page for experiment views
 */

import { useEffect, useMemo } from "react";
import { FlaskConical, BarChart2, Database, Play } from "lucide-react";
import { useQuery } from "@tanstack/react-query";

import { Button } from "@/shared/ui/button";
import { cn } from "@/shared/lib/utils";

import { JobList } from "./JobList";
import { JobDetails } from "./JobDetails";
import { DatasetCatalog, type InitialExperimentDataset } from "./DatasetCatalog";
import { getRunningExperimentJobs, selectHasActiveJobs, useExperimentStore } from "./useExperimentStore";
import { TrainingDialog, useTrainingStore } from "@/features/training";
import { fetchTrainingJobs } from "./trainingJobsApi";
import { JOB_LIST_PARAMS } from "./jobListParams";
import {
  EXPERIMENT_DASHBOARD_CLASS_NAMES,
  EXPERIMENT_DASHBOARD_PARAMS,
} from "./experimentDashboardParams";

// ============================================================================
// Stats Card
// ============================================================================

interface StatsCardProps {
  title: string;
  value: string | number;
  subtitle?: string;
  icon: React.ReactNode;
  trend?: {
    value: number;
    isPositive: boolean;
  };
}

function StatsCard({ title, value, subtitle, icon, trend }: StatsCardProps) {
  return (
    <div className={EXPERIMENT_DASHBOARD_CLASS_NAMES.card}>
      <div className="flex items-start justify-between">
        <div>
          <p className="text-xs text-muted-foreground">{title}</p>
          <p className="mt-1 text-lg font-semibold">{value}</p>
          {subtitle && (
            <p className="text-xs text-muted-foreground mt-1">{subtitle}</p>
          )}
          {trend && (
            <p
              className={cn(
                "text-xs mt-1",
                trend.isPositive ? "text-green-600" : "text-red-600"
              )}
            >
              {trend.isPositive ? "+" : ""}{trend.value}% from last week
            </p>
          )}
        </div>
        <div className={EXPERIMENT_DASHBOARD_CLASS_NAMES.iconTile}>{icon}</div>
      </div>
    </div>
  );
}

// ============================================================================
// Overview Tab
// ============================================================================

function OverviewTab() {
  const jobs = useExperimentStore((state) => state.jobs);
  const total = useExperimentStore((state) => state.total);
  const runningJobs = useMemo(() => getRunningExperimentJobs(jobs), [jobs]);

  const completedJobs = jobs.filter((j) => j.status === "completed").length;
  const failedJobs = jobs.filter((j) => j.status === "failed").length;

  const successRate = total > 0
    ? Math.round((completedJobs / (completedJobs + failedJobs)) * 100) || 0
    : 0;

  return (
    <div className="space-y-5">
      {/* Stats Grid */}
      <div className={EXPERIMENT_DASHBOARD_CLASS_NAMES.statsGrid}>
        <StatsCard
          title="Total Jobs"
          value={total}
          icon={<FlaskConical className="h-4 w-4 text-muted-foreground" />}
        />
        <StatsCard
          title="Running"
          value={runningJobs.length}
          subtitle={runningJobs.length > 0 ? "Jobs in progress" : "No active jobs"}
          icon={<Play className="h-4 w-4 text-sky-500" />}
        />
        <StatsCard
          title="Completed"
          value={completedJobs}
          icon={<BarChart2 className="h-4 w-4 text-emerald-500" />}
        />
        <StatsCard
          title="Success Rate"
          value={`${successRate}%`}
          subtitle={`${failedJobs} failed`}
          icon={<Database className="h-4 w-4 text-muted-foreground" />}
        />
      </div>

      {/* Running Jobs Quick View */}
      {runningJobs.length > 0 && (
        <div className={EXPERIMENT_DASHBOARD_CLASS_NAMES.card}>
          <h3 className="mb-3 text-xs font-medium uppercase tracking-[0.08em] text-muted-foreground">Active Jobs</h3>
          <div className="space-y-3">
            {runningJobs.slice(0, EXPERIMENT_DASHBOARD_PARAMS.activeJobsPreviewLimit).map((job) => (
              <div key={job.id} className={EXPERIMENT_DASHBOARD_CLASS_NAMES.activeJobRow}>
                <div className="flex items-center gap-3">
                  <div className="h-2 w-2 rounded-full bg-sky-500 animate-pulse" />
                  <div>
                    <p className="text-sm font-medium">{job.name}</p>
                    <p className="text-xs text-muted-foreground">
                      {job.modelArchitecture.toUpperCase()} - {job.datasetId}
                    </p>
                  </div>
                </div>
                {job.progress && (
                  <div className="text-right">
                    <p className="text-sm font-medium">{Math.round(job.progress.overallProgress)}%</p>
                    <p className="text-xs text-muted-foreground">
                      Epoch {job.progress.currentEpoch}/{job.progress.totalEpochs}
                    </p>
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

// ============================================================================
// Jobs Tab with Split View
// ============================================================================

function JobsTab() {
  const selectedJobId = useExperimentStore((state) => state.selectedJobId);

  return (
    <div className={EXPERIMENT_DASHBOARD_CLASS_NAMES.splitView}>
      {/* Jobs List */}
      <div
        className={cn(
          EXPERIMENT_DASHBOARD_CLASS_NAMES.splitPanel,
          "transition-all",
          selectedJobId ? "w-1/2" : "w-full"
        )}
      >
        <JobList />
      </div>

      {/* Job Details */}
      {selectedJobId && (
        <div className={cn("w-1/2", EXPERIMENT_DASHBOARD_CLASS_NAMES.splitPanel)}>
          <JobDetails />
        </div>
      )}
    </div>
  );
}

// ============================================================================
// Main Component
// ============================================================================

interface ExperimentDashboardProps {
  activeView?: ExperimentDashboardView;
  initialDataset?: InitialExperimentDataset | null;
}

export type ExperimentDashboardView = "datasets" | "overview" | "jobs";

const VIEW_COPY: Record<ExperimentDashboardView, { title: string; subtitle: string }> = {
  datasets: {
    title: "Datasets",
    subtitle: "Studio LeRobot exports and training dataset sources",
  },
  overview: {
    title: "Overview",
    subtitle: "Monitor training activity and job health",
  },
  jobs: {
    title: "Jobs",
    subtitle: "Inspect, filter, and manage training jobs",
  },
};

export function ExperimentDashboard({
  activeView = "datasets",
  initialDataset = null,
}: ExperimentDashboardProps) {
  const reset = useExperimentStore((state) => state.reset);
  const hasActiveJobs = useExperimentStore(selectHasActiveJobs);
  const page = useExperimentStore((state) => state.page);
  const pageSize = useExperimentStore((state) => state.pageSize);
  const setJobs = useExperimentStore((state) => state.setJobs);
  const setIsLoading = useExperimentStore((state) => state.setIsLoading);
  const setError = useExperimentStore((state) => state.setError);
  const openTrainingDialog = useTrainingStore((state) => state.openDialog);
  const viewCopy = VIEW_COPY[activeView];
  const {
    data: jobsData,
    error: jobsError,
    isFetching: isJobsFetching,
  } = useQuery({
    queryKey: ["jobs", page, pageSize],
    queryFn: () => fetchTrainingJobs(page, pageSize),
    staleTime: JOB_LIST_PARAMS.queryStaleTimeMs,
  });

  useEffect(() => {
    if (jobsData) {
      setJobs(jobsData.jobs, jobsData.total);
    }
  }, [jobsData, setJobs]);

  useEffect(() => {
    setIsLoading(isJobsFetching);
  }, [isJobsFetching, setIsLoading]);

  useEffect(() => {
    if (!jobsError) {
      setError(null);
      return;
    }
    setError(jobsError instanceof Error ? jobsError.message : "Failed to fetch jobs");
  }, [jobsError, setError]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      // Don't reset if there are active jobs to preserve polling
      if (!hasActiveJobs) {
        reset();
      }
    };
  }, [hasActiveJobs, reset]);

  return (
    <div className={EXPERIMENT_DASHBOARD_CLASS_NAMES.shell}>
      {/* Header */}
      <div className={EXPERIMENT_DASHBOARD_CLASS_NAMES.header}>
        <div>
          <h1 className={EXPERIMENT_DASHBOARD_CLASS_NAMES.title}>{viewCopy.title}</h1>
          <p className={EXPERIMENT_DASHBOARD_CLASS_NAMES.subtitle}>
            {viewCopy.subtitle}
          </p>
        </div>
        <Button onClick={openTrainingDialog} className={EXPERIMENT_DASHBOARD_CLASS_NAMES.actionButton}>
          <Play className="mr-1.5 h-3.5 w-3.5" />
          New training
        </Button>
      </div>

      {/* Content */}
      <div
        className={cn(
          EXPERIMENT_DASHBOARD_CLASS_NAMES.content,
          activeView === "jobs" ? "overflow-hidden" : "overflow-auto",
        )}
      >
        {activeView === "datasets" && <DatasetCatalog initialDataset={initialDataset} />}
        {activeView === "overview" && <OverviewTab />}
        {activeView === "jobs" && <JobsTab />}
      </div>
      <TrainingDialog />
    </div>
  );
}
