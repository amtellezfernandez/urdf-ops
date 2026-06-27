/**
 * JobList component - Table of training jobs with filters
 */

import { useCallback, useEffect, useMemo } from "react";
import {
  Play,
  CheckCircle,
  XCircle,
  Clock,
  Loader2,
  Ban,
  Search,
  RefreshCw,
  ChevronLeft,
  ChevronRight,
} from "lucide-react";
import { useQuery } from "@tanstack/react-query";

import { Button } from "@/shared/ui/button";
import { Input } from "@/shared/ui/input";
import { Badge } from "@/shared/ui/badge";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/shared/ui/select";
import { cn } from "@/shared/lib/utils";

import { filterExperimentJobs, useExperimentStore } from "./useExperimentStore";
import type { TrainingJob, JobStatus, JobFilterStatus, JobsListResponse } from "./types";
import { fetchTrainingJobs } from "./trainingJobsApi";
import {
  JOB_LIST_CLASS_NAMES,
  JOB_LIST_PARAMS,
} from "./jobListParams";

// ============================================================================
// Status Helpers
// ============================================================================

const STATUS_CONFIG: Record<JobStatus, { icon: typeof Play; color: string; label: string }> = {
  pending: { icon: Clock, color: "border-amber-500/20 bg-amber-500/10 text-amber-600", label: "Pending" },
  queued: { icon: Clock, color: "border-amber-500/20 bg-amber-500/10 text-amber-600", label: "Queued" },
  running: { icon: Loader2, color: "border-sky-500/20 bg-sky-500/10 text-sky-600", label: "Running" },
  completed: { icon: CheckCircle, color: "border-emerald-500/20 bg-emerald-500/10 text-emerald-600", label: "Completed" },
  failed: { icon: XCircle, color: "border-red-500/20 bg-red-500/10 text-red-600", label: "Failed" },
  cancelled: { icon: Ban, color: "border-border/50 bg-muted/25 text-muted-foreground", label: "Cancelled" },
};

function StatusBadge({ status }: { status: JobStatus }) {
  const config = STATUS_CONFIG[status];
  const Icon = config.icon;
  const isAnimated = status === "running";

  return (
    <Badge className={cn("gap-1.5 font-medium", config.color)}>
      <Icon className={cn("h-3 w-3", isAnimated && "animate-spin")} />
      {config.label}
    </Badge>
  );
}

// ============================================================================
// Filter Bar
// ============================================================================

function FilterBar() {
  const { filters, setFilters, setStatusFilter, clearFilters } = useExperimentStore();

  const handleSearchChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      setFilters({ searchQuery: e.target.value || undefined });
    },
    [setFilters]
  );

  return (
    <div className={JOB_LIST_CLASS_NAMES.filterBar}>
      {/* Search */}
      <div className={JOB_LIST_CLASS_NAMES.searchField}>
        <Search className="absolute left-2.5 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-muted-foreground" />
        <Input
          placeholder="Search jobs..."
          value={filters.searchQuery || ""}
          onChange={handleSearchChange}
          className={JOB_LIST_CLASS_NAMES.searchInput}
        />
      </div>

      {/* Status filter */}
      <Select
        value={filters.status}
        onValueChange={(value) => setStatusFilter(value as JobFilterStatus)}
      >
        <SelectTrigger className={cn("w-[140px]", JOB_LIST_CLASS_NAMES.selectTrigger)}>
          <SelectValue placeholder="Status" />
        </SelectTrigger>
        <SelectContent>
          <SelectItem value="all">All Status</SelectItem>
          <SelectItem value="running">Running</SelectItem>
          <SelectItem value="completed">Completed</SelectItem>
          <SelectItem value="failed">Failed</SelectItem>
          <SelectItem value="pending">Pending</SelectItem>
          <SelectItem value="cancelled">Cancelled</SelectItem>
        </SelectContent>
      </Select>

      {/* Model filter */}
      <Select
        value={filters.modelArchitecture || "all"}
        onValueChange={(value) =>
          setFilters({ modelArchitecture: value === "all" ? undefined : value })
        }
      >
        <SelectTrigger className={cn("w-[160px]", JOB_LIST_CLASS_NAMES.selectTrigger)}>
          <SelectValue placeholder="Model" />
        </SelectTrigger>
        <SelectContent>
          <SelectItem value="all">All Models</SelectItem>
          <SelectItem value="act">ACT</SelectItem>
          <SelectItem value="diffusion_policy">Diffusion Policy</SelectItem>
          <SelectItem value="tdmpc">TD-MPC</SelectItem>
          <SelectItem value="vq_bet">VQ-BeT</SelectItem>
        </SelectContent>
      </Select>

      {/* Clear filters */}
      {(filters.status !== "all" || filters.modelArchitecture || filters.searchQuery) && (
        <Button variant="ghost" size="sm" className="h-7 text-xs" onClick={clearFilters}>
          Clear filters
        </Button>
      )}
    </div>
  );
}

// ============================================================================
// Job Row
// ============================================================================

interface JobRowProps {
  job: TrainingJob;
  isSelected: boolean;
  onSelect: (jobId: string) => void;
}

function JobRow({ job, isSelected, onSelect }: JobRowProps) {
  const formatDate = (dateStr: string) => {
    const date = new Date(dateStr);
    return date.toLocaleDateString() + " " + date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  };

  const formatDuration = (start: string, end?: string) => {
    const startDate = new Date(start);
    const endDate = end ? new Date(end) : new Date();
    const diffMs = endDate.getTime() - startDate.getTime();
    const diffMins = Math.floor(diffMs / JOB_LIST_PARAMS.millisecondsPerMinute);
    const diffHours = Math.floor(diffMins / JOB_LIST_PARAMS.minutesPerHour);

    if (diffHours > 0) {
      return `${diffHours}h ${diffMins % JOB_LIST_PARAMS.minutesPerHour}m`;
    }
    return `${diffMins}m`;
  };

  return (
    <tr
      className={cn(
        JOB_LIST_CLASS_NAMES.row,
        isSelected && JOB_LIST_CLASS_NAMES.rowSelected,
      )}
      onClick={() => onSelect(job.id)}
    >
      <td className={JOB_LIST_CLASS_NAMES.cell}>
        <div className="font-medium text-sm">{job.name}</div>
        <div className="text-xs text-muted-foreground font-mono">
          {job.id.slice(0, JOB_LIST_PARAMS.jobIdPreviewLength)}
        </div>
      </td>
      <td className={JOB_LIST_CLASS_NAMES.cell}>
        <StatusBadge status={job.status} />
      </td>
      <td className={JOB_LIST_CLASS_NAMES.cell}>
        <div className="text-xs">{job.modelArchitecture.toUpperCase()}</div>
      </td>
      <td className={JOB_LIST_CLASS_NAMES.cell}>
        <div className="text-sm truncate max-w-[200px]" title={job.datasetId}>
          {job.datasetId}
        </div>
        <div className="text-xs text-muted-foreground">{job.datasetSource}</div>
      </td>
      <td className={JOB_LIST_CLASS_NAMES.cell}>
        <div className="text-xs">{formatDate(job.startedAt)}</div>
        <div className="text-xs text-muted-foreground">
          {formatDuration(job.startedAt, job.finishedAt)}
        </div>
      </td>
      <td className={JOB_LIST_CLASS_NAMES.cell}>
        {job.progress && (
          <div className="w-full max-w-[100px]">
            <div className="flex justify-between text-xs mb-1">
              <span>{Math.round(job.progress.overallProgress)}%</span>
            </div>
            <div className={JOB_LIST_CLASS_NAMES.progressTrack}>
              <div
                className={JOB_LIST_CLASS_NAMES.progressIndicator}
                style={{ width: `${job.progress.overallProgress}%` }}
              />
            </div>
          </div>
        )}
        {!job.progress && job.status === "completed" && (
          <span className="text-sm text-muted-foreground">100%</span>
        )}
      </td>
    </tr>
  );
}

// ============================================================================
// Pagination
// ============================================================================

function Pagination() {
  const { page, pageSize, total, setPage } = useExperimentStore();
  const totalPages = Math.ceil(total / pageSize);

  if (totalPages <= 1) return null;

  return (
    <div className="flex items-center justify-between px-4 py-3 border-t">
      <div className="text-sm text-muted-foreground">
        Showing {(page - 1) * pageSize + 1} to {Math.min(page * pageSize, total)} of {total} jobs
      </div>
      <div className="flex items-center gap-2">
        <Button
          variant="outline"
          size="sm"
          onClick={() => setPage(page - 1)}
          disabled={page <= 1}
        >
          <ChevronLeft className="h-4 w-4" />
        </Button>
        <span className="text-sm">
          Page {page} of {totalPages}
        </span>
        <Button
          variant="outline"
          size="sm"
          onClick={() => setPage(page + 1)}
          disabled={page >= totalPages}
        >
          <ChevronRight className="h-4 w-4" />
        </Button>
      </div>
    </div>
  );
}

// ============================================================================
// Empty State
// ============================================================================

function EmptyState({ hasFilters }: { hasFilters: boolean }) {
  const { clearFilters } = useExperimentStore();

  return (
    <div className="flex flex-col items-center justify-center py-12 text-center">
      <div className={JOB_LIST_CLASS_NAMES.emptyIcon}>
        <Play className="h-5 w-5 text-muted-foreground" />
      </div>
      <h3 className="mb-2 text-base font-medium">
        {hasFilters ? "No matching jobs" : "No training jobs yet"}
      </h3>
      <p className="text-sm text-muted-foreground max-w-sm mb-4">
        {hasFilters
          ? "Try adjusting your filters to find what you're looking for."
          : "Start a new training job to see it appear here."}
      </p>
      {hasFilters && (
        <Button variant="outline" size="sm" onClick={clearFilters}>
          Clear filters
        </Button>
      )}
    </div>
  );
}

// ============================================================================
// Main Component
// ============================================================================

export function JobList() {
  const {
    jobs,
    isLoading,
    error,
    page,
    pageSize,
    filters,
    selectedJobId,
    setJobs,
    setIsLoading,
    setError,
    selectJob,
    isPolling,
    setIsPolling,
    setPollIntervalId,
    pollIntervalId,
  } = useExperimentStore();

  const filteredJobs = useMemo(() => filterExperimentJobs(jobs, filters), [jobs, filters]);
  const hasFilters =
    filters.status !== "all" ||
    !!filters.modelArchitecture ||
    !!filters.searchQuery;

  // Fetch jobs
  const { data, refetch, isFetching } = useQuery<JobsListResponse>({
    queryKey: ["jobs", page, pageSize],
    queryFn: () => fetchTrainingJobs(page, pageSize),
    staleTime: JOB_LIST_PARAMS.queryStaleTimeMs,
  });

  // Update store when data changes
  useEffect(() => {
    if (data) {
      setJobs(data.jobs, data.total);
    }
  }, [data, setJobs]);

  // Start polling for active jobs
  useEffect(() => {
    const hasActiveJobs = jobs.some(
      (j) => j.status === "running" || j.status === "pending" || j.status === "queued"
    );

    if (hasActiveJobs && !isPolling) {
      setIsPolling(true);
      const id = window.setInterval(() => {
        refetch();
      }, JOB_LIST_PARAMS.activeJobsPollIntervalMs);
      setPollIntervalId(id);
    } else if (!hasActiveJobs && isPolling) {
      if (pollIntervalId) {
        clearInterval(pollIntervalId);
        setPollIntervalId(null);
      }
      setIsPolling(false);
    }

    return () => {
      if (pollIntervalId) {
        clearInterval(pollIntervalId);
      }
    };
  }, [jobs, isPolling, pollIntervalId, refetch, setIsPolling, setPollIntervalId]);

  return (
    <div className={JOB_LIST_CLASS_NAMES.shell}>
      {/* Header with filters */}
      <div className={JOB_LIST_CLASS_NAMES.header}>
        <div className={JOB_LIST_CLASS_NAMES.headerRow}>
          <h2 className={JOB_LIST_CLASS_NAMES.title}>Training Jobs</h2>
          <Button
            variant="outline"
            size="sm"
            className={JOB_LIST_CLASS_NAMES.refreshButton}
            onClick={() => refetch()}
            disabled={isFetching}
          >
            <RefreshCw className={cn("mr-1.5 h-3.5 w-3.5", isFetching && "animate-spin")} />
            Refresh
          </Button>
        </div>
        <FilterBar />
      </div>

      {/* Error state */}
      {error && (
        <div className="mx-4 mb-4 p-3 bg-destructive/10 border border-destructive/20 rounded text-sm text-destructive">
          {error}
        </div>
      )}

      {/* Loading state */}
      {isLoading && !jobs.length && (
        <div className="flex items-center justify-center py-12">
          <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
        </div>
      )}

      {/* Empty state */}
      {!isLoading && filteredJobs.length === 0 && <EmptyState hasFilters={hasFilters} />}

      {/* Jobs table */}
      {filteredJobs.length > 0 && (
        <div className="flex-1 overflow-auto">
          <table className="w-full">
            <thead className={JOB_LIST_CLASS_NAMES.tableHeader}>
              <tr className={JOB_LIST_CLASS_NAMES.tableHeaderRow}>
                <th className={JOB_LIST_CLASS_NAMES.tableHeaderCell}>Job</th>
                <th className={JOB_LIST_CLASS_NAMES.tableHeaderCell}>Status</th>
                <th className={JOB_LIST_CLASS_NAMES.tableHeaderCell}>Model</th>
                <th className={JOB_LIST_CLASS_NAMES.tableHeaderCell}>Dataset</th>
                <th className={JOB_LIST_CLASS_NAMES.tableHeaderCell}>Started</th>
                <th className={JOB_LIST_CLASS_NAMES.tableHeaderCell}>Progress</th>
              </tr>
            </thead>
            <tbody>
              {filteredJobs.map((job) => (
                <JobRow
                  key={job.id}
                  job={job}
                  isSelected={selectedJobId === job.id}
                  onSelect={selectJob}
                />
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* Pagination */}
      <Pagination />
    </div>
  );
}
