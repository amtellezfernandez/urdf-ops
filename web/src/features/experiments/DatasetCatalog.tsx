import { FormEvent, useEffect, useMemo, useState, type ReactNode } from "react";
import {
  CheckCircle2,
  Database,
  ExternalLink,
  FolderOpen,
  GitBranchPlus,
  Layers3,
  Loader2,
  Play,
  Plus,
  RefreshCw,
  Search,
} from "lucide-react";
import { useQuery } from "@tanstack/react-query";
import { toast } from "sonner";

import { Button } from "@/shared/ui/button";
import { Input } from "@/shared/ui/input";
import { Badge } from "@/shared/ui/badge";
import { API_BASE_URL } from "@/shared/config/api";
import { cn } from "@/shared/lib/utils";
import { useTrainingStore } from "@/features/training";
import type { DatasetConfig, DatasetSource } from "@/features/training/types";

import { EXPERIMENT_DASHBOARD_CLASS_NAMES } from "./experimentDashboardParams";

type DatasetCatalogSource = "studio_export" | "local";
type TrainingSetSourceKind = "local" | "huggingface";
type MixStatus = "idle" | "queued" | "running" | "succeeded" | "failed" | "rejected";

interface DatasetCatalogItem {
  id: string;
  name: string;
  source: DatasetCatalogSource;
  path: string;
  format_version?: string | null;
  robot_type?: string | null;
  total_episodes?: number | null;
  total_frames?: number | null;
  fps?: number | null;
  export_mode?: string | null;
  recording_id?: string | null;
  created_at?: string | null;
}

interface DatasetCatalogResponse {
  datasets: DatasetCatalogItem[];
  roots: string[];
}

interface DatasetMixResponse {
  jobId?: string;
  job_id?: string;
  status: "rejected" | "queued" | "running" | "succeeded" | "failed";
  success: boolean;
  message?: string | null;
  outputPath?: string | null;
  output_path?: string | null;
  error?: string | null;
}

interface TrainingSetSource {
  id: string;
  kind: TrainingSetSourceKind;
  name: string;
  value: string;
  sourceLabel: string;
  robot?: string | null;
  format?: string | null;
  totalEpisodes?: number | null;
  totalFrames?: number | null;
  updatedAt?: string | null;
  selectedEpisodes?: number[];
}

export interface InitialExperimentDataset {
  id: string;
  name: string;
  source: DatasetSource;
  author?: string;
}

const HF_DATASETS_STORAGE_KEY = "urdf-ops:hf-datasets";
const EMPTY_DATASETS: DatasetCatalogItem[] = [];
const MIX_REQUIRED_REPRESENTATION_ID = "rep:joint_pos_abs:semantic:v1";
const MIX_POLL_INTERVAL_MS = 1_000;
const MIX_MAX_POLLS = 180;

const numberFormatter = new Intl.NumberFormat();

function normalizeHfDatasetId(value: string): string {
  return value
    .trim()
    .replace(/^https:\/\/huggingface\.co\/datasets\//i, "")
    .split(/[?#]/)[0]
    .replace(/^\/+|\/+$/g, "");
}

function normalizeHfDatasetIds(values: string[]): string[] {
  return Array.from(
    new Set(values.map(normalizeHfDatasetId).filter((value) => value.length > 0)),
  ).sort((a, b) => a.localeCompare(b));
}

function readStoredHfDatasetIds(): string[] {
  if (typeof window === "undefined") return [];
  try {
    const raw = window.localStorage.getItem(HF_DATASETS_STORAGE_KEY);
    const parsed = raw ? JSON.parse(raw) : [];
    return normalizeHfDatasetIds(Array.isArray(parsed) ? parsed.filter((item) => typeof item === "string") : []);
  } catch {
    return [];
  }
}

function formatCount(value: number | null | undefined, label: string): string {
  if (value == null) return `- ${label}s`;
  return `${numberFormatter.format(value)} ${label}${value === 1 ? "" : "s"}`;
}

function formatDate(value: string | null | undefined): string {
  if (!value) return "-";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "-";
  return date.toLocaleString([], {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

function localSourceFromDataset(dataset: DatasetCatalogItem): TrainingSetSource {
  return {
    id: `local:${dataset.path}`,
    kind: "local",
    name: dataset.name,
    value: dataset.path,
    sourceLabel: dataset.source === "studio_export" ? "Studio export" : "Local",
    robot: dataset.robot_type,
    format: dataset.format_version || "LeRobot",
    totalEpisodes: dataset.total_episodes,
    totalFrames: dataset.total_frames,
    updatedAt: dataset.created_at,
  };
}

function hfSourceFromRepo(repoId: string): TrainingSetSource {
  return {
    id: `hf:${repoId}`,
    kind: "huggingface",
    name: repoId,
    value: repoId,
    sourceLabel: "Hugging Face",
    format: "LeRobot",
  };
}

function configFromSource(source: TrainingSetSource): DatasetConfig {
  const episodes = source.selectedEpisodes;
  if (source.kind === "local") {
    return {
      source: "local",
      localPath: source.value,
      episodes,
    };
  }
  return {
    source: "huggingface",
    repoId: source.value,
    episodes,
  };
}

function selectedEpisodeLabel(source: TrainingSetSource): string {
  if (source.selectedEpisodes === undefined) {
    return source.totalEpisodes != null ? `All ${source.totalEpisodes}` : "All";
  }
  if (source.selectedEpisodes.length === 0) return "None";
  return `${source.selectedEpisodes.length} selected`;
}

function sortedEpisodes(episodes: number[]): number[] {
  return Array.from(new Set(episodes)).sort((a, b) => a - b);
}

function mixResponseJobId(response: DatasetMixResponse): string | undefined {
  return response.jobId ?? response.job_id;
}

function mixResponseOutputPath(response: DatasetMixResponse): string | undefined {
  return response.outputPath ?? response.output_path ?? undefined;
}

function buildMixPayload(sources: TrainingSetSource[]) {
  const repoSources = sources.filter((source) => source.kind === "huggingface");
  const localSources = sources.filter((source) => source.kind === "local");
  const orderedSources = [...repoSources, ...localSources];
  return {
    repo_ids: repoSources.map((source) => source.value),
    local_paths: localSources.map((source) => source.value),
    episode_filters: [
      ...repoSources.flatMap((source, sourceIndex) =>
        source.selectedEpisodes
          ? [{
              source_kind: "repo",
              source_index: sourceIndex,
              episodes: source.selectedEpisodes,
            }]
          : [],
      ),
      ...localSources.flatMap((source, sourceIndex) =>
        source.selectedEpisodes
          ? [{
              source_kind: "local",
              source_index: sourceIndex,
              episodes: source.selectedEpisodes,
            }]
          : [],
      ),
    ],
    alignment: {
      required_representation_id: MIX_REQUIRED_REPRESENTATION_ID,
      datasets: orderedSources.map((source) => ({
        dataset_id: source.kind === "huggingface" ? `hf:${source.value}` : `local:${source.name}`,
        representation_id: MIX_REQUIRED_REPRESENTATION_ID,
        naming_status: "named",
      })),
    },
  };
}

interface SourceRowProps {
  source: TrainingSetSource;
  selected: boolean;
  onToggle: () => void;
  onUseWholeSource: () => void;
  onToggleEpisode: (episodeIndex: number) => void;
}

function SourceRow({
  source,
  selected,
  onToggle,
  onUseWholeSource,
  onToggleEpisode,
}: SourceRowProps) {
  const episodeCount = Math.max(0, source.totalEpisodes ?? 0);
  const selectedSet = new Set(source.selectedEpisodes ?? []);
  const usingWholeSource = source.selectedEpisodes === undefined;
  const episodeIndexes = Array.from({ length: episodeCount }, (_, index) => index);

  return (
    <div className={cn("border-b border-border/50 last:border-b-0", selected && "bg-primary/5")}>
      <div className="grid gap-3 px-3 py-3 md:grid-cols-[36px_minmax(0,1.4fr)_130px_130px_120px_110px] md:items-center">
        <div>
          <input
            aria-label={`Use ${source.name}`}
            type="checkbox"
            checked={selected}
            onChange={onToggle}
            className="h-4 w-4 accent-primary"
          />
        </div>
        <div className="min-w-0">
          <div className="flex flex-wrap items-center gap-2">
            <p className="truncate text-sm font-medium" title={source.name}>
              {source.name}
            </p>
            <Badge className="border-border/60 bg-muted/30 text-[11px] font-normal text-muted-foreground">
              {source.sourceLabel}
            </Badge>
            {selected && (
              <Badge className="border-emerald-500/30 bg-emerald-500/10 text-[11px] font-normal text-emerald-600">
                In training set
              </Badge>
            )}
          </div>
          <p className="mt-1 truncate font-mono text-xs text-muted-foreground" title={source.value}>
            {source.value}
          </p>
        </div>
        <div className="text-xs">
          <div className="font-medium text-foreground">{source.robot || "-"}</div>
          <div className="text-muted-foreground">{source.format || "LeRobot"}</div>
        </div>
        <div className="text-xs text-muted-foreground">
          <div>{formatCount(source.totalEpisodes, "episode")}</div>
          <div>{formatCount(source.totalFrames, "frame")}</div>
        </div>
        <div className="text-xs text-muted-foreground">{formatDate(source.updatedAt)}</div>
        <div className="flex items-center gap-2 md:justify-end">
          <Badge className="border-border/60 bg-background text-[11px] font-normal text-foreground">
            {selectedEpisodeLabel(source)}
          </Badge>
        </div>
      </div>

      {selected && (
        <div className="border-t border-border/40 bg-background/60 px-3 py-3 md:ml-9">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <div>
              <p className="text-xs font-medium">Episode scope</p>
              <p className="text-xs text-muted-foreground">
                {episodeCount > 0
                  ? "Use the whole source or pin explicit episodes."
                  : "Episode metadata is unavailable; this source is used whole."}
              </p>
            </div>
            <Button
              type="button"
              variant={usingWholeSource ? "secondary" : "outline"}
              size="sm"
              className="h-7 text-xs"
              onClick={onUseWholeSource}
            >
              Whole source
            </Button>
          </div>
          {episodeCount > 0 && (
            <div className="mt-2 flex max-h-28 flex-wrap gap-1 overflow-auto pr-1">
              {episodeIndexes.map((episodeIndex) => {
                const episodeSelected = usingWholeSource || selectedSet.has(episodeIndex);
                return (
                  <button
                    key={episodeIndex}
                    type="button"
                    onClick={() => onToggleEpisode(episodeIndex)}
                    className={cn(
                      "h-7 min-w-9 rounded-md border px-2 text-xs transition-colors",
                      episodeSelected
                        ? "border-foreground bg-foreground text-background"
                        : "border-border/70 bg-background text-muted-foreground hover:border-foreground/50 hover:text-foreground",
                    )}
                  >
                    {episodeIndex}
                  </button>
                );
              })}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

interface SourceTableProps {
  title: string;
  subtitle: string;
  icon: ReactNode;
  sources: TrainingSetSource[];
  selectedSources: TrainingSetSource[];
  query: string;
  emptyTitle: string;
  emptySubtitle: string;
  onToggleSource: (source: TrainingSetSource) => void;
  onUseWholeSource: (source: TrainingSetSource) => void;
  onToggleEpisode: (source: TrainingSetSource, episodeIndex: number) => void;
}

function SourceTable({
  title,
  subtitle,
  icon,
  sources,
  selectedSources,
  query,
  emptyTitle,
  emptySubtitle,
  onToggleSource,
  onUseWholeSource,
  onToggleEpisode,
}: SourceTableProps) {
  const selectedById = new Map(selectedSources.map((source) => [source.id, source]));
  const renderedSources = sources
    .map((source) => selectedById.get(source.id) ?? source)
    .filter((source) => {
      if (!query) return true;
      const haystack = `${source.name} ${source.value} ${source.robot ?? ""} ${source.sourceLabel}`.toLowerCase();
      return haystack.includes(query.toLowerCase());
    });

  return (
    <section className="overflow-hidden rounded-md border border-border/60 bg-background/95 shadow-sm">
      <div className="flex items-center justify-between border-b border-border/60 px-3 py-3">
        <div>
          <h2 className="text-sm font-medium">{title}</h2>
          <p className="text-xs text-muted-foreground">{subtitle}</p>
        </div>
        <div className={EXPERIMENT_DASHBOARD_CLASS_NAMES.iconTile}>{icon}</div>
      </div>

      {renderedSources.length > 0 ? (
        <div>
          <div className="hidden grid-cols-[36px_minmax(0,1.4fr)_130px_130px_120px_110px] border-b border-border/60 bg-muted/20 px-3 py-2 text-xs text-muted-foreground md:grid">
            <div />
            <div>Source</div>
            <div>Robot</div>
            <div>Size</div>
            <div>Updated</div>
            <div className="text-right">Scope</div>
          </div>
          {renderedSources.map((source) => (
            <SourceRow
              key={source.id}
              source={source}
              selected={selectedById.has(source.id)}
              onToggle={() => onToggleSource(source)}
              onUseWholeSource={() => onUseWholeSource(source)}
              onToggleEpisode={(episodeIndex) => onToggleEpisode(source, episodeIndex)}
            />
          ))}
        </div>
      ) : (
        <div className="px-3 py-8 text-center">
          <p className="text-sm font-medium">{emptyTitle}</p>
          <p className="mx-auto mt-2 max-w-md text-xs text-muted-foreground">{emptySubtitle}</p>
        </div>
      )}
    </section>
  );
}

interface DatasetCatalogProps {
  initialDataset?: InitialExperimentDataset | null;
}

export function DatasetCatalog({ initialDataset }: DatasetCatalogProps) {
  const [hfDatasetId, setHfDatasetId] = useState("");
  const [hfDatasetIds, setHfDatasetIds] = useState<string[]>(readStoredHfDatasetIds);
  const [selectedSources, setSelectedSources] = useState<TrainingSetSource[]>([]);
  const [materializedDataset, setMaterializedDataset] = useState<{ path: string; jobId: string } | null>(null);
  const [mixStatus, setMixStatus] = useState<{ status: MixStatus; message?: string; jobId?: string }>({ status: "idle" });
  const [query, setQuery] = useState("");
  const setDatasetConfig = useTrainingStore((state) => state.setDatasetConfig);
  const openTrainingDialog = useTrainingStore((state) => state.openDialog);
  const setTrainingStep = useTrainingStore((state) => state.setStep);

  const { data, error, isFetching, isLoading, refetch } = useQuery<DatasetCatalogResponse>({
    queryKey: ["datasets", "catalog"],
    queryFn: async () => {
      const response = await fetch(`${API_BASE_URL}/datasets/catalog`);
      if (!response.ok) {
        throw new Error(`GET ${API_BASE_URL}/datasets/catalog returned ${response.status}`);
      }
      return response.json();
    },
    staleTime: 5_000,
    retry: false,
  });

  const datasets = data?.datasets ?? EMPTY_DATASETS;
  const localSources = useMemo(
    () => datasets.map(localSourceFromDataset),
    [datasets],
  );
  const hfSources = useMemo(
    () => hfDatasetIds.map(hfSourceFromRepo),
    [hfDatasetIds],
  );
  const studioExportCount = datasets.filter((dataset) => dataset.source === "studio_export").length;
  const selectedSourceCount = selectedSources.length;
  const selectedEpisodeSubsetCount = selectedSources.filter((source) => source.selectedEpisodes !== undefined).length;
  const hasEmptySubset = selectedSources.some((source) => source.selectedEpisodes?.length === 0);
  const canUseDirectSelection = selectedSources.length === 1 && !hasEmptySubset;
  const canMaterializeMix = selectedSources.length > 1 && !hasEmptySubset;
  const canConfigureTraining = canUseDirectSelection || Boolean(materializedDataset);
  const catalogErrorMessage = error instanceof Error ? error.message : "Dataset catalog is unavailable";
  const catalogErrorHint =
    catalogErrorMessage.includes(":8000/") && catalogErrorMessage.includes("returned 404")
      ? "This UI is pointed at the Studio backend on port 8000. Start URDF Ops on port 8001."
      : "Restart the URDF Ops backend to enable local Studio export discovery.";
  const normalizedHfDatasetId = normalizeHfDatasetId(hfDatasetId);

  useEffect(() => {
    if (typeof window === "undefined") return;
    window.localStorage.setItem(HF_DATASETS_STORAGE_KEY, JSON.stringify(hfDatasetIds));
  }, [hfDatasetIds]);

  useEffect(() => {
    if (!initialDataset) return;
    const source =
      initialDataset.source === "local"
        ? {
            id: `local:${initialDataset.id}`,
            kind: "local" as const,
            name: initialDataset.name,
            value: initialDataset.id,
            sourceLabel: "Opened from Studio",
            format: "LeRobot",
          }
        : hfSourceFromRepo(normalizeHfDatasetId(initialDataset.id));
    if (source.kind === "huggingface") {
      setHfDatasetIds((current) => normalizeHfDatasetIds([...current, source.value]));
    }
    setSelectedSources((current) => (current.some((item) => item.id === source.id) ? current : [source]));
  }, [initialDataset]);

  useEffect(() => {
    if (materializedDataset) {
      setDatasetConfig({
        source: "local",
        localPath: materializedDataset.path,
      });
      return;
    }
    if (selectedSources.length === 1 && !hasEmptySubset) {
      setDatasetConfig(configFromSource(selectedSources[0]));
      return;
    }
    setDatasetConfig(null);
  }, [hasEmptySubset, materializedDataset, selectedSources, setDatasetConfig]);

  const resetMaterializedMix = () => {
    setMaterializedDataset(null);
    setMixStatus({ status: "idle" });
  };

  const updateSelectedSource = (source: TrainingSetSource, updater: (current: TrainingSetSource) => TrainingSetSource) => {
    resetMaterializedMix();
    setSelectedSources((current) => {
      const index = current.findIndex((item) => item.id === source.id);
      if (index === -1) {
        return [...current, updater(source)];
      }
      const next = [...current];
      next[index] = updater(next[index]);
      return next;
    });
  };

  const toggleSource = (source: TrainingSetSource) => {
    resetMaterializedMix();
    setSelectedSources((current) =>
      current.some((item) => item.id === source.id)
        ? current.filter((item) => item.id !== source.id)
        : [...current, source],
    );
  };

  const useWholeSource = (source: TrainingSetSource) => {
    updateSelectedSource(source, (current) => ({
      ...current,
      selectedEpisodes: undefined,
    }));
  };

  const toggleEpisode = (source: TrainingSetSource, episodeIndex: number) => {
    updateSelectedSource(source, (current) => {
      const currentEpisodes = current.selectedEpisodes ?? [];
      const selected = new Set(currentEpisodes);
      if (current.selectedEpisodes !== undefined && selected.has(episodeIndex)) {
        selected.delete(episodeIndex);
      } else {
        selected.add(episodeIndex);
      }
      return {
        ...current,
        selectedEpisodes: sortedEpisodes(Array.from(selected)),
      };
    });
  };

  const configureTraining = () => {
    if (!canConfigureTraining) return;
    openTrainingDialog();
    setTrainingStep("model");
  };

  const wait = (delayMs: number) => new Promise((resolve) => window.setTimeout(resolve, delayMs));

  const pollMixJob = async (jobId: string): Promise<DatasetMixResponse> => {
    for (let attempt = 0; attempt < MIX_MAX_POLLS; attempt += 1) {
      await wait(MIX_POLL_INTERVAL_MS);
      const response = await fetch(`${API_BASE_URL}/datasets/mix/${jobId}`);
      if (!response.ok) {
        throw new Error(`GET /datasets/mix/${jobId} returned ${response.status}`);
      }
      const payload: DatasetMixResponse = await response.json();
      setMixStatus({
        status: payload.status === "queued" ? "queued" : payload.status === "running" ? "running" : payload.status,
        message: payload.message ?? payload.error ?? undefined,
        jobId,
      });
      if (payload.status === "succeeded" || payload.status === "failed" || payload.status === "rejected") {
        return payload;
      }
    }
    throw new Error("Dataset mix timed out before completion.");
  };

  const materializeMix = async () => {
    if (!canMaterializeMix) return;
    try {
      setMixStatus({ status: "queued", message: "Submitting dataset mix..." });
      const response = await fetch(`${API_BASE_URL}/datasets/mix`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(buildMixPayload(selectedSources)),
      });
      if (!response.ok) {
        const payload = await response.json().catch(() => null);
        throw new Error(payload?.detail || `POST /datasets/mix returned ${response.status}`);
      }
      const submitted: DatasetMixResponse = await response.json();
      const jobId = mixResponseJobId(submitted);
      if (submitted.status === "rejected" || submitted.status === "failed") {
        throw new Error(submitted.error || submitted.message || "Dataset mix was rejected.");
      }
      const completed =
        submitted.status === "succeeded"
          ? submitted
          : jobId
            ? await pollMixJob(jobId)
            : submitted;
      if (completed.status !== "succeeded") {
        throw new Error(completed.error || completed.message || "Dataset mix failed.");
      }
      const outputPath = mixResponseOutputPath(completed);
      if (!outputPath) {
        throw new Error("Dataset mix succeeded without an output path.");
      }
      setMaterializedDataset({ path: outputPath, jobId: jobId ?? "dataset-mix" });
      setMixStatus({
        status: "succeeded",
        message: "Mixed dataset is ready for training.",
        jobId,
      });
      toast.success("Mixed dataset materialized");
    } catch (mixError) {
      const message = mixError instanceof Error ? mixError.message : "Dataset mix failed.";
      setMixStatus({ status: "failed", message });
      toast.error(message);
    }
  };

  const handleUseHuggingFace = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (!normalizedHfDatasetId) return;
    setHfDatasetIds((current) => normalizeHfDatasetIds([...current, normalizedHfDatasetId]));
    setSelectedSources((current) => {
      const source = hfSourceFromRepo(normalizedHfDatasetId);
      return current.some((item) => item.id === source.id) ? current : [...current, source];
    });
    setHfDatasetId("");
    resetMaterializedMix();
  };

  return (
    <div className="grid min-h-full gap-4 lg:grid-cols-[340px_minmax(0,1fr)]">
      <aside className="space-y-3">
        <section className="rounded-md border border-border/60 bg-background/95 shadow-sm">
          <div className="border-b border-border/60 px-3 py-3">
            <div className="flex items-center gap-2">
              <div className={EXPERIMENT_DASHBOARD_CLASS_NAMES.iconTile}>
                <Layers3 className="h-4 w-4 text-muted-foreground" />
              </div>
              <div>
                <h2 className="text-sm font-medium">Training set</h2>
                <p className="text-xs text-muted-foreground">
                  {selectedSourceCount} sources, {selectedEpisodeSubsetCount} scoped
                </p>
              </div>
            </div>
          </div>
          <div className="space-y-3 p-3">
            {selectedSources.length > 0 ? (
              <div className="space-y-2">
                {selectedSources.map((source) => (
                  <div key={source.id} className="rounded-md border border-border/60 bg-muted/15 p-2">
                    <div className="flex items-start justify-between gap-2">
                      <div className="min-w-0">
                        <p className="truncate text-sm font-medium" title={source.name}>
                          {source.name}
                        </p>
                        <p className="truncate font-mono text-[11px] text-muted-foreground" title={source.value}>
                          {source.value}
                        </p>
                      </div>
                      <Badge className="border-border/60 bg-background text-[11px] font-normal">
                        {selectedEpisodeLabel(source)}
                      </Badge>
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <p className="text-sm text-muted-foreground">Choose one or more sources from the tables.</p>
            )}

            {hasEmptySubset && (
              <p className="text-xs text-destructive">
                A source has zero episodes selected. Use the whole source or select at least one episode.
              </p>
            )}

            {selectedSources.length > 1 && (
              <div className="rounded-md border border-border/60 bg-muted/10 p-2">
                <div className="flex items-start gap-2">
                  <GitBranchPlus className="mt-0.5 h-4 w-4 text-muted-foreground" />
                  <div>
                    <p className="text-xs font-medium">Mixed training set</p>
                    <p className="text-xs text-muted-foreground">
                      Materialize the selected sources into one local LeRobot dataset before training.
                    </p>
                    {mixStatus.status !== "idle" && (
                      <p className={cn("mt-1 text-xs", mixStatus.status === "failed" ? "text-destructive" : "text-muted-foreground")}>
                        {mixStatus.message || mixStatus.status}
                      </p>
                    )}
                  </div>
                </div>
                <Button
                  type="button"
                  variant={materializedDataset ? "secondary" : "outline"}
                  size="sm"
                  className="mt-2 h-8 w-full text-xs"
                  onClick={materializeMix}
                  disabled={!canMaterializeMix || mixStatus.status === "queued" || mixStatus.status === "running"}
                >
                  {mixStatus.status === "queued" || mixStatus.status === "running" ? (
                    <Loader2 className="mr-1.5 h-3.5 w-3.5 animate-spin" />
                  ) : materializedDataset ? (
                    <CheckCircle2 className="mr-1.5 h-3.5 w-3.5" />
                  ) : (
                    <GitBranchPlus className="mr-1.5 h-3.5 w-3.5" />
                  )}
                  {materializedDataset ? "Mixed dataset ready" : "Materialize mix"}
                </Button>
              </div>
            )}

            <Button
              type="button"
              className={cn(EXPERIMENT_DASHBOARD_CLASS_NAMES.trainButton, "w-full")}
              onClick={configureTraining}
              disabled={!canConfigureTraining}
            >
              <Play className="mr-1.5 h-3.5 w-3.5" />
              Configure training
            </Button>
          </div>
        </section>

        <section className="rounded-md border border-border/60 bg-background/95 shadow-sm">
          <form className="space-y-3 p-3" onSubmit={handleUseHuggingFace}>
            <div className="flex items-center gap-2">
              <div className={EXPERIMENT_DASHBOARD_CLASS_NAMES.iconTile}>
                <Plus className="h-4 w-4 text-muted-foreground" />
              </div>
              <div>
                <label className="text-sm font-medium" htmlFor="hf-dataset-id">
                  Add Hugging Face dataset
                </label>
                <p className="text-xs text-muted-foreground">Repo ID or dataset URL</p>
              </div>
            </div>
            <Input
              id="hf-dataset-id"
              value={hfDatasetId}
              onChange={(event) => setHfDatasetId(event.target.value)}
              placeholder="lerobot/pusht"
              className="h-9 text-sm"
            />
            <div className="grid grid-cols-[1fr_auto] gap-2">
              <Button type="submit" size="sm" className="h-8 text-xs" disabled={!normalizedHfDatasetId}>
                Add to training set
              </Button>
              <Button
                type="button"
                variant="outline"
                size="sm"
                className="h-8 px-2"
                disabled={!normalizedHfDatasetId}
                onClick={() => {
                  window.open(`https://huggingface.co/datasets/${normalizedHfDatasetId}`, "_blank");
                }}
              >
                <ExternalLink className="h-3.5 w-3.5" />
              </Button>
            </div>
          </form>
        </section>

        <section className="rounded-md border border-border/60 bg-background/95 p-3 shadow-sm">
          <div className="grid grid-cols-3 gap-3">
            <div>
              <p className="text-xs text-muted-foreground">Sources</p>
              <p className="mt-1 text-lg font-semibold">{localSources.length + hfSources.length}</p>
            </div>
            <div>
              <p className="text-xs text-muted-foreground">Studio</p>
              <p className="mt-1 text-lg font-semibold">{studioExportCount}</p>
            </div>
            <div>
              <p className="text-xs text-muted-foreground">Selected</p>
              <p className="mt-1 text-lg font-semibold">{selectedSourceCount}</p>
            </div>
          </div>
          <Button
            type="button"
            variant="outline"
            size="sm"
            className="mt-3 h-8 w-full text-xs"
            onClick={() => refetch()}
            disabled={isFetching}
          >
            <RefreshCw className={cn("mr-1.5 h-3.5 w-3.5", isFetching && "animate-spin")} />
            Refresh local catalog
          </Button>
        </section>
      </aside>

      <div className="space-y-4">
        <div className="rounded-md border border-border/60 bg-background/95 p-3 shadow-sm">
          <div className="flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
            <div>
              <h2 className="text-sm font-medium">Dataset sources</h2>
              <p className="text-xs text-muted-foreground">
                Build a training set from Studio exports, local LeRobot roots, and Hugging Face repos.
              </p>
            </div>
            <div className="relative w-full md:w-80">
              <Search className="absolute left-2.5 top-2.5 h-4 w-4 text-muted-foreground" />
              <Input
                value={query}
                onChange={(event) => setQuery(event.target.value)}
                placeholder="Search sources"
                className="h-9 pl-9 text-sm"
              />
            </div>
          </div>
        </div>

        {isLoading && (
          <div className="flex items-center justify-center rounded-md border border-border/60 bg-background/95 py-12">
            <Loader2 className="h-7 w-7 animate-spin text-muted-foreground" />
          </div>
        )}

        {!isLoading && error && (
          <div className="rounded-md border border-border/60 bg-background/95 px-3 py-3 shadow-sm">
            <p className="text-sm font-medium">Local catalog unavailable</p>
            <p className="mt-1 text-xs text-muted-foreground">
              {catalogErrorMessage}. {catalogErrorHint} Hugging Face sources can still be added.
            </p>
          </div>
        )}

        {!isLoading && (
          <SourceTable
            title="Studio and local datasets"
            subtitle="LeRobot exports discovered from configured roots"
            icon={<FolderOpen className="h-4 w-4 text-muted-foreground" />}
            sources={localSources}
            selectedSources={selectedSources}
            query={query}
            emptyTitle={error ? "No local datasets loaded" : "No local LeRobot datasets found"}
            emptySubtitle={
              !error && data?.roots?.length
                ? data.roots.join(", ")
                : "Recorded Studio episodes appear here after Studio writes local LeRobot exports."
            }
            onToggleSource={toggleSource}
            onUseWholeSource={useWholeSource}
            onToggleEpisode={toggleEpisode}
          />
        )}

        <SourceTable
          title="Hugging Face datasets"
          subtitle="Repo IDs added to this Ops workspace"
          icon={<Database className="h-4 w-4 text-muted-foreground" />}
          sources={hfSources}
          selectedSources={selectedSources}
          query={query}
          emptyTitle="No Hugging Face datasets added"
          emptySubtitle="Use the Add Hugging Face dataset panel to add a repo ID or URL."
          onToggleSource={toggleSource}
          onUseWholeSource={useWholeSource}
          onToggleEpisode={toggleEpisode}
        />
      </div>
    </div>
  );
}
