import { FormEvent, useEffect, useMemo, useState } from "react";
import {
  CheckCircle2,
  Database,
  ExternalLink,
  FolderOpen,
  ListChecks,
  Loader2,
  Play,
  Plus,
  RefreshCw,
} from "lucide-react";
import { useQuery } from "@tanstack/react-query";

import { Button } from "@/shared/ui/button";
import { Input } from "@/shared/ui/input";
import { Badge } from "@/shared/ui/badge";
import { API_BASE_URL } from "@/shared/config/api";
import { cn } from "@/shared/lib/utils";
import { useTrainingStore } from "@/features/training";
import type { DatasetConfig } from "@/features/training/types";

import { EXPERIMENT_DASHBOARD_CLASS_NAMES } from "./experimentDashboardParams";

type DatasetCatalogSource = "studio_export" | "local";

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

export interface InitialExperimentDataset {
  id: string;
  name: string;
  source: "huggingface" | "local";
  author?: string;
}

const HF_DATASETS_STORAGE_KEY = "urdf-ops:hf-datasets";
const EMPTY_DATASETS: DatasetCatalogItem[] = [];

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
  if (value == null) return `0 ${label}s`;
  return `${numberFormatter.format(value)} ${label}${value === 1 ? "" : "s"}`;
}

function formatDate(value: string | null | undefined): string {
  if (!value) return "Unknown";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "Unknown";
  return date.toLocaleString([], {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

function sourceLabel(source: DatasetCatalogSource): string {
  return source === "studio_export" ? "Studio export" : "Local";
}

function sortedEpisodes(episodes: number[]): number[] {
  return Array.from(new Set(episodes)).sort((a, b) => a - b);
}

function isLocalDatasetSelected(datasetConfig: DatasetConfig | null, dataset: DatasetCatalogItem): boolean {
  return datasetConfig?.source === "local" && datasetConfig.localPath === dataset.path;
}

function isHfDatasetSelected(datasetConfig: DatasetConfig | null, repoId: string): boolean {
  return datasetConfig?.source === "huggingface" && datasetConfig.repoId === repoId;
}

function selectionScopeLabel(datasetConfig: DatasetConfig | null, totalEpisodes?: number | null): string {
  if (!datasetConfig) return "No dataset selected";
  if (!datasetConfig.episodes) {
    return totalEpisodes ? `All ${formatCount(totalEpisodes, "episode")}` : "Whole dataset";
  }
  if (datasetConfig.episodes.length === 0) return "No episodes selected";
  return `${formatCount(datasetConfig.episodes.length, "episode")} selected`;
}

interface EpisodeSelectorProps {
  totalEpisodes: number | null | undefined;
  selectedEpisodes?: number[];
  onUseWholeDataset: () => void;
  onToggleEpisode: (episodeIndex: number) => void;
}

function EpisodeSelector({
  totalEpisodes,
  selectedEpisodes,
  onUseWholeDataset,
  onToggleEpisode,
}: EpisodeSelectorProps) {
  const episodeCount = Math.max(0, totalEpisodes ?? 0);
  const selectedSet = new Set(selectedEpisodes ?? []);
  const usingWholeDataset = selectedEpisodes === undefined;
  const episodeIndexes = Array.from({ length: episodeCount }, (_, index) => index);

  return (
    <div className="rounded-md border border-border/60 bg-background/80 p-2">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div>
          <p className="text-xs font-medium">Episode scope</p>
          <p className="text-xs text-muted-foreground">
            {usingWholeDataset
              ? episodeCount > 0
                ? `Using all ${episodeCount} episodes`
                : "Using whole dataset"
              : `${selectedSet.size} selected`}
          </p>
        </div>
        <Button
          type="button"
          variant={usingWholeDataset ? "default" : "outline"}
          size="sm"
          className="h-7 text-xs"
          onClick={onUseWholeDataset}
        >
          Whole dataset
        </Button>
      </div>

      {episodeCount > 0 ? (
        <div className="mt-2 flex max-h-28 flex-wrap gap-1 overflow-auto pr-1">
          {episodeIndexes.map((episodeIndex) => {
            const selected = usingWholeDataset || selectedSet.has(episodeIndex);
            return (
              <button
                key={episodeIndex}
                type="button"
                onClick={() => onToggleEpisode(episodeIndex)}
                className={cn(
                  "h-7 min-w-9 rounded-md border px-2 text-xs transition-colors",
                  selected
                    ? "border-foreground bg-foreground text-background"
                    : "border-border/70 bg-background text-muted-foreground hover:border-foreground/50 hover:text-foreground",
                )}
              >
                {episodeIndex}
              </button>
            );
          })}
        </div>
      ) : (
        <p className="mt-2 text-xs text-muted-foreground">
          Episode metadata is not available for this dataset, so training uses the full dataset.
        </p>
      )}
    </div>
  );
}

interface LocalDatasetRowProps {
  dataset: DatasetCatalogItem;
  selected: boolean;
  selectedEpisodes?: number[];
  onUseWholeDataset: () => void;
  onToggleEpisode: (episodeIndex: number) => void;
  onConfigureTraining: () => void;
}

function LocalDatasetRow({
  dataset,
  selected,
  selectedEpisodes,
  onUseWholeDataset,
  onToggleEpisode,
  onConfigureTraining,
}: LocalDatasetRowProps) {
  return (
    <div
      className={cn(
        "grid gap-3 border-b border-border/50 px-3 py-3 last:border-b-0 md:grid-cols-[minmax(0,1.45fr)_140px_140px_190px] md:items-center",
        selected && "bg-primary/5",
      )}
    >
      <div className="min-w-0">
        <div className="flex flex-wrap items-center gap-2">
          <p className="truncate text-sm font-medium" title={dataset.name}>
            {dataset.name}
          </p>
          <Badge className="border-border/60 bg-muted/30 text-[11px] font-normal text-muted-foreground">
            {sourceLabel(dataset.source)}
          </Badge>
          {selected && (
            <Badge className="border-emerald-500/30 bg-emerald-500/10 text-[11px] font-normal text-emerald-600">
              Selected
            </Badge>
          )}
        </div>
        <p className="mt-1 truncate font-mono text-xs text-muted-foreground" title={dataset.path}>
          {dataset.path}
        </p>
        <p className="mt-1 text-xs text-muted-foreground md:hidden">
          {selectionScopeLabel(selected ? { source: "local", localPath: dataset.path, episodes: selectedEpisodes } : null, dataset.total_episodes)}
        </p>
      </div>

      <div className="text-xs text-muted-foreground">
        <div className="font-medium text-foreground">
          {dataset.robot_type || "Unknown robot"}
        </div>
        <div>{dataset.format_version || "LeRobot"}</div>
      </div>

      <div className="text-xs text-muted-foreground">
        <div>{formatCount(dataset.total_episodes, "episode")}</div>
        <div>{formatCount(dataset.total_frames, "frame")}</div>
      </div>

      <div className="flex flex-wrap items-center gap-2 md:justify-end">
        <span className="text-xs text-muted-foreground md:hidden">
          {formatDate(dataset.created_at)}
        </span>
        <Button
          type="button"
          variant={selected ? "secondary" : "outline"}
          size="sm"
          className="h-8 text-xs"
          onClick={onUseWholeDataset}
        >
          {selected ? (
            <CheckCircle2 className="mr-1.5 h-3.5 w-3.5" />
          ) : (
            <ListChecks className="mr-1.5 h-3.5 w-3.5" />
          )}
          Use all
        </Button>
        {selected && (
          <Button
            type="button"
            size="sm"
            className={cn(EXPERIMENT_DASHBOARD_CLASS_NAMES.trainButton, "h-8")}
            onClick={onConfigureTraining}
            disabled={selectedEpisodes?.length === 0}
          >
            <Play className="mr-1.5 h-3.5 w-3.5" />
            Configure
          </Button>
        )}
      </div>

      {selected && (
        <div className="md:col-span-4">
          <EpisodeSelector
            totalEpisodes={dataset.total_episodes}
            selectedEpisodes={selectedEpisodes}
            onUseWholeDataset={onUseWholeDataset}
            onToggleEpisode={onToggleEpisode}
          />
        </div>
      )}
    </div>
  );
}

interface HfDatasetRowProps {
  repoId: string;
  selected: boolean;
  onSelect: () => void;
  onConfigureTraining: () => void;
}

function HfDatasetRow({ repoId, selected, onSelect, onConfigureTraining }: HfDatasetRowProps) {
  return (
    <div
      className={cn(
        "grid gap-3 border-b border-border/50 px-3 py-3 last:border-b-0 md:grid-cols-[minmax(0,1fr)_180px] md:items-center",
        selected && "bg-primary/5",
      )}
    >
      <div className="min-w-0">
        <div className="flex flex-wrap items-center gap-2">
          <p className="truncate text-sm font-medium" title={repoId}>
            {repoId}
          </p>
          <Badge className="border-border/60 bg-muted/30 text-[11px] font-normal text-muted-foreground">
            Hugging Face
          </Badge>
          {selected && (
            <Badge className="border-emerald-500/30 bg-emerald-500/10 text-[11px] font-normal text-emerald-600">
              Selected
            </Badge>
          )}
        </div>
        <p className="mt-1 truncate font-mono text-xs text-muted-foreground">
          huggingface.co/datasets/{repoId}
        </p>
      </div>
      <div className="flex flex-wrap items-center gap-2 md:justify-end">
        <Button type="button" variant={selected ? "secondary" : "outline"} size="sm" className="h-8 text-xs" onClick={onSelect}>
          {selected ? "Selected" : "Use dataset"}
        </Button>
        {selected && (
          <Button
            type="button"
            size="sm"
            className={cn(EXPERIMENT_DASHBOARD_CLASS_NAMES.trainButton, "h-8")}
            onClick={onConfigureTraining}
          >
            <Play className="mr-1.5 h-3.5 w-3.5" />
            Configure
          </Button>
        )}
      </div>
    </div>
  );
}

interface DatasetCatalogProps {
  initialDataset?: InitialExperimentDataset | null;
}

export function DatasetCatalog({ initialDataset }: DatasetCatalogProps) {
  const [hfDatasetId, setHfDatasetId] = useState("");
  const [hfDatasetIds, setHfDatasetIds] = useState<string[]>(readStoredHfDatasetIds);
  const datasetConfig = useTrainingStore((state) => state.datasetConfig);
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
  const studioExports = useMemo(
    () => datasets.filter((dataset) => dataset.source === "studio_export"),
    [datasets],
  );
  const catalogErrorMessage = error instanceof Error ? error.message : "Dataset catalog is unavailable";
  const catalogErrorHint =
    catalogErrorMessage.includes(":8000/") && catalogErrorMessage.includes("returned 404")
      ? "This URDF Ops UI is pointed at the URDF Studio backend on port 8000. Restart URDF Ops from the updated repo so the UI uses the Ops backend on port 8001."
      : "Restart the URDF Ops backend to enable local Studio export discovery.";
  const normalizedHfDatasetId = normalizeHfDatasetId(hfDatasetId);
  const initialDatasetSelected =
    initialDataset?.source === "local"
      ? datasetConfig?.source === "local" && datasetConfig.localPath === initialDataset.id
      : datasetConfig?.source === "huggingface" && datasetConfig.repoId === initialDataset?.id;

  const selectedLocalDataset = useMemo(
    () =>
      datasetConfig?.source === "local"
        ? datasets.find((dataset) => dataset.path === datasetConfig.localPath)
        : undefined,
    [datasetConfig, datasets],
  );
  const selectedDatasetTitle =
    datasetConfig?.source === "local"
      ? selectedLocalDataset?.name ?? datasetConfig.localPath ?? "Local dataset"
      : datasetConfig?.repoId ?? "No dataset selected";
  const selectedDatasetValue =
    datasetConfig?.source === "local"
      ? datasetConfig.localPath ?? selectedLocalDataset?.path
      : datasetConfig?.repoId;
  const canConfigureTraining = Boolean(datasetConfig) && (datasetConfig?.episodes === undefined || datasetConfig.episodes.length > 0);
  const selectedScope = selectionScopeLabel(
    datasetConfig,
    datasetConfig?.source === "local" ? selectedLocalDataset?.total_episodes : null,
  );

  useEffect(() => {
    if (typeof window === "undefined") return;
    window.localStorage.setItem(HF_DATASETS_STORAGE_KEY, JSON.stringify(hfDatasetIds));
  }, [hfDatasetIds]);

  useEffect(() => {
    if (!initialDataset || initialDatasetSelected) return;
    if (initialDataset.source === "local") {
      setDatasetConfig({
        source: "local",
        localPath: initialDataset.id,
      });
    } else {
      const normalizedInitialDataset = normalizeHfDatasetId(initialDataset.id);
      setHfDatasetIds((current) => normalizeHfDatasetIds([...current, normalizedInitialDataset]));
      setDatasetConfig({
        source: "huggingface",
        repoId: normalizedInitialDataset,
      });
    }
  }, [initialDataset, initialDatasetSelected, setDatasetConfig]);

  const configureTraining = () => {
    if (!canConfigureTraining) return;
    openTrainingDialog();
    setTrainingStep("model");
  };

  const selectLocalDataset = (dataset: DatasetCatalogItem, episodes?: number[]) => {
    setDatasetConfig({
      source: "local",
      localPath: dataset.path,
      episodes,
    });
  };

  const toggleLocalEpisode = (dataset: DatasetCatalogItem, episodeIndex: number) => {
    const selected = isLocalDatasetSelected(datasetConfig, dataset);
    const currentEpisodes = selected && datasetConfig?.episodes ? datasetConfig.episodes : [];
    const nextEpisodes = new Set(currentEpisodes);
    if (selected && datasetConfig?.episodes && nextEpisodes.has(episodeIndex)) {
      nextEpisodes.delete(episodeIndex);
    } else {
      nextEpisodes.add(episodeIndex);
    }
    selectLocalDataset(dataset, sortedEpisodes(Array.from(nextEpisodes)));
  };

  const handleUseHuggingFace = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (!normalizedHfDatasetId) return;
    setHfDatasetIds((current) => normalizeHfDatasetIds([...current, normalizedHfDatasetId]));
    setDatasetConfig({
      source: "huggingface",
      repoId: normalizedHfDatasetId,
    });
    setHfDatasetId("");
  };

  return (
    <div className="grid min-h-full gap-4 lg:grid-cols-[320px_minmax(0,1fr)]">
      <aside className="space-y-3">
        <section className="rounded-md border border-border/60 bg-background/95 shadow-sm">
          <div className="border-b border-border/60 px-3 py-3">
            <div className="flex items-center gap-2">
              <div className={EXPERIMENT_DASHBOARD_CLASS_NAMES.iconTile}>
                <ListChecks className="h-4 w-4 text-muted-foreground" />
              </div>
              <div>
                <h2 className="text-sm font-medium">Training set</h2>
                <p className="text-xs text-muted-foreground">{selectedScope}</p>
              </div>
            </div>
          </div>
          <div className="space-y-3 p-3">
            {datasetConfig ? (
              <div>
                <p className="truncate text-sm font-medium" title={selectedDatasetTitle}>
                  {selectedDatasetTitle}
                </p>
                {selectedDatasetValue && (
                  <p className="mt-1 truncate font-mono text-xs text-muted-foreground" title={selectedDatasetValue}>
                    {selectedDatasetValue}
                  </p>
                )}
                {datasetConfig.episodes?.length === 0 && (
                  <p className="mt-2 text-xs text-destructive">
                    Select at least one episode or use the whole dataset.
                  </p>
                )}
              </div>
            ) : (
              <p className="text-sm text-muted-foreground">Select a dataset from the catalog.</p>
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
                Use dataset
              </Button>
              <Button
                type="button"
                variant="outline"
                size="sm"
                className="h-8 px-2"
                disabled={!normalizedHfDatasetId}
                onClick={() => {
                  window.open(
                    `https://huggingface.co/datasets/${normalizedHfDatasetId}`,
                    "_blank",
                  );
                }}
              >
                <ExternalLink className="h-3.5 w-3.5" />
              </Button>
            </div>
          </form>
        </section>

        <section className="rounded-md border border-border/60 bg-background/95 p-3 shadow-sm">
          <div className="grid grid-cols-2 gap-3">
            <div>
              <p className="text-xs text-muted-foreground">Visible</p>
              <p className="mt-1 text-lg font-semibold">{datasets.length + hfDatasetIds.length}</p>
            </div>
            <div>
              <p className="text-xs text-muted-foreground">Studio exports</p>
              <p className="mt-1 text-lg font-semibold">{studioExports.length}</p>
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
        <section className="overflow-hidden rounded-md border border-border/60 bg-background/95 shadow-sm">
          <div className="flex items-center justify-between border-b border-border/60 px-3 py-3">
            <div>
              <h2 className="text-sm font-medium">Local datasets</h2>
              <p className="text-xs text-muted-foreground">
                Studio LeRobot exports and configured local roots
              </p>
            </div>
            <div className={EXPERIMENT_DASHBOARD_CLASS_NAMES.iconTile}>
              <FolderOpen className="h-4 w-4 text-muted-foreground" />
            </div>
          </div>

          {isLoading && (
            <div className="flex items-center justify-center py-12">
              <Loader2 className="h-7 w-7 animate-spin text-muted-foreground" />
            </div>
          )}

          {!isLoading && error && (
            <div className="border-b border-border/60 px-3 py-3">
              <p className="text-sm font-medium">Local catalog unavailable</p>
              <p className="mt-1 text-xs text-muted-foreground">
                {catalogErrorMessage}. {catalogErrorHint} Hugging Face IDs and
                Studio deep links can still be used.
              </p>
            </div>
          )}

          {!isLoading && datasets.length === 0 && (
            <div className="px-3 py-8 text-center">
              <p className="text-sm font-medium">
                {error ? "No local datasets loaded" : "No local LeRobot datasets found"}
              </p>
              {!error && data?.roots?.length ? (
                <p className="mt-1 truncate font-mono text-xs text-muted-foreground">
                  {data.roots.join(", ")}
                </p>
              ) : null}
              {!error ? (
                <p className="mx-auto mt-2 max-w-md text-xs text-muted-foreground">
                  Recorded Studio episodes appear here after Studio writes local
                  LeRobot exports for URDF Ops.
                </p>
              ) : null}
            </div>
          )}

          {!isLoading && !error && datasets.length > 0 && (
            <div>
              <div className="hidden grid-cols-[minmax(0,1.45fr)_140px_140px_190px] border-b border-border/60 bg-muted/20 px-3 py-2 text-xs text-muted-foreground md:grid">
                <div>Dataset</div>
                <div>Robot</div>
                <div>Size</div>
                <div className="text-right">Selection</div>
              </div>
              {datasets.map((dataset) => {
                const selected = isLocalDatasetSelected(datasetConfig, dataset);
                return (
                  <LocalDatasetRow
                    key={dataset.id}
                    dataset={dataset}
                    selected={selected}
                    selectedEpisodes={selected ? datasetConfig?.episodes : undefined}
                    onUseWholeDataset={() => selectLocalDataset(dataset)}
                    onToggleEpisode={(episodeIndex) => toggleLocalEpisode(dataset, episodeIndex)}
                    onConfigureTraining={configureTraining}
                  />
                );
              })}
            </div>
          )}
        </section>

        <section className="overflow-hidden rounded-md border border-border/60 bg-background/95 shadow-sm">
          <div className="flex items-center justify-between border-b border-border/60 px-3 py-3">
            <div>
              <h2 className="text-sm font-medium">Hugging Face datasets</h2>
              <p className="text-xs text-muted-foreground">Configured repo IDs for training</p>
            </div>
            <div className={EXPERIMENT_DASHBOARD_CLASS_NAMES.iconTile}>
              <Database className="h-4 w-4 text-muted-foreground" />
            </div>
          </div>

          {hfDatasetIds.length > 0 ? (
            <div>
              {hfDatasetIds.map((repoId) => {
                const selected = isHfDatasetSelected(datasetConfig, repoId);
                return (
                  <HfDatasetRow
                    key={repoId}
                    repoId={repoId}
                    selected={selected}
                    onSelect={() => setDatasetConfig({ source: "huggingface", repoId })}
                    onConfigureTraining={configureTraining}
                  />
                );
              })}
            </div>
          ) : (
            <div className="px-3 py-8 text-center">
              <p className="text-sm font-medium">No Hugging Face datasets configured</p>
              <p className="mx-auto mt-2 max-w-md text-xs text-muted-foreground">
                Add a repo ID from the side panel to make it available here.
              </p>
            </div>
          )}
        </section>
      </div>
    </div>
  );
}
