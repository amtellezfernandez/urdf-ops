import { FormEvent, useEffect, useMemo, useState } from "react";
import {
  Database,
  ExternalLink,
  FolderOpen,
  Loader2,
  Play,
  RefreshCw,
} from "lucide-react";
import { useQuery } from "@tanstack/react-query";

import { Button } from "@/shared/ui/button";
import { Input } from "@/shared/ui/input";
import { Badge } from "@/shared/ui/badge";
import { API_BASE_URL } from "@/shared/config/api";
import { cn } from "@/shared/lib/utils";
import { useTrainingStore } from "@/features/training";

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

const numberFormatter = new Intl.NumberFormat();

function normalizeHfDatasetId(value: string): string {
  return value
    .trim()
    .replace(/^https:\/\/huggingface\.co\/datasets\//i, "")
    .split(/[?#]/)[0]
    .replace(/^\/+|\/+$/g, "");
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

interface DatasetRowProps {
  dataset: DatasetCatalogItem;
  onTrain: (dataset: DatasetCatalogItem) => void;
}

function DatasetRow({ dataset, onTrain }: DatasetRowProps) {
  return (
    <div className="grid gap-3 border-b border-border/50 px-3 py-3 last:border-b-0 md:grid-cols-[minmax(0,1.5fr)_140px_140px_120px] md:items-center">
      <div className="min-w-0">
        <div className="flex flex-wrap items-center gap-2">
          <p className="truncate text-sm font-medium" title={dataset.name}>
            {dataset.name}
          </p>
          <Badge className="border-border/60 bg-muted/30 text-[11px] font-normal text-muted-foreground">
            {sourceLabel(dataset.source)}
          </Badge>
        </div>
        <p className="mt-1 truncate font-mono text-xs text-muted-foreground" title={dataset.path}>
          {dataset.path}
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

      <div className="flex items-center justify-between gap-2 md:justify-end">
        <span className="text-xs text-muted-foreground md:hidden">
          {formatDate(dataset.created_at)}
        </span>
        <Button size="sm" className="h-7 text-xs" onClick={() => onTrain(dataset)}>
          <Play className="mr-1.5 h-3.5 w-3.5" />
          Train
        </Button>
      </div>
    </div>
  );
}

interface InitialDatasetCardProps {
  dataset: InitialExperimentDataset;
  onTrain: () => void;
}

function InitialDatasetCard({ dataset, onTrain }: InitialDatasetCardProps) {
  const datasetValue = dataset.id;
  return (
    <div className="rounded-md border border-sky-500/30 bg-sky-500/5 p-3">
      <div className="flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
        <div className="min-w-0">
          <div className="flex flex-wrap items-center gap-2">
            <p className="truncate text-sm font-medium" title={dataset.name}>
              {dataset.name}
            </p>
            <Badge className="border-sky-500/30 bg-sky-500/10 text-[11px] font-normal text-sky-500">
              Opened from Studio
            </Badge>
            <Badge className="border-border/60 bg-muted/30 text-[11px] font-normal text-muted-foreground">
              {dataset.source === "local" ? "Local" : "Hugging Face"}
            </Badge>
          </div>
          <p className="mt-1 truncate font-mono text-xs text-muted-foreground" title={datasetValue}>
            {datasetValue}
          </p>
        </div>
        <Button size="sm" className="h-7 text-xs md:self-end" onClick={onTrain}>
          <Play className="mr-1.5 h-3.5 w-3.5" />
          Train
        </Button>
      </div>
    </div>
  );
}

interface DatasetCatalogProps {
  initialDataset?: InitialExperimentDataset | null;
}

export function DatasetCatalog({ initialDataset }: DatasetCatalogProps) {
  const [hfDatasetId, setHfDatasetId] = useState("");
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

  const datasets = data?.datasets ?? [];
  const studioExports = useMemo(
    () => datasets.filter((dataset) => dataset.source === "studio_export"),
    [datasets],
  );
  const catalogErrorMessage = error instanceof Error ? error.message : "Dataset catalog is unavailable";
  const normalizedHfDatasetId = normalizeHfDatasetId(hfDatasetId);
  const initialDatasetSelected =
    initialDataset?.source === "local"
      ? datasetConfig?.source === "local" && datasetConfig.localPath === initialDataset.id
      : datasetConfig?.source === "huggingface" && datasetConfig.repoId === initialDataset?.id;

  const openTrainingWithDataset = () => {
    openTrainingDialog();
    setTrainingStep("model");
  };

  useEffect(() => {
    if (!initialDataset || initialDatasetSelected) return;
    if (initialDataset.source === "local") {
      setDatasetConfig({
        source: "local",
        localPath: initialDataset.id,
      });
    } else {
      setDatasetConfig({
        source: "huggingface",
        repoId: initialDataset.id,
      });
    }
  }, [initialDataset, initialDatasetSelected, setDatasetConfig]);

  const handleTrainLocal = (dataset: DatasetCatalogItem) => {
    setDatasetConfig({
      source: "local",
      localPath: dataset.path,
    });
    openTrainingWithDataset();
  };

  const handleTrainInitialDataset = () => {
    if (!initialDataset) return;
    if (initialDataset.source === "local") {
      setDatasetConfig({
        source: "local",
        localPath: initialDataset.id,
      });
    } else {
      setDatasetConfig({
        source: "huggingface",
        repoId: initialDataset.id,
      });
    }
    openTrainingWithDataset();
  };

  const handleUseHuggingFace = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (!normalizedHfDatasetId) return;
    setDatasetConfig({
      source: "huggingface",
      repoId: normalizedHfDatasetId,
    });
    openTrainingWithDataset();
  };

  return (
    <div className="space-y-4">
      <div className="grid gap-3 md:grid-cols-3">
        <div className={EXPERIMENT_DASHBOARD_CLASS_NAMES.card}>
          <div className="flex items-start justify-between">
            <div>
              <p className="text-xs text-muted-foreground">Visible Datasets</p>
              <p className="mt-1 text-lg font-semibold">{datasets.length}</p>
            </div>
            <div className={EXPERIMENT_DASHBOARD_CLASS_NAMES.iconTile}>
              <Database className="h-4 w-4 text-muted-foreground" />
            </div>
          </div>
        </div>
        <div className={EXPERIMENT_DASHBOARD_CLASS_NAMES.card}>
          <div className="flex items-start justify-between">
            <div>
              <p className="text-xs text-muted-foreground">Studio Exports</p>
              <p className="mt-1 text-lg font-semibold">{studioExports.length}</p>
            </div>
            <div className={EXPERIMENT_DASHBOARD_CLASS_NAMES.iconTile}>
              <FolderOpen className="h-4 w-4 text-sky-500" />
            </div>
          </div>
        </div>
        <form className={EXPERIMENT_DASHBOARD_CLASS_NAMES.card} onSubmit={handleUseHuggingFace}>
          <label className="text-xs text-muted-foreground" htmlFor="hf-dataset-id">
            Add Hugging Face Dataset
          </label>
          <div className="mt-2 flex gap-2">
            <Input
              id="hf-dataset-id"
              value={hfDatasetId}
              onChange={(event) => setHfDatasetId(event.target.value)}
              placeholder="lerobot/pusht"
              className="h-8 text-xs"
            />
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
            <Button type="submit" size="sm" className="h-8 text-xs" disabled={!normalizedHfDatasetId}>
              Use
            </Button>
          </div>
        </form>
      </div>

      {initialDataset && (
        <InitialDatasetCard dataset={initialDataset} onTrain={handleTrainInitialDataset} />
      )}

      <div className="overflow-hidden rounded-md border border-border/60 bg-background/95 shadow-sm">
        <div className="flex items-center justify-between border-b border-border/60 px-3 py-2">
          <div>
            <h2 className="text-sm font-medium">Datasets</h2>
            <p className="text-xs text-muted-foreground">
              Studio LeRobot exports and configured local roots
            </p>
          </div>
          <Button
            variant="outline"
            size="sm"
            className="h-7 text-xs"
            onClick={() => refetch()}
            disabled={isFetching}
          >
            <RefreshCw className={cn("mr-1.5 h-3.5 w-3.5", isFetching && "animate-spin")} />
            Refresh
          </Button>
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
              {catalogErrorMessage}. Restart the URDF Ops backend to enable local
              Studio export discovery. Hugging Face IDs and Studio deep links can
              still be used.
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
          </div>
        )}

        {!isLoading && !error && datasets.length > 0 && (
          <div>
            <div className="hidden grid-cols-[minmax(0,1.5fr)_140px_140px_120px] border-b border-border/60 bg-muted/20 px-3 py-2 text-xs text-muted-foreground md:grid">
              <div>Dataset</div>
              <div>Robot</div>
              <div>Size</div>
              <div className="text-right">Action</div>
            </div>
            {datasets.map((dataset) => (
              <DatasetRow key={dataset.id} dataset={dataset} onTrain={handleTrainLocal} />
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
