import type { DatasetInfo, DatasetSource } from "@/features/datasets";
import { URDF_OPS_QUERY_PARAMS } from "@/shared/config/urdfOpsRoutes";

const URDF_OPS_DATASET_SOURCE_VALUES = new Set<DatasetSource>([
  "huggingface",
  "local",
]);

const resolveUrdfOpsDatasetSource = (
  value: string | null | undefined,
): DatasetSource => {
  if (value && URDF_OPS_DATASET_SOURCE_VALUES.has(value as DatasetSource)) {
    return value as DatasetSource;
  }
  return "huggingface";
};

export const buildUrdfOpsInitialDataset = (
  searchParams: URLSearchParams,
): DatasetInfo | null => {
  const datasetId = searchParams.get(URDF_OPS_QUERY_PARAMS.dataset)?.trim();
  if (!datasetId) return null;

  const source = resolveUrdfOpsDatasetSource(
    searchParams.get(URDF_OPS_QUERY_PARAMS.source),
  );
  const pathParts = datasetId.split("/").filter(Boolean);
  const name = pathParts.at(-1) || datasetId;

  return {
    id: datasetId,
    name,
    source,
    author: source === "huggingface" ? pathParts.at(0) : undefined,
    tags: [],
  };
};
