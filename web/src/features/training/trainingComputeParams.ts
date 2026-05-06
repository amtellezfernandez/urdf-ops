import type { ComputeType } from "./types";

export type PlannedTrainingComputeBackend = "aws";
export type TrainingComputeBackendId = ComputeType | PlannedTrainingComputeBackend;

export interface TrainingComputeBackendOption {
  id: TrainingComputeBackendId;
  selectableType?: ComputeType;
  name: string;
  description: string;
  enabled: boolean;
  productionReady: boolean;
  reason?: string;
}

export const TRAINING_COMPUTE_PARAMS = {
  cloudDisabledMessage:
    "Cloud training runners are disabled in this build. Use local training until provider execution, log streaming, artifact download, and cancellation are wired.",
  cloudUnavailableBadge: "Disabled",
  localReadyBadge: "Available",
  localRuntimeReviewMessage: "Local runtime is checked before launch.",
  selectedComputeLabel: "Selected Compute",
  pollStatusIntervalMs: 2000,
} as const;

export const TRAINING_COMPUTE_BACKEND_NAMES: Record<TrainingComputeBackendId, string> = {
  local: "Local GPU",
  modal: "Modal",
  runpod: "RunPod",
  macrodata: "Macrodata Cloud",
  aws: "AWS",
};

export const TRAINING_COMPUTE_BACKENDS: readonly TrainingComputeBackendOption[] = [
  {
    id: "local",
    selectableType: "local",
    name: TRAINING_COMPUTE_BACKEND_NAMES.local,
    description: "Train on this computer with the local LeRobot runtime.",
    enabled: true,
    productionReady: true,
  },
  {
    id: "modal",
    selectableType: "modal",
    name: TRAINING_COMPUTE_BACKEND_NAMES.modal,
    description:
      "Cloud runner is held closed until execution, logs, artifacts, and cancellation are wired.",
    enabled: false,
    productionReady: false,
    reason: TRAINING_COMPUTE_PARAMS.cloudDisabledMessage,
  },
  {
    id: "runpod",
    selectableType: "runpod",
    name: TRAINING_COMPUTE_BACKEND_NAMES.runpod,
    description:
      "Cloud runner is held closed until execution, logs, artifacts, and cancellation are wired.",
    enabled: false,
    productionReady: false,
    reason: TRAINING_COMPUTE_PARAMS.cloudDisabledMessage,
  },
  {
    id: "macrodata",
    selectableType: "macrodata",
    name: TRAINING_COMPUTE_BACKEND_NAMES.macrodata,
    description:
      "Platform runner is held closed until submit, status, logs, cancel, and artifacts are wired.",
    enabled: false,
    productionReady: false,
    reason: TRAINING_COMPUTE_PARAMS.cloudDisabledMessage,
  },
  {
    id: "aws",
    name: TRAINING_COMPUTE_BACKEND_NAMES.aws,
    description: "AWS compute is not connected in this build.",
    enabled: false,
    productionReady: false,
    reason: TRAINING_COMPUTE_PARAMS.cloudDisabledMessage,
  },
] as const;

export const TRAINING_LOCAL_DEVICES = [
  { value: "cuda", label: "CUDA (NVIDIA GPU)" },
  { value: "mps", label: "MPS (Apple Silicon)" },
  { value: "cpu", label: "CPU" },
] as const;
