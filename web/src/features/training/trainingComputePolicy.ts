import { TRAINING_COMPUTE_PARAMS } from "./trainingComputeParams";
import type { ComputeConfig } from "./types";

export const canUseConfiguredComputeBackend = (computeConfig: ComputeConfig): boolean =>
  computeConfig.type === "local";

export const getConfiguredComputeBackendBlockReason = (
  computeConfig: ComputeConfig,
): string | null => {
  if (canUseConfiguredComputeBackend(computeConfig)) {
    return null;
  }
  return TRAINING_COMPUTE_PARAMS.cloudDisabledMessage;
};

export const canStartWithTrainingCompute = ({
  computeConfig,
  localRuntimeAvailable,
}: {
  computeConfig: ComputeConfig;
  localRuntimeAvailable: boolean;
}): boolean => canUseConfiguredComputeBackend(computeConfig) && localRuntimeAvailable;
