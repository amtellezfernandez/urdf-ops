import { TRAINING_COMPUTE_PARAMS } from "./trainingComputeParams";
import type { ComputeConfig } from "./types";

const hasRemoteDockerConfig = (computeConfig: ComputeConfig): boolean =>
  Boolean(computeConfig.sshHost?.trim() && computeConfig.sshUser?.trim());

export const canUseConfiguredComputeBackend = (computeConfig: ComputeConfig): boolean =>
  computeConfig.type === "local" ||
  (computeConfig.type === "ssh" && hasRemoteDockerConfig(computeConfig));

export const getConfiguredComputeBackendBlockReason = (
  computeConfig: ComputeConfig,
): string | null => {
  if (canUseConfiguredComputeBackend(computeConfig)) {
    return null;
  }
  if (computeConfig.type === "ssh") {
    return "Remote Docker training requires an SSH host and user.";
  }
  return TRAINING_COMPUTE_PARAMS.cloudDisabledMessage;
};

export const canStartWithTrainingCompute = ({
  computeConfig,
  localRuntimeAvailable,
}: {
  computeConfig: ComputeConfig;
  localRuntimeAvailable: boolean;
}): boolean =>
  computeConfig.type === "ssh"
    ? canUseConfiguredComputeBackend(computeConfig)
    : canUseConfiguredComputeBackend(computeConfig) && localRuntimeAvailable;
