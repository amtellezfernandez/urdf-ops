/**
 * Training review component - final step before launching.
 * Shows a summary of all configuration and validates readiness.
 */

import { useEffect, useState } from "react";
import { AlertCircle, CheckCircle2, Loader2 } from "lucide-react";

import { useTrainingStore } from "./useTrainingStore";
import { fetchTrainingRuntimeCheck } from "./trainingApi";
import {
  TRAINING_COMPUTE_BACKEND_NAMES,
  TRAINING_COMPUTE_PARAMS,
} from "./trainingComputeParams";
import {
  canStartWithTrainingCompute,
  getConfiguredComputeBackendBlockReason,
} from "./trainingComputePolicy";
import type { TrainingRuntimeCheckResponse } from "./types";

interface ReviewSectionProps {
  title: string;
  valid: boolean;
  children: React.ReactNode;
}

function ReviewSection({ title, valid, children }: ReviewSectionProps) {
  return (
    <div className="p-3 rounded-lg border bg-card">
      <div className="flex items-center gap-2 mb-2">
        {valid ? (
          <CheckCircle2 className="w-4 h-4 text-green-500" />
        ) : (
          <AlertCircle className="w-4 h-4 text-amber-500" />
        )}
        <span className="text-sm font-medium">{title}</span>
      </div>
      <div className="text-sm text-muted-foreground pl-6">{children}</div>
    </div>
  );
}

export function TrainingReview() {
  const [runtimeCheck, setRuntimeCheck] = useState<TrainingRuntimeCheckResponse | null>(null);
  const [runtimeCheckError, setRuntimeCheckError] = useState<string | null>(null);
  const [isCheckingRuntime, setIsCheckingRuntime] = useState(false);
  const {
    datasetConfig,
    modelConfig,
    trainingParams,
    trackerConfig,
    computeConfig,
  } = useTrainingStore();

  useEffect(() => {
    if (computeConfig.type !== "local") {
      setRuntimeCheck(null);
      setRuntimeCheckError(null);
      return;
    }

    let cancelled = false;
    setIsCheckingRuntime(true);
    fetchTrainingRuntimeCheck()
      .then((result) => {
        if (!cancelled) {
          setRuntimeCheck(result);
          setRuntimeCheckError(null);
        }
      })
      .catch((error: unknown) => {
        if (!cancelled) {
          setRuntimeCheck(null);
          setRuntimeCheckError(error instanceof Error ? error.message : "Failed to check runtime");
        }
      })
      .finally(() => {
        if (!cancelled) {
          setIsCheckingRuntime(false);
        }
      });

    return () => {
      cancelled = true;
    };
  }, [computeConfig.type]);

  // Get model display name
  const getModelDisplayName = (arch: string) => {
    const names: Record<string, string> = {
      act: "ACT (Action Chunking Transformer)",
      diffusion_policy: "Diffusion Policy",
      dreamzero: "DreamZero",
      lereal_world_model: "LeRealWorldModel (JEPA + GC-IDM)",
      tdmpc: "TD-MPC",
      vq_bet: "VQ-BeT",
      custom: "Custom Model",
    };
    return names[arch] || arch;
  };

  // Get tracker display name
  const getTrackerDisplayName = (type: string) => {
    const names: Record<string, string> = {
      none: "None (Local only)",
      mlflow: "MLflow",
      wandb: "Weights & Biases",
    };
    return names[type] || type;
  };

  const computeBlockReason = getConfiguredComputeBackendBlockReason(computeConfig);
  const localRuntimeAvailable = Boolean(runtimeCheck?.available);
  const canStartCompute = canStartWithTrainingCompute({
    computeConfig,
    localRuntimeAvailable,
  });

  return (
    <div className="space-y-3">
      <p className="text-sm text-muted-foreground mb-4">
        Review your configuration before starting training.
      </p>

      {/* Dataset */}
      <ReviewSection title="Dataset" valid={datasetConfig !== null}>
        {datasetConfig ? (
          <div className="space-y-1">
            <div className="font-mono text-xs">
              {datasetConfig.source === "huggingface"
                ? datasetConfig.repoId
                : datasetConfig.localPath}
            </div>
            <div className="text-xs opacity-75">
              Source: {datasetConfig.source === "huggingface" ? "HuggingFace Hub" : "Local"}
            </div>
          </div>
        ) : (
          <span className="text-amber-500">No dataset selected</span>
        )}
      </ReviewSection>

      {computeConfig.type === "local" ? (
        <ReviewSection
          title="Local Runtime"
          valid={Boolean(runtimeCheck?.available)}
        >
          {isCheckingRuntime ? (
            <div className="flex items-center gap-2 text-xs">
              <Loader2 className="h-3.5 w-3.5 animate-spin" />
              Checking backend runtime...
            </div>
          ) : runtimeCheck ? (
            <div className="space-y-2">
              <div>{runtimeCheck.message}</div>
              <div className="space-y-1 text-xs">
                {runtimeCheck.dependencies.map((dependency) => (
                  <div key={dependency.name} className="flex items-center justify-between gap-3">
                    <span className="font-mono">{dependency.name}</span>
                    <span className={dependency.installed ? "text-green-600" : "text-amber-600"}>
                      {dependency.installed ? dependency.version || "installed" : "missing"}
                    </span>
                  </div>
                ))}
              </div>
              <div className="text-xs opacity-75">
                Python: <span className="font-mono">{runtimeCheck.pythonExecutable}</span>
              </div>
            </div>
          ) : (
            <span className="text-amber-600">{runtimeCheckError || "Runtime check unavailable"}</span>
          )}
        </ReviewSection>
      ) : null}

      {/* Model */}
      <ReviewSection title="Model" valid={modelConfig !== null}>
        {modelConfig ? (
          <div className="space-y-1">
            <div>{getModelDisplayName(modelConfig.architecture)}</div>
            {Object.keys(modelConfig.config).length > 0 && (
              <div className="text-xs opacity-75">
                Config: {Object.entries(modelConfig.config)
                  .slice(0, 3)
                  .map(([k, v]) => `${k}=${v}`)
                  .join(", ")}
                {Object.keys(modelConfig.config).length > 3 && " ..."}
              </div>
            )}
          </div>
        ) : (
          <span className="text-amber-500">No model selected</span>
        )}
      </ReviewSection>

      {/* Training Parameters */}
      <ReviewSection title="Training Parameters" valid={true}>
        <div className="grid grid-cols-2 gap-x-4 gap-y-1 text-xs">
          <div>Batch size: {trainingParams.batchSize}</div>
          <div>Learning rate: {trainingParams.learningRate}</div>
          <div>Epochs: {trainingParams.epochs}</div>
          <div>Max steps: {trainingParams.maxSteps || "full run"}</div>
          <div>Scheduler: {trainingParams.lrScheduler}</div>
          <div>Warmup: {trainingParams.warmupSteps} steps</div>
          <div>Checkpoints: every {trainingParams.checkpointInterval} epochs</div>
        </div>
        {trainingParams.runName && (
          <div className="mt-1 text-xs">
            Run name: <span className="font-mono">{trainingParams.runName}</span>
          </div>
        )}
      </ReviewSection>

      {/* Experiment Tracking */}
      <ReviewSection title="Experiment Tracking" valid={true}>
        <div>
          {getTrackerDisplayName(trackerConfig.type)}
          {trackerConfig.type === "mlflow" && trackerConfig.trackingUri && (
            <div className="text-xs opacity-75 mt-1">
              URI: {trackerConfig.trackingUri}
            </div>
          )}
          {trackerConfig.type === "wandb" && trackerConfig.project && (
            <div className="text-xs opacity-75 mt-1">
              Project: {trackerConfig.entity ? `${trackerConfig.entity}/` : ""}{trackerConfig.project}
            </div>
          )}
        </div>
      </ReviewSection>

      {/* Compute */}
      <ReviewSection title="Compute" valid={!computeBlockReason}>
        <div className="space-y-1">
          <div>{TRAINING_COMPUTE_BACKEND_NAMES[computeConfig.type]}</div>
          <div className="text-xs opacity-75">Device: {computeConfig.device}</div>
          {computeConfig.type === "ssh" ? (
            <div className="text-xs opacity-75">
              SSH: {computeConfig.sshUser || "user"}@{computeConfig.sshHost || "host"}:
              {computeConfig.sshPort || 22}
              <br />
              Image: <span className="font-mono">{computeConfig.dockerImage || "urdf-ops:training"}</span>
            </div>
          ) : null}
          {computeBlockReason ? (
            <div className="text-xs text-amber-600">{computeBlockReason}</div>
          ) : null}
        </div>
      </ReviewSection>

      {/* Warnings */}
      {computeBlockReason ? (
        <div className="p-3 bg-amber-500/10 border border-amber-500/20 rounded-lg">
          <div className="flex items-start gap-2">
            <AlertCircle className="w-4 h-4 text-amber-500 mt-0.5" />
            <div>
              <div className="text-sm font-medium text-amber-600">Compute Unavailable</div>
              <div className="text-xs text-muted-foreground">
                {computeBlockReason}
              </div>
            </div>
          </div>
        </div>
      ) : null}

      {/* Ready to start */}
      {datasetConfig && modelConfig && canStartCompute && (
        <div className="p-3 bg-green-500/10 border border-green-500/20 rounded-lg">
          <div className="flex items-start gap-2">
            <CheckCircle2 className="w-4 h-4 text-green-500 mt-0.5" />
            <div>
              <div className="text-sm font-medium text-green-600">Ready to Start</div>
              <div className="text-xs text-muted-foreground">
                All required configuration is complete. Click "Start Training" to begin.
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
