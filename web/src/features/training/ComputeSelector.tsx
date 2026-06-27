/**
 * Compute backend selection for the training wizard.
 */

import { useEffect, useMemo, useState } from "react";
import { AlertCircle, CheckCircle2, Cloud, Cpu, Info, Loader2, Lock, Server, Zap } from "lucide-react";

import { Label } from "@/shared/ui/label";
import { Input } from "@/shared/ui/input";
import { Button } from "@/shared/ui/button";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/shared/ui/select";
import {
  fetchTrainingComputeBackends,
  fetchTrainingComputeInstances,
  runTrainingPreflight,
} from "./trainingApi";
import { buildTrainingPayload } from "./buildTrainingPayload";
import {
  TRAINING_COMPUTE_BACKENDS,
  TRAINING_COMPUTE_PARAMS,
  TRAINING_LOCAL_DEVICES,
  type TrainingComputeBackendId,
} from "./trainingComputeParams";
import { useTrainingStore } from "./useTrainingStore";
import type {
  ComputeInstanceInfo,
  TrainingComputeBackendCapability,
  TrainingPreflightResponse,
} from "./types";

const COMPUTE_BACKEND_ICONS: Record<TrainingComputeBackendId, typeof Cpu> = {
  local: Cpu,
  ssh: Server,
  modal: Cloud,
  runpod: Zap,
  macrodata: Cloud,
  aws: Server,
};

export function ComputeSelector() {
  const {
    datasetConfig,
    modelConfig,
    trainingParams,
    trackerConfig,
    computeConfig,
    setComputeConfig,
  } = useTrainingStore();

  const [instances, setInstances] = useState<Record<string, ComputeInstanceInfo[]>>({});
  const [backendCapabilities, setBackendCapabilities] = useState<
    Record<string, TrainingComputeBackendCapability>
  >({});
  const [preflightResult, setPreflightResult] = useState<TrainingPreflightResponse | null>(null);
  const [preflightError, setPreflightError] = useState<string | null>(null);
  const [isPreflighting, setIsPreflighting] = useState(false);

  useEffect(() => {
    let cancelled = false;

    async function fetchReadiness() {
      try {
        const [nextInstances, nextBackends] = await Promise.all([
          fetchTrainingComputeInstances(),
          fetchTrainingComputeBackends(),
        ]);
        if (cancelled) {
          return;
        }
        setInstances(nextInstances.instances);
        setBackendCapabilities(
          Object.fromEntries(nextBackends.backends.map((backend) => [backend.type, backend])),
        );
      } catch (error) {
        console.warn("Failed to fetch compute readiness", error);
      }
    }

    void fetchReadiness();

    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    if (!["local", "ssh"].includes(computeConfig.type)) {
      setComputeConfig({ type: "local", apiKey: undefined, gpu: undefined });
    }
  }, [computeConfig.type, setComputeConfig]);

  const selectedBackend = TRAINING_COMPUTE_BACKENDS.find(
    (backend) => backend.id === computeConfig.type,
  );
  const canRunPreflight = Boolean(datasetConfig && modelConfig);
  const localInstanceSummary = useMemo(() => {
    const localInstances = instances.local || [];
    if (localInstances.length === 0) {
      return TRAINING_COMPUTE_PARAMS.localRuntimeReviewMessage;
    }
    return `Detected ${localInstances.map((instance) => instance.name).join(", ")}`;
  }, [instances.local]);

  const handleRunPreflight = async () => {
    if (!datasetConfig || !modelConfig) {
      setPreflightError("Select a dataset and model before running preflight.");
      return;
    }

    setIsPreflighting(true);
    setPreflightError(null);
    setPreflightResult(null);
    try {
      const result = await runTrainingPreflight(
        buildTrainingPayload({
          datasetConfig,
          modelConfig,
          trainingParams,
          trackerConfig,
          computeConfig,
        }),
      );
      setPreflightResult(result);
    } catch (error) {
      setPreflightError(error instanceof Error ? error.message : "Failed to run preflight");
    } finally {
      setIsPreflighting(false);
    }
  };

  return (
    <div className="space-y-4">
      <div className="space-y-2">
        <Label>Compute Backend</Label>
        <div className="grid gap-2">
          {TRAINING_COMPUTE_BACKENDS.map((backend) => {
            const Icon = COMPUTE_BACKEND_ICONS[backend.id];
            const capability = backendCapabilities[backend.id];
            const enabled = backend.enabled && (capability ? capability.enabled : true);
            const reason = capability?.reason || backend.reason;
            const isSelected = backend.selectableType === computeConfig.type;

            return (
              <button
                key={backend.id}
                type="button"
                disabled={!enabled || !backend.selectableType}
                onClick={() => {
                  if (enabled && backend.selectableType) {
                    setComputeConfig({ type: backend.selectableType });
                  }
                }}
                className={`w-full rounded-lg border p-3 text-left transition-colors disabled:cursor-not-allowed disabled:opacity-60 ${
                  isSelected
                    ? "border-primary bg-primary/5"
                    : "border-border hover:border-primary/50 hover:bg-muted/50"
                }`}
              >
                <div className="flex items-start gap-3">
                  <Icon className="mt-0.5 h-5 w-5 text-muted-foreground" />
                  <div className="min-w-0 flex-1">
                    <div className="flex flex-wrap items-center gap-2">
                      <span className="text-sm font-medium">{backend.name}</span>
                      <span
                        className={`rounded border px-1.5 py-0.5 text-[10px] uppercase tracking-wide ${
                          enabled
                            ? "border-emerald-500/30 text-emerald-700"
                            : "border-muted-foreground/25 text-muted-foreground"
                        }`}
                      >
                        {enabled
                          ? backend.id === "ssh"
                            ? TRAINING_COMPUTE_PARAMS.remoteReadyBadge
                            : TRAINING_COMPUTE_PARAMS.localReadyBadge
                          : TRAINING_COMPUTE_PARAMS.cloudUnavailableBadge}
                      </span>
                    </div>
                    <div className="mt-0.5 text-xs text-muted-foreground">{backend.description}</div>
                    {!enabled && reason ? (
                      <div className="mt-2 flex items-start gap-1.5 text-xs text-muted-foreground">
                        <Lock className="mt-0.5 h-3.5 w-3.5 shrink-0" />
                        <span>{reason}</span>
                      </div>
                    ) : null}
                  </div>
                </div>
              </button>
            );
          })}
        </div>
      </div>

      <div className="space-y-3 rounded-lg bg-muted/30 p-4">
        <div className="flex items-center gap-2">
          <Info className="h-4 w-4 text-muted-foreground" />
          <Label>
            {computeConfig.type === "ssh" ? "Remote Docker Configuration" : "Local Configuration"}
          </Label>
        </div>

        <div className="space-y-1">
          <Label className="text-xs">Device</Label>
          <Select
            value={computeConfig.device}
            onValueChange={(device) => setComputeConfig({ device })}
          >
            <SelectTrigger className="h-8 text-sm">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {TRAINING_LOCAL_DEVICES.map((device) => (
                <SelectItem key={device.value} value={device.value}>
                  {device.label}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>

        {computeConfig.type === "ssh" ? (
          <div className="grid grid-cols-2 gap-3">
            <div className="space-y-1">
              <Label htmlFor="training-ssh-host" className="text-xs">Host / IP</Label>
              <Input
                id="training-ssh-host"
                value={computeConfig.sshHost || ""}
                onChange={(event) => setComputeConfig({ sshHost: event.target.value })}
                className="h-8 text-sm"
              />
            </div>
            <div className="space-y-1">
              <Label htmlFor="training-ssh-user" className="text-xs">User</Label>
              <Input
                id="training-ssh-user"
                value={computeConfig.sshUser || ""}
                onChange={(event) => setComputeConfig({ sshUser: event.target.value })}
                className="h-8 text-sm"
              />
            </div>
            <div className="space-y-1">
              <Label htmlFor="training-ssh-port" className="text-xs">Port</Label>
              <Input
                id="training-ssh-port"
                type="number"
                min={1}
                max={65535}
                value={computeConfig.sshPort || 22}
                onChange={(event) =>
                  setComputeConfig({ sshPort: parseInt(event.target.value, 10) || 22 })
                }
                className="h-8 text-sm"
              />
            </div>
            <div className="space-y-1">
              <Label htmlFor="training-ssh-key" className="text-xs">SSH Key Path</Label>
              <Input
                id="training-ssh-key"
                value={computeConfig.sshKeyPath || ""}
                onChange={(event) => setComputeConfig({ sshKeyPath: event.target.value })}
                placeholder="~/.ssh/id_rsa"
                className="h-8 text-sm"
              />
            </div>
            <div className="space-y-1">
              <Label htmlFor="training-ssh-output" className="text-xs">Remote Output</Label>
              <Input
                id="training-ssh-output"
                value={computeConfig.remoteOutputDir || ""}
                onChange={(event) => setComputeConfig({ remoteOutputDir: event.target.value })}
                className="h-8 text-sm"
              />
            </div>
            <div className="space-y-1">
              <Label htmlFor="training-ssh-image" className="text-xs">Trainer Image</Label>
              <Input
                id="training-ssh-image"
                value={computeConfig.dockerImage || ""}
                onChange={(event) => setComputeConfig({ dockerImage: event.target.value })}
                className="h-8 text-sm"
              />
            </div>
            <div className="col-span-2 space-y-1">
              <Label htmlFor="training-ssh-docker-args" className="text-xs">Docker Args</Label>
              <Input
                id="training-ssh-docker-args"
                value={computeConfig.dockerArgs || ""}
                onChange={(event) => setComputeConfig({ dockerArgs: event.target.value })}
                placeholder="--shm-size 8g"
                className="h-8 text-sm"
              />
            </div>
          </div>
        ) : (
          <p className="text-xs text-muted-foreground">{localInstanceSummary}</p>
        )}
      </div>

      <div className="rounded-lg bg-muted/50 p-3">
        <div className="text-xs text-muted-foreground">
          {TRAINING_COMPUTE_PARAMS.selectedComputeLabel}
        </div>
        <div className="mt-1 text-sm font-medium">
          {selectedBackend?.name || "Local GPU"}
          <span className="ml-2 font-normal text-muted-foreground">{computeConfig.device}</span>
        </div>
      </div>

      <div className="space-y-3 rounded-lg border p-3">
        <div className="flex items-center justify-between gap-3">
          <div>
            <div className="text-sm font-medium">Preflight</div>
            <div className="text-xs text-muted-foreground">
              Validate the selected compute path before launch.
            </div>
          </div>
          <Button
            type="button"
            variant="outline"
            size="sm"
            onClick={handleRunPreflight}
            disabled={!canRunPreflight || isPreflighting}
          >
            {isPreflighting ? (
              <Loader2 className="mr-2 h-4 w-4 animate-spin" />
            ) : (
              <CheckCircle2 className="mr-2 h-4 w-4" />
            )}
            Run
          </Button>
        </div>
        {preflightError ? (
          <div className="flex items-start gap-2 text-xs text-amber-600">
            <AlertCircle className="mt-0.5 h-3.5 w-3.5 shrink-0" />
            <span>{preflightError}</span>
          </div>
        ) : null}
        {preflightResult ? (
          <div className="space-y-2">
            <div
              className={`text-xs font-medium ${
                preflightResult.ready ? "text-emerald-700" : "text-amber-700"
              }`}
            >
              {preflightResult.recommendation}
            </div>
            <div className="space-y-1">
              {preflightResult.checks.map((check) => (
                <div key={check.name} className="flex items-start gap-2 text-xs">
                  {check.status === "pass" ? (
                    <CheckCircle2 className="mt-0.5 h-3.5 w-3.5 shrink-0 text-emerald-600" />
                  ) : (
                    <AlertCircle
                      className={`mt-0.5 h-3.5 w-3.5 shrink-0 ${
                        check.status === "warn" ? "text-amber-600" : "text-red-600"
                      }`}
                    />
                  )}
                  <div>
                    <span className="font-medium">{check.label}: </span>
                    <span className="text-muted-foreground">{check.message}</span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        ) : null}
      </div>
    </div>
  );
}
