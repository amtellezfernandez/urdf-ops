import { API_BASE_URL } from "@/shared/config/api";
import type {
  ComputeInstancesResponse,
  TrainingComputeBackendsResponse,
  TrainingPreflightResponse,
  TrainingRuntimeCheckResponse,
} from "./types";

export async function fetchTrainingRuntimeCheck(): Promise<TrainingRuntimeCheckResponse> {
  const response = await fetch(`${API_BASE_URL}/training/runtime-check`);
  if (!response.ok) {
    throw new Error("Failed to check local training runtime");
  }
  return response.json();
}

export async function fetchTrainingComputeBackends(): Promise<TrainingComputeBackendsResponse> {
  const response = await fetch(`${API_BASE_URL}/training/compute/backends`);
  if (!response.ok) {
    throw new Error("Failed to fetch training compute backends");
  }
  return response.json();
}

export async function fetchTrainingComputeInstances(): Promise<ComputeInstancesResponse> {
  const response = await fetch(`${API_BASE_URL}/training/compute/instances`);
  if (!response.ok) {
    throw new Error("Failed to fetch training compute instances");
  }
  return response.json();
}

export async function runTrainingPreflight(payload: unknown): Promise<TrainingPreflightResponse> {
  const response = await fetch(`${API_BASE_URL}/training/preflight`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!response.ok) {
    throw new Error("Failed to run training preflight");
  }
  return response.json();
}
