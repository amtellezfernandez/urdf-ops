/**
 * Zustand store for training feature state management.
 */

import { create } from "zustand";
import {
  DEFAULT_COMPUTE_CONFIG,
  DEFAULT_TRACKER_CONFIG,
  DEFAULT_TRAINING_PARAMS,
  type ComputeConfig,
  type DatasetConfig,
  type ModelConfig,
  type TrackerConfig,
  type TrainingParams,
  type TrainingStatusResponse,
} from "./types";
import { canUseConfiguredComputeBackend } from "./trainingComputePolicy";

// ============================================================================
// Types
// ============================================================================

type TrainingStep =
  | "model"
  | "training"
  | "tracker"
  | "compute"
  | "review";

interface TrainingState {
  // Dialog state
  isDialogOpen: boolean;
  currentStep: TrainingStep;

  // Configuration
  datasetConfig: DatasetConfig | null;
  modelConfig: ModelConfig | null;
  trainingParams: TrainingParams;
  trackerConfig: TrackerConfig;
  computeConfig: ComputeConfig;

  // Job state
  activeJobId: string | null;
  jobStatus: TrainingStatusResponse | null;
  isPolling: boolean;
  pollIntervalId: number | null;

  // UI state
  isSubmitting: boolean;
  error: string | null;

  // Actions
  openDialog: () => void;
  closeDialog: () => void;
  setStep: (step: TrainingStep) => void;
  nextStep: () => void;
  prevStep: () => void;

  // Config setters
  setDatasetConfig: (config: DatasetConfig) => void;
  setModelConfig: (config: ModelConfig) => void;
  setTrainingParams: (params: Partial<TrainingParams>) => void;
  setTrackerConfig: (config: Partial<TrackerConfig>) => void;
  setComputeConfig: (config: Partial<ComputeConfig>) => void;

  // Job actions
  setActiveJobId: (jobId: string | null) => void;
  setJobStatus: (status: TrainingStatusResponse | null) => void;
  setIsPolling: (polling: boolean) => void;
  setPollIntervalId: (id: number | null) => void;

  // UI actions
  setIsSubmitting: (submitting: boolean) => void;
  setError: (error: string | null) => void;

  // Reset
  resetConfig: () => void;
  resetAll: () => void;
}

// ============================================================================
// Default Values
// ============================================================================

const defaultTrainingParams: TrainingParams = { ...DEFAULT_TRAINING_PARAMS };
const defaultTrackerConfig: TrackerConfig = { ...DEFAULT_TRACKER_CONFIG };
const defaultComputeConfig: ComputeConfig = { ...DEFAULT_COMPUTE_CONFIG };

// ============================================================================
// Step Order
// ============================================================================

const STEP_ORDER: TrainingStep[] = [
  "model",
  "training",
  "tracker",
  "compute",
  "review",
];

// ============================================================================
// Store
// ============================================================================

export const useTrainingStore = create<TrainingState>((set, get) => ({
  // Initial state
  isDialogOpen: false,
  currentStep: "model",

  datasetConfig: null,
  modelConfig: null,
  trainingParams: { ...defaultTrainingParams },
  trackerConfig: { ...defaultTrackerConfig },
  computeConfig: { ...defaultComputeConfig },

  activeJobId: null,
  jobStatus: null,
  isPolling: false,
  pollIntervalId: null,

  isSubmitting: false,
  error: null,

  // Dialog actions
  openDialog: () => set({ isDialogOpen: true, currentStep: "model" }),
  closeDialog: () => {
    const { pollIntervalId } = get();
    if (pollIntervalId) {
      clearInterval(pollIntervalId);
    }
    set({
      isDialogOpen: false,
      isPolling: false,
      pollIntervalId: null,
    });
  },

  // Step navigation
  setStep: (step) => set({ currentStep: step }),

  nextStep: () => {
    const { currentStep } = get();
    const currentIndex = STEP_ORDER.indexOf(currentStep);
    if (currentIndex < STEP_ORDER.length - 1) {
      set({ currentStep: STEP_ORDER[currentIndex + 1] });
    }
  },

  prevStep: () => {
    const { currentStep } = get();
    const currentIndex = STEP_ORDER.indexOf(currentStep);
    if (currentIndex > 0) {
      set({ currentStep: STEP_ORDER[currentIndex - 1] });
    }
  },

  // Config setters
  setDatasetConfig: (config) => set({ datasetConfig: config, error: null }),

  setModelConfig: (config) => set({ modelConfig: config, error: null }),

  setTrainingParams: (params) =>
    set((state) => ({
      trainingParams: { ...state.trainingParams, ...params },
      error: null,
    })),

  setTrackerConfig: (config) =>
    set((state) => ({
      trackerConfig: { ...state.trackerConfig, ...config },
      error: null,
    })),

  setComputeConfig: (config) =>
    set((state) => ({
      computeConfig: { ...state.computeConfig, ...config },
      error: null,
    })),

  // Job actions
  setActiveJobId: (jobId) => set({ activeJobId: jobId }),
  setJobStatus: (status) => set({ jobStatus: status }),
  setIsPolling: (polling) => set({ isPolling: polling }),
  setPollIntervalId: (id) => set({ pollIntervalId: id }),

  // UI actions
  setIsSubmitting: (submitting) => set({ isSubmitting: submitting }),
  setError: (error) => set({ error }),

  // Reset
  resetConfig: () =>
    set((state) => ({
      datasetConfig: state.datasetConfig,
      modelConfig: null,
      trainingParams: { ...defaultTrainingParams },
      trackerConfig: { ...defaultTrackerConfig },
      computeConfig: { ...defaultComputeConfig },
      currentStep: "model",
      error: null,
    })),

  resetAll: () => {
    const { pollIntervalId } = get();
    if (pollIntervalId) {
      clearInterval(pollIntervalId);
    }
    set({
      isDialogOpen: false,
      currentStep: "model",
      datasetConfig: null,
      modelConfig: null,
      trainingParams: { ...defaultTrainingParams },
      trackerConfig: { ...defaultTrackerConfig },
      computeConfig: { ...defaultComputeConfig },
      activeJobId: null,
      jobStatus: null,
      isPolling: false,
      pollIntervalId: null,
      isSubmitting: false,
      error: null,
    });
  },
}));

// ============================================================================
// Selectors
// ============================================================================

const selectIsConfigComplete = (state: TrainingState): boolean => {
  return state.datasetConfig !== null && state.modelConfig !== null;
};

export const selectCanStartTraining = (state: TrainingState): boolean => {
  return (
    selectIsConfigComplete(state) &&
    canUseConfiguredComputeBackend(state.computeConfig) &&
    !state.isSubmitting &&
    state.activeJobId === null
  );
};

export const selectIsJobRunning = (state: TrainingState): boolean => {
  const status = state.jobStatus?.status;
  return status === "pending" || status === "queued" || status === "running";
};
