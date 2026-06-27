import { describe, expect, it } from "vitest";

import {
  canStartWithTrainingCompute,
  canUseConfiguredComputeBackend,
  getConfiguredComputeBackendBlockReason,
} from "./trainingComputePolicy";
import { TRAINING_COMPUTE_PARAMS } from "./trainingComputeParams";
import type { ComputeConfig } from "./types";

const TEST_COMPUTE_CONFIGS = {
  local: {
    type: "local",
    device: "cuda",
    useSpot: true,
    timeoutHours: 4,
  } satisfies ComputeConfig,
  runpod: {
    type: "runpod",
    device: "cuda",
    useSpot: true,
    timeoutHours: 4,
    apiKey: "secret",
  } satisfies ComputeConfig,
  ssh: {
    type: "ssh",
    device: "cuda",
    useSpot: true,
    timeoutHours: 4,
    sshHost: "203.0.113.10",
    sshUser: "ubuntu",
  } satisfies ComputeConfig,
  sshMissingHost: {
    type: "ssh",
    device: "cuda",
    useSpot: true,
    timeoutHours: 4,
    sshUser: "ubuntu",
  } satisfies ComputeConfig,
} as const;

describe("training compute policy", () => {
  it("allows local and configured remote Docker production backends", () => {
    expect(canUseConfiguredComputeBackend(TEST_COMPUTE_CONFIGS.local)).toBe(true);
    expect(canUseConfiguredComputeBackend(TEST_COMPUTE_CONFIGS.ssh)).toBe(true);
    expect(canUseConfiguredComputeBackend(TEST_COMPUTE_CONFIGS.sshMissingHost)).toBe(false);
    expect(canUseConfiguredComputeBackend(TEST_COMPUTE_CONFIGS.runpod)).toBe(false);
  });

  it("blocks cloud compute even when credentials are present", () => {
    expect(getConfiguredComputeBackendBlockReason(TEST_COMPUTE_CONFIGS.runpod)).toBe(
      TRAINING_COMPUTE_PARAMS.cloudDisabledMessage,
    );
  });

  it("requires a healthy local runtime before launch", () => {
    expect(
      canStartWithTrainingCompute({
        computeConfig: TEST_COMPUTE_CONFIGS.local,
        localRuntimeAvailable: true,
      }),
    ).toBe(true);
    expect(
      canStartWithTrainingCompute({
        computeConfig: TEST_COMPUTE_CONFIGS.local,
        localRuntimeAvailable: false,
      }),
    ).toBe(false);
    expect(
      canStartWithTrainingCompute({
        computeConfig: TEST_COMPUTE_CONFIGS.ssh,
        localRuntimeAvailable: false,
      }),
    ).toBe(true);
  });
});
