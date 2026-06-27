import { describe, expect, it } from "vitest";

import { buildUrdfOpsInitialDataset } from "./urdfOpsPageData";

describe("UrdfOps page data", () => {
  it("hydrates a Hugging Face dataset from route query params", () => {
    const dataset = buildUrdfOpsInitialDataset(
      new URLSearchParams("dataset=owner%2Frobotics+dataset&source=huggingface"),
    );

    expect(dataset).toEqual({
      id: "owner/robotics dataset",
      name: "robotics dataset",
      source: "huggingface",
      author: "owner",
      tags: [],
    });
  });

  it("falls back to Hugging Face for unknown dataset sources", () => {
    const dataset = buildUrdfOpsInitialDataset(
      new URLSearchParams("dataset=owner%2Frobotics&source=unexpected"),
    );

    expect(dataset?.source).toBe("huggingface");
  });

  it("hydrates a local Studio dataset path from route query params", () => {
    const dataset = buildUrdfOpsInitialDataset(
      new URLSearchParams(
        "dataset=%2Ftmp%2Furdf-studio-teleop-replays%2Fpick-place-001&source=local",
      ),
    );

    expect(dataset).toEqual({
      id: "/tmp/urdf-studio-teleop-replays/pick-place-001",
      name: "pick-place-001",
      source: "local",
      author: undefined,
      tags: [],
    });
  });

  it("ignores empty dataset query params", () => {
    expect(buildUrdfOpsInitialDataset(new URLSearchParams("dataset=+"))).toBeNull();
  });
});
