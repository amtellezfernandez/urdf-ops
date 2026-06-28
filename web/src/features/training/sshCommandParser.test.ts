import { describe, expect, it } from "vitest";

import { parseSshCommand } from "./sshCommandParser";

describe("parseSshCommand", () => {
  it("parses a RunPod SSH command", () => {
    expect(parseSshCommand("ssh 4clkaznp2byq60-64411f1f@ssh.runpod.io")).toEqual({
      user: "4clkaznp2byq60-64411f1f",
      host: "ssh.runpod.io",
    });
  });

  it("parses explicit ports", () => {
    expect(parseSshCommand("ssh -p 22022 ubuntu@gpu.example.com")).toEqual({
      user: "ubuntu",
      host: "gpu.example.com",
      port: 22022,
    });
  });

  it("parses compact port and user option forms", () => {
    expect(parseSshCommand("ssh -p22022 -l ubuntu gpu.example.com")).toEqual({
      user: "ubuntu",
      host: "gpu.example.com",
      port: 22022,
    });
  });

  it("returns null for empty input", () => {
    expect(parseSshCommand("")).toBeNull();
  });
});
