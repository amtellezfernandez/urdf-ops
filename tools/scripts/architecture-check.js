#!/usr/bin/env node

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const root = path.resolve(__dirname, "..", "..");

const readFile = (relativePath) =>
  fs.readFileSync(path.join(root, relativePath), "utf8");

const ensure = (condition, message, errors) => {
  if (!condition) errors.push(message);
};

const ensureFileExists = (relativePath, errors) => {
  const absolutePath = path.join(root, relativePath);
  ensure(fs.existsSync(absolutePath), `${relativePath} is missing.`, errors);
};

const MAX_LOC_GLOBAL = 1200;
const MAX_LOC_STRICT = 900;
const STRICT_LOC_PREFIXES = [
  "web/src/app/pages/",
  "web/src/features/layout/",
  "web/src/features/viewer/",
];

const LOC_ALLOWLIST = new Map([
  ["web/src/features/layout/Sidebar.tsx", 7600],
  ["web/src/features/viewer/Viewer3D.tsx", 6977],
  ["web/src/features/dataset/EpisodeViewer3DModal.tsx", 5200],
  ["web/src/features/dataset/FolderUploadScreen.tsx", 5102],
  ["web/src/app/pages/Index.tsx", 3694],
  ["web/src/features/layout/JointListSidebar.tsx", 3653],
  ["web/src/features/layout/page/HealthActionPanel.tsx", 3336],
  ["web/src/features/layout/page/HealthActionPanel.test.tsx", 3082],
  ["web/src/features/urdf/inertia/inertialSynthesis.ts", 2102],
  ["web/src/features/locomotion/approach/approachNavigation.ts", 2093],
  ["web/src/features/urdf/github/githubRepo.ts", 2100],
  ["web/src/features/layout/page/repeatedInertiaSymmetry.ts", 2076],
  ["web/src/features/layout/sidebar/useHfDatasetImportController.ts", 1889],
  ["web/src/features/dataset/JointMappingDialog.tsx", 1800],
  ["web/src/features/viewer/roverApproachBeforeIkSolve.ts", 1826],
  ["web/src/features/viewer/components/WorldObjectEditHandles.tsx", 1729],
  ["web/src/features/dataset/ExportDialog.tsx", 1500],
  ["web/src/features/layout/JointControl.tsx", 1400],
  ["web/src/features/layout/sidebar/sidebarHelpers.ts", 1400],
  ["web/src/features/viewer/useIkSolver.ts", 1386],
  ["web/src/features/dataset/FolderUploadRobotLoader.tsx", 1382],
  ["web/src/features/urdf/github/githubRepo.test.ts", 1335],
  ["web/src/features/layout/panels/EpisodesPanel.tsx", 1204],
  ["web/src/features/layout/page/robotMirrorSymmetryFix.ts", 1140],
  ["web/src/features/viewer/IKDragControls.tsx", 1100],
  ["web/src/features/layout/page/lekiwiSymmetry.probe.test.ts", 961],
  ["web/src/features/layout/sidebar/useLocalDatasetImportController.ts", 947],
]);

const walk = (dir, files = []) => {
  const entries = fs.readdirSync(dir, { withFileTypes: true });
  for (const entry of entries) {
    const full = path.join(dir, entry.name);
    if (entry.isDirectory()) {
      walk(full, files);
    } else {
      files.push(full);
    }
  }
  return files;
};

const toRel = (absolutePath) => path.relative(root, absolutePath).replace(/\\/g, "/");

const isStrictLocPath = (relativePath) =>
  STRICT_LOC_PREFIXES.some((prefix) => relativePath.startsWith(prefix));

const getLocThreshold = (relativePath) => {
  const strictCap = isStrictLocPath(relativePath) ? MAX_LOC_STRICT : MAX_LOC_GLOBAL;
  const allowlisted = LOC_ALLOWLIST.get(relativePath);
  if (typeof allowlisted === "number") {
    return allowlisted;
  }
  return strictCap;
};

const runFileSizeChecks = (errors) => {
  const srcRoot = path.join(root, "web", "src");
  const files = walk(srcRoot)
    .filter((file) => file.endsWith(".ts") || file.endsWith(".tsx"))
    .filter((file) => !file.endsWith(".d.ts"));

  const offenders = [];
  for (const file of files) {
    const relativePath = toRel(file);
    const lineCount = fs.readFileSync(file, "utf8").split("\n").length;
    const threshold = getLocThreshold(relativePath);
    if (lineCount > threshold) {
      offenders.push({ relativePath, lineCount, threshold });
    }
  }

  offenders.sort((a, b) => b.lineCount - a.lineCount);
  offenders.forEach((offender) => {
    errors.push(
      `${offender.relativePath} is too large (${offender.lineCount} LOC > ${offender.threshold} LOC cap).`
    );
  });
};

const run = () => {
  const errors = [];

  const viewerPath = "web/src/features/viewer/Viewer3D.tsx";
  const previewPath = "web/src/features/camera/EpisodeCameraPreview.tsx";
  const sharedLoaderPath = "web/src/features/urdf/runtime/urdfMeshLoader.ts";

  const viewerCode = readFile(viewerPath);
  const previewCode = readFile(previewPath);
  const sharedLoaderCode = readFile(sharedLoaderPath);

  ensure(
    viewerCode.includes("createUrdfMeshLoadCallback("),
    `${viewerPath} must use createUrdfMeshLoadCallback.`,
    errors
  );
  ensure(
    previewCode.includes("createUrdfMeshLoadCallback("),
    `${previewPath} must use createUrdfMeshLoadCallback.`,
    errors
  );

  ensure(
    !/loader\.loadMeshCb\s*=\s*\(/.test(viewerCode),
    `${viewerPath} must not assign inline loader.loadMeshCb callbacks.`,
    errors
  );
  ensure(
    !/loader\.loadMeshCb\s*=\s*\(/.test(previewCode),
    `${previewPath} must not assign inline loader.loadMeshCb callbacks.`,
    errors
  );

  const hasRuntimeMeshLoaderExport = (name) =>
    new RegExp(`export\\s+const\\s+${name}\\s*=`).test(sharedLoaderCode) ||
    new RegExp(`export\\s*\\{[^}]*\\b${name}\\b[^}]*\\}\\s*from\\s*["']@runtime-private/urdf/urdfMeshLoader["']`).test(
      sharedLoaderCode
    );

  ensure(
    hasRuntimeMeshLoaderExport("loadMeshObjectForUrdfReference"),
    `${sharedLoaderPath} must export loadMeshObjectForUrdfReference.`,
    errors
  );
  ensure(
    hasRuntimeMeshLoaderExport("createUrdfMeshLoadCallback"),
    `${sharedLoaderPath} must export createUrdfMeshLoadCallback.`,
    errors
  );

  ensureFileExists("web/src/studio_core/index.ts", errors);
  ensureFileExists("web/src/runtime_engine/index.ts", errors);
  ensureFileExists("web/src/studio_ui/index.ts", errors);

  const viewerHostPath = "web/src/features/layout/page/ViewerHost.tsx";
  const viewerHostCode = readFile(viewerHostPath);
  ensure(
    viewerHostCode.includes("@/runtime_engine/rosviz/session/runtimeSelector"),
    `${viewerHostPath} must consume runtime selector from runtime_engine.`,
    errors
  );
  ensure(
    viewerHostCode.includes("@/studio_ui/rosviz/RosVizV2Viewer"),
    `${viewerHostPath} must load RosViz viewer from studio_ui.`,
    errors
  );

  const runtimeHealthWrapperPath = "web/src/features/layout/panels/RuntimeHealthPanel.tsx";
  const runtimeHealthWrapperCode = readFile(runtimeHealthWrapperPath);
  ensure(
    runtimeHealthWrapperCode.includes("@/studio_ui/panels/RuntimeHealthPanel"),
    `${runtimeHealthWrapperPath} must re-export RuntimeHealthPanel from studio_ui.`,
    errors
  );

  const runtimeHealthPanelPath = "web/src/studio_ui/panels/RuntimeHealthPanel.tsx";
  const runtimeHealthPanelCode = readFile(runtimeHealthPanelPath);
  ensure(
    runtimeHealthPanelCode.includes("@/runtime_engine/rosviz/state/runtimeHealthStore"),
    `${runtimeHealthPanelPath} must read runtime health state from runtime_engine.`,
    errors
  );

  const rosVizViewerPath = "web/src/studio_ui/rosviz/RosVizV2Viewer.tsx";
  const rosVizViewerCode = readFile(rosVizViewerPath);
  ensure(
    rosVizViewerCode.includes("@/runtime_engine/rosviz/state/runtimeHealthStore"),
    `${rosVizViewerPath} must read runtime health state from runtime_engine.`,
    errors
  );

  runFileSizeChecks(errors);

  if (errors.length > 0) {
    console.error("Architecture check failed:");
    errors.forEach((error) => console.error(`  - ${error}`));
    process.exit(1);
  }

  console.log("Architecture check passed.");
};

run();
