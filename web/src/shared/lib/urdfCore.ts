// Studio-local boundary over the public i-love-urdf package.
// Web code should import from here instead of reaching into i-love-urdf directly.

export * from "i-love-urdf";

export {
  buildPackageRootsFromMeshBlobMap,
  getJointLinks,
  normalizeMeshPathForMatch,
  parseJointLimitsFromURDF,
  parseMeshReference,
  parseUrdfDocument,
  parseURDF,
  prettyPrintURDF,
  resolveMeshCandidates,
  resolveMeshBlobFromReference,
  serializeUrdfDocument,
  type JointAxisInfo,
  type JointAxisMap,
  type JointHierarchyNode,
  type JointLimitInfo,
  type JointLimits,
  type LinkData,
  type MeshBounds,
  type OriginData,
} from "./urdfBrowser";
