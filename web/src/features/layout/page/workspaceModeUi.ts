import type { WorkspaceMode } from "@/features/workspace/types";

const isAssemblyWorkspaceMode = (workspaceMode: WorkspaceMode) =>
  workspaceMode === "assembly";

export const getWorkspaceModeUiPolicy = (workspaceMode: WorkspaceMode) => {
  const isAssembly = isAssemblyWorkspaceMode(workspaceMode);
  const isStudio = workspaceMode === "studio";
  const isRuntime = workspaceMode === "runtime";

  return {
    isAssembly,
    isStudio,
    isRuntime,
    showAssemblyActions: isAssembly,
    showStudioChrome: !isAssembly,
    showIkPanel: !isAssembly,
    showWorldDialogs: !isAssembly,
    showStudioIssueReport: isStudio,
  };
};
