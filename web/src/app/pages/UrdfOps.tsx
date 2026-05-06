import { useState, type ComponentType } from "react";
import { Link, useSearchParams } from "react-router-dom";
import { FlaskConical, BarChart2, Play, ChevronLeft, ArrowLeft } from "lucide-react";

import { Button } from "@/shared/ui/button";
import { cn } from "@/shared/lib/utils";
import { ExperimentDashboard, useExperimentStore } from "@/features/experiments";
import { LossCurve } from "@/features/metrics";
import { EvaluationPanel } from "@/features/evaluation";
import {
  URDF_OPS_QUERY_PARAMS,
  URDF_OPS_TABS,
  buildUrdfOpsTabSearchParams,
  resolveUrdfOpsTab,
  type UrdfOpsTab,
} from "@/shared/config/urdfOpsRoutes";
import { URDF_OPS_PAGE_CLASS_NAMES, URDF_OPS_PAGE_PARAMS } from "./urdfOpsPageParams";

type StandaloneUrdfOpsTab = Extract<UrdfOpsTab, "experiments" | "metrics" | "evaluation">;

interface UrdfOpsNavItem {
  tab: StandaloneUrdfOpsTab;
  label: string;
  Icon: ComponentType<{ className?: string }>;
}

const URDF_OPS_NAV_ITEMS = [
  { tab: URDF_OPS_TABS.experiments, label: "Experiments", Icon: FlaskConical },
  { tab: URDF_OPS_TABS.metrics, label: "Metrics", Icon: BarChart2 },
  { tab: URDF_OPS_TABS.evaluation, label: "Evaluation", Icon: Play },
] satisfies readonly UrdfOpsNavItem[];

const resolveStandaloneTab = (tab: UrdfOpsTab): StandaloneUrdfOpsTab => {
  if (tab === URDF_OPS_TABS.metrics || tab === URDF_OPS_TABS.evaluation) return tab;
  return URDF_OPS_TABS.experiments;
};

function NavItem({
  tab,
  label,
  Icon,
  activeTab,
  collapsed,
  onClick,
}: UrdfOpsNavItem & {
  activeTab: StandaloneUrdfOpsTab;
  collapsed: boolean;
  onClick: (tab: StandaloneUrdfOpsTab) => void;
}) {
  const isActive = activeTab === tab;

  if (collapsed) {
    return (
      <Button
        variant={isActive ? "secondary" : "ghost"}
        size="sm"
        className={cn(
          URDF_OPS_PAGE_CLASS_NAMES.collapsedNavButton,
          isActive && URDF_OPS_PAGE_CLASS_NAMES.navItemActive,
        )}
        onClick={() => onClick(tab)}
        title={label}
        aria-label={label}
      >
        <Icon className="h-4 w-4" />
      </Button>
    );
  }

  return (
    <button
      type="button"
      className={cn(
        URDF_OPS_PAGE_CLASS_NAMES.navItem,
        isActive && URDF_OPS_PAGE_CLASS_NAMES.navItemActive,
      )}
      onClick={() => onClick(tab)}
    >
      <span
        className={cn(
          URDF_OPS_PAGE_CLASS_NAMES.navIcon,
          isActive && URDF_OPS_PAGE_CLASS_NAMES.navIconActive,
        )}
      >
        <Icon className="h-4 w-4" />
      </span>
      {label}
    </button>
  );
}

function Sidebar({
  activeTab,
  onTabChange,
  collapsed,
  onToggleCollapse,
}: {
  activeTab: StandaloneUrdfOpsTab;
  onTabChange: (tab: StandaloneUrdfOpsTab) => void;
  collapsed: boolean;
  onToggleCollapse: () => void;
}) {
  const logoUrl = (import.meta.env.BASE_URL || "/") + URDF_OPS_PAGE_PARAMS.logoAssetPath;

  return (
    <aside className={cn(URDF_OPS_PAGE_CLASS_NAMES.sidebar, collapsed ? "w-16" : "w-64")}>
      <div className={cn(URDF_OPS_PAGE_CLASS_NAMES.sidebarHeader, collapsed && "flex-col gap-2 px-2")}>
        <div className="flex min-w-0 items-center gap-2">
          <Button asChild variant="ghost" size="sm" className={cn(URDF_OPS_PAGE_CLASS_NAMES.backButton, collapsed && "w-full gap-1 px-1")}>
            <Link to="/" title="URDF Ops" aria-label="URDF Ops home">
              <ArrowLeft className="h-4 w-4" />
              <img src={logoUrl} alt="" className="h-5 w-auto object-contain" />
            </Link>
          </Button>
          {!collapsed && <h1 className={URDF_OPS_PAGE_CLASS_NAMES.title}>UrdfOps</h1>}
        </div>
        <Button
          variant="ghost"
          size="sm"
          className={cn(URDF_OPS_PAGE_CLASS_NAMES.collapseButton, collapsed && "w-full")}
          onClick={onToggleCollapse}
          aria-label={collapsed ? "Expand UrdfOps navigation" : "Collapse UrdfOps navigation"}
        >
          <ChevronLeft className={cn("h-4 w-4 transition-transform", collapsed && "rotate-180")} />
        </Button>
      </div>

      <nav className="flex-1 space-y-1 p-2 pt-3">
        {URDF_OPS_NAV_ITEMS.map((item) => (
          <NavItem key={item.tab} {...item} activeTab={activeTab} collapsed={collapsed} onClick={onTabChange} />
        ))}
      </nav>
    </aside>
  );
}

export default function UrdfOps() {
  const [searchParams, setSearchParams] = useSearchParams();
  const activeTab = resolveStandaloneTab(resolveUrdfOpsTab(searchParams.get(URDF_OPS_QUERY_PARAMS.tab)));
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const selectedJobId = useExperimentStore((state) => state.selectedJobId);

  const handleTabChange = (tab: StandaloneUrdfOpsTab) => {
    setSearchParams(buildUrdfOpsTabSearchParams(searchParams, tab), { replace: true });
  };

  return (
    <div className="flex h-screen bg-background text-foreground">
      <Sidebar
        activeTab={activeTab}
        onTabChange={handleTabChange}
        collapsed={sidebarCollapsed}
        onToggleCollapse={() => setSidebarCollapsed(!sidebarCollapsed)}
      />

      <main className="min-w-0 flex-1 overflow-hidden bg-background">
        <div className="h-full overflow-hidden">
          {activeTab === URDF_OPS_TABS.experiments && <ExperimentDashboard />}
          {activeTab === URDF_OPS_TABS.metrics && (
            <div className="h-full overflow-auto p-5">
              <div className="mx-auto max-w-5xl">
                {selectedJobId ? (
                  <LossCurve jobId={selectedJobId} />
                ) : (
                  <div className="rounded-md border border-border/60 bg-background/95 p-8 text-center shadow-sm">
                    <div className="mx-auto mb-4 grid h-12 w-12 place-items-center rounded-md border border-border/60 bg-muted/35 text-muted-foreground">
                      <BarChart2 className="h-6 w-6" />
                    </div>
                    <h2 className="mb-2 text-lg font-medium">No job selected</h2>
                    <p className="mb-4 text-sm text-muted-foreground">Select a training job to view its metrics.</p>
                    <Button onClick={() => handleTabChange(URDF_OPS_TABS.experiments)}>Open jobs</Button>
                  </div>
                )}
              </div>
            </div>
          )}
          {activeTab === URDF_OPS_TABS.evaluation && <EvaluationPanel className="h-full" />}
        </div>
      </main>
    </div>
  );
}
