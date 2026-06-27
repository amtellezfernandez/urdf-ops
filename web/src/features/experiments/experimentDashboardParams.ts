export const EXPERIMENT_DASHBOARD_PARAMS = {
  activeJobsPreviewLimit: 3,
} as const;

export const EXPERIMENT_DASHBOARD_CLASS_NAMES = {
  shell: "flex h-full flex-col bg-background text-foreground",
  header: "flex items-center justify-between gap-3 border-b border-border/50 px-4 py-3 sm:px-5",
  title: "text-base font-semibold tracking-normal",
  subtitle: "text-xs text-muted-foreground",
  actionButton:
    "h-8 rounded-md bg-foreground px-3 text-xs text-background shadow-sm hover:bg-foreground/90",
  trainButton:
    "h-8 rounded-md bg-foreground px-3 text-xs text-background shadow-sm hover:bg-foreground/90",
  content: "flex-1 overflow-hidden p-3 sm:p-5",
  tabsList: "mb-4 h-8 rounded-md bg-muted/20 p-1",
  tabTrigger:
    "h-6 rounded-md px-3 py-1 text-xs data-[state=active]:bg-background data-[state=active]:shadow-sm",
  statsGrid: "grid grid-cols-1 gap-3 md:grid-cols-2 lg:grid-cols-4",
  card: "rounded-md border border-border/60 bg-background/95 p-3 shadow-sm",
  iconTile: "rounded-md bg-muted/30 p-1.5",
  activeJobRow:
    "flex items-center justify-between rounded-md border border-border/40 bg-muted/20 p-2.5",
  splitView: "flex h-[calc(100vh-15rem)] gap-3",
  splitPanel: "overflow-hidden rounded-md border border-border/60 bg-background/95 shadow-sm",
} as const;
