export const JOB_LIST_PARAMS = {
  activeJobsPollIntervalMs: 5_000,
  jobIdPreviewLength: 8,
  millisecondsPerMinute: 60_000,
  minutesPerHour: 60,
  queryStaleTimeMs: 10_000,
} as const;

export const JOB_LIST_CLASS_NAMES = {
  shell: "flex h-full flex-col",
  header: "px-3 pt-3",
  headerRow: "mb-3 flex items-center justify-between",
  title: "text-base font-semibold",
  refreshButton:
    "h-7 rounded-md border-border/60 bg-background/70 px-2.5 text-xs text-muted-foreground hover:bg-muted/30 hover:text-foreground",
  filterBar: "mb-3 flex flex-wrap items-center gap-2",
  searchField: "relative min-w-[200px] max-w-sm flex-1",
  searchInput: "h-8 pl-8 text-xs",
  selectTrigger: "h-8 text-xs",
  row:
    "cursor-pointer border-b border-border/50 transition-colors hover:bg-muted/25",
  rowSelected: "bg-muted/35",
  cell: "px-3 py-2.5",
  tableHeader: "sticky top-0 bg-muted/20",
  tableHeaderRow: "text-left text-xs text-muted-foreground",
  tableHeaderCell: "px-3 py-2.5 font-medium",
  emptyIcon:
    "mb-3 flex h-10 w-10 items-center justify-center rounded-md border border-border/50 bg-muted/25",
  progressTrack: "h-1.5 overflow-hidden rounded-full bg-muted",
  progressIndicator: "h-full bg-sky-500 transition-all duration-300",
} as const;
