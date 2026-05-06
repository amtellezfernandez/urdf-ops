export const URDF_OPS_PAGE_PARAMS = {
  logoAssetPath: "assets/urdf-studio-logo.png",
} as const;

export const URDF_OPS_PAGE_CLASS_NAMES = {
  sidebar:
    "flex h-full flex-col border-r border-border/60 bg-background text-foreground transition-all",
  sidebarHeader:
    "flex items-center justify-between border-b border-border/50 px-3 py-2",
  backButton:
    "h-7 rounded-md bg-transparent px-2 text-muted-foreground hover:bg-muted/25 hover:text-foreground",
  title: "truncate text-sm font-semibold text-foreground",
  navItem:
    "flex w-full items-center gap-2 rounded-md border border-transparent px-2.5 py-1.5 text-xs font-medium text-muted-foreground transition-colors hover:bg-muted/25 hover:text-foreground",
  navItemActive:
    "bg-muted/35 text-foreground",
  navIcon:
    "grid h-5 w-5 place-items-center text-muted-foreground",
  navIconActive: "text-foreground",
  collapsedNavButton:
    "h-8 w-full rounded-md bg-transparent text-muted-foreground hover:bg-muted/25 hover:text-foreground",
  collapseButton:
    "h-7 w-7 rounded-md bg-transparent p-0 text-muted-foreground hover:bg-muted/25 hover:text-foreground",
} as const;
