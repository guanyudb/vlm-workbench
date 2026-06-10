import { ReactNode } from "react";
import { Microscope } from "lucide-react";
import { ModeToggle } from "@/components/apx/mode-toggle";

export function Navbar({ title = "Surgical Vision Workbench", subtitle, right }: {
  title?: string;
  subtitle?: ReactNode;
  right?: ReactNode;
}) {
  // Header spans the full viewport (no max-w-6xl crop) so the title sits at
  // the left edge and the runs-pill / theme toggle sit at the right edge —
  // matches the rest of the app's full-width tab layouts.
  return (
    <header className="sticky top-0 z-50 border-b bg-background/80 backdrop-blur-sm">
      <div className="flex h-14 items-center gap-4 px-4 sm:px-6">
        <div className="flex items-center gap-2.5">
          <div className="flex size-8 items-center justify-center rounded-md bg-primary text-primary-foreground">
            <Microscope className="size-4" />
          </div>
          <div className="flex flex-col leading-tight">
            <span className="text-sm font-semibold tracking-tight">{title}</span>
            {subtitle && <span className="text-[11px] text-muted-foreground">{subtitle}</span>}
          </div>
        </div>
        <div className="flex-1" />
        <div className="flex items-center gap-2">
          {right}
          <ModeToggle />
        </div>
      </div>
    </header>
  );
}
