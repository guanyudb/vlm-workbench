import { useState } from "react";
import {
  Activity, CheckCircle2, ExternalLink, Film, Loader2, Rocket, Sparkles,
  Tag, Wand2, X, XCircle,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle,
} from "@/components/ui/dialog";
import { Badge } from "@/components/ui/badge";
import {
  useRunsTracker, isRunActive, isRunFailed, elapsedFor, type RunRow,
} from "@/lib/runs-tracker";
import { cn } from "@/lib/utils";

const KIND_META: Record<RunRow["kind"], { label: string; icon: any }> = {
  ingest:            { label: "Ingest",          icon: Film },
  cache:             { label: "Model cache",     icon: Activity },
  optimize:          { label: "Optimize",        icon: Wand2 },
  train:             { label: "Train",           icon: Sparkles },
  repin:             { label: "Re-pin",          icon: Tag },
  "local-inference": { label: "Local inference", icon: Rocket },
  other:             { label: "Run",             icon: Activity },
};

/** Navbar pill listing all workbench job runs (in-flight + recent).
 * Click → drawer with each run, its state, elapsed time, and a link to
 * the Databricks run page. Generic replacement for the optimize-only
 * indicator we shipped first — same look, broader scope. */
export function RunsIndicator() {
  const { runs, dismissed, dismiss, refresh } = useRunsTracker();
  const [open, setOpen] = useState(false);

  const visible = runs.filter((r) => !dismissed.has(r.run_id));
  const running = visible.filter(isRunActive);
  const finished = visible.filter((r) => !isRunActive(r));

  if (visible.length === 0) return null;

  return (
    <>
      <button
        onClick={() => setOpen(true)}
        className={cn(
          "flex items-center gap-1.5 rounded-md border px-2 py-1 text-xs transition-colors hover:bg-accent",
          running.length > 0 ? "border-primary/40" : "border-border",
        )}
        title="Show jobs"
      >
        {running.length > 0 ? (
          <Loader2 className="size-3 animate-spin text-primary" />
        ) : (
          <Activity className="size-3" />
        )}
        <span>
          {running.length > 0
            ? `${running.length} running${finished.length ? ` · ${finished.length} done` : ""}`
            : `${finished.length} run${finished.length === 1 ? "" : "s"}`}
        </span>
      </button>

      <Dialog open={open} onOpenChange={setOpen}>
        <DialogContent className="max-w-3xl">
          <DialogHeader>
            <div className="flex items-center justify-between">
              <div>
                <DialogTitle>Workbench jobs</DialogTitle>
                <DialogDescription>
                  Every Databricks Job submitted by the App SP in the last 24 hours. Polls every 15s.
                </DialogDescription>
              </div>
              <Button variant="outline" size="sm" onClick={refresh}>Refresh</Button>
            </div>
          </DialogHeader>

          {visible.length === 0 ? (
            <p className="text-sm text-muted-foreground">No recent runs.</p>
          ) : (
            <div className="max-h-[60vh] overflow-y-auto divide-y rounded-md border">
              {[...running, ...finished].map((r) => (
                <RunRowItem key={r.run_id} run={r} onDismiss={() => dismiss(r.run_id)} />
              ))}
            </div>
          )}
        </DialogContent>
      </Dialog>
    </>
  );
}

function RunRowItem({ run, onDismiss }: { run: RunRow; onDismiss: () => void }) {
  const meta = KIND_META[run.kind] || KIND_META.other;
  const Icon = meta.icon;
  const active = isRunActive(run);
  const failed = isRunFailed(run);
  return (
    <div className="flex items-start gap-3 px-3 py-2 text-sm">
      <Icon className="mt-0.5 size-4 shrink-0 text-muted-foreground" />
      <div className="min-w-0 flex-1">
        <div className="flex flex-wrap items-center gap-2">
          <span className="font-medium">{meta.label}</span>
          <code className="rounded bg-muted px-1.5 py-0.5 text-[10px] text-muted-foreground">{run.name}</code>
          {active ? (
            <Badge variant="secondary" className="gap-1 font-normal">
              <Loader2 className="size-3 animate-spin" />
              {run.life_cycle_state.toLowerCase()}
            </Badge>
          ) : failed ? (
            <Badge variant="destructive" className="gap-1 font-normal">
              <XCircle className="size-3" />
              {run.result_state.toLowerCase() || "failed"}
            </Badge>
          ) : (
            <Badge variant="default" className="gap-1 font-normal bg-emerald-600 hover:bg-emerald-600">
              <CheckCircle2 className="size-3" />
              {run.result_state.toLowerCase() || "succeeded"}
            </Badge>
          )}
        </div>
        {run.state_message && (
          <p className="mt-0.5 truncate text-xs text-muted-foreground">{run.state_message}</p>
        )}
        <div className="mt-1 flex flex-wrap items-center gap-3 text-[11px] text-muted-foreground">
          <span>elapsed {elapsedFor(run)}</span>
          {run.run_url && (
            <a
              href={run.run_url}
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center gap-1 text-primary hover:underline"
            >
              <ExternalLink className="size-3" /> open run
            </a>
          )}
        </div>
      </div>
      {!active && (
        <button
          onClick={onDismiss}
          className="text-muted-foreground hover:text-foreground"
          title="Dismiss"
        >
          <X className="size-3.5" />
        </button>
      )}
    </div>
  );
}
