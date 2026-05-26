import { useEffect, useState } from "react";
import { CheckCircle2, ExternalLink, Loader2, Wand2, X, XCircle } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Badge } from "@/components/ui/badge";
import { useOptimizeTracker, type OptimizeJob } from "@/lib/optimize-tracker";
import { cn } from "@/lib/utils";

function elapsed(j: OptimizeJob): string {
  const end = j.finished_at ?? Date.now();
  const sec = Math.max(0, Math.round((end - j.started_at) / 1000));
  if (sec < 60) return `${sec}s`;
  const m = Math.floor(sec / 60);
  const s = sec % 60;
  return `${m}m ${s}s`;
}

/** Sticky pill in the navbar that shows in-flight + recently-completed
 * optimize runs. Click → drawer with details and "apply" CTAs.
 *
 * When a job transitions to succeeded/failed, fires a one-time toast so the
 * user notices without staring at the dialog.
 */
export function OptimizeStatusIndicator({
  onApplyPrompt,
}: {
  onApplyPrompt?: (prompt: string) => void;
}) {
  const { jobs, dismissJob, markNotified } = useOptimizeTracker();
  const [open, setOpen] = useState(false);
  const [toast, setToast] = useState<OptimizeJob | null>(null);

  // Detect newly-finished jobs and fire a toast for each, exactly once.
  useEffect(() => {
    const fresh = jobs.find((j) => (j.state === "succeeded" || j.state === "failed") && !j.notified);
    if (!fresh) return;
    setToast(fresh);
    markNotified(fresh.run_id);
    const t = setTimeout(() => setToast((cur) => (cur?.run_id === fresh.run_id ? null : cur)), 10000);
    return () => clearTimeout(t);
  }, [jobs, markNotified]);

  const running = jobs.filter((j) => j.state === "running");
  const recent = jobs.filter((j) => j.state !== "running").slice(0, 5);

  if (jobs.length === 0) return null;

  return (
    <>
      <button
        onClick={() => setOpen(true)}
        className={cn(
          "flex items-center gap-1.5 rounded-md border px-2 py-1 text-xs transition-colors hover:bg-accent",
          running.length > 0 ? "border-primary/40" : "border-border"
        )}
        title="Show optimize jobs"
      >
        {running.length > 0 ? (
          <Loader2 className="size-3 animate-spin text-primary" />
        ) : (
          <Wand2 className="size-3" />
        )}
        <span>
          {running.length > 0
            ? `${running.length} optimize${running.length === 1 ? "" : "s"} running`
            : `${recent.length} optimize result${recent.length === 1 ? "" : "s"}`}
        </span>
      </button>

      <Dialog open={open} onOpenChange={setOpen}>
        <DialogContent className="max-w-3xl">
          <DialogHeader>
            <DialogTitle>Optimize jobs</DialogTitle>
            <DialogDescription>
              Polls in the background — close this panel and keep working. We'll pop a toast when each job finishes.
            </DialogDescription>
          </DialogHeader>
          <div className="max-h-[60vh] space-y-3 overflow-y-auto">
            {running.map((j) => (
              <JobCard key={j.run_id} job={j} onDismiss={() => dismissJob(j.run_id)} />
            ))}
            {recent.map((j) => (
              <JobCard
                key={j.run_id}
                job={j}
                onDismiss={() => dismissJob(j.run_id)}
                onApply={onApplyPrompt && j.best_prompt ? () => { onApplyPrompt(j.best_prompt!); setOpen(false); } : undefined}
              />
            ))}
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setOpen(false)}>Close</Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {toast && (
        <ToastPanel
          job={toast}
          onClose={() => setToast(null)}
          onView={() => { setToast(null); setOpen(true); }}
        />
      )}
    </>
  );
}

function JobCard({
  job,
  onDismiss,
  onApply,
}: {
  job: OptimizeJob;
  onDismiss: () => void;
  onApply?: () => void;
}) {
  const [showPrompt, setShowPrompt] = useState(false);
  return (
    <div className="rounded-md border p-3">
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0 flex-1">
          <div className="flex flex-wrap items-center gap-1.5">
            <Badge variant="outline" className="font-normal">{job.optimizer}</Badge>
            <span className="text-sm font-medium">→ {job.student_model}</span>
            <span className="text-xs text-muted-foreground">
              teacher: {job.teacher_model === "__gold__" ? "gold labels" : job.teacher_model}
            </span>
          </div>
          <div className="mt-1 flex flex-wrap items-center gap-1.5 text-xs">
            {job.state === "running" ? (
              <span className="flex items-center gap-1 text-primary">
                <Loader2 className="size-3 animate-spin" /> running · {elapsed(job)}
              </span>
            ) : job.state === "succeeded" ? (
              <span className="flex items-center gap-1 text-emerald-600 dark:text-emerald-400">
                <CheckCircle2 className="size-3" /> done in {elapsed(job)}
                {typeof job.score === "number" && (
                  <span className="text-muted-foreground">· score {job.score.toFixed(3)}</span>
                )}
              </span>
            ) : (
              <span className="flex items-center gap-1 text-destructive">
                <XCircle className="size-3" /> failed in {elapsed(job)}
              </span>
            )}
            {job.snapshot_name && (
              <span className="text-muted-foreground">· snapshot {job.snapshot_name}</span>
            )}
          </div>
          {job.state === "failed" && job.state_message && (
            <p className="mt-1 truncate text-xs text-muted-foreground" title={job.state_message}>
              {job.state_message}
            </p>
          )}
          {job.state === "succeeded" && job.best_prompt && (
            <button
              onClick={() => setShowPrompt((v) => !v)}
              className="mt-1 text-xs text-primary hover:underline"
            >
              {showPrompt ? "Hide" : "Show"} optimized prompt
            </button>
          )}
          {showPrompt && job.best_prompt && (
            <pre className="mt-2 max-h-48 overflow-auto whitespace-pre-wrap rounded border bg-muted/40 p-2 text-xs">
              {job.best_prompt}
            </pre>
          )}
        </div>
        <div className="flex shrink-0 flex-col gap-1.5">
          {job.run_page_url && (
            <a
              href={job.run_page_url}
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center gap-1 text-[11px] text-muted-foreground hover:text-foreground"
            >
              <ExternalLink className="size-3" /> Job UI
            </a>
          )}
          {onApply && (
            <Button size="sm" variant="default" onClick={onApply} className="h-7 px-2 text-xs">
              Apply
            </Button>
          )}
          {job.state !== "running" && (
            <Button size="sm" variant="ghost" onClick={onDismiss} className="h-7 px-2 text-xs">
              Dismiss
            </Button>
          )}
        </div>
      </div>
    </div>
  );
}

function ToastPanel({
  job,
  onClose,
  onView,
}: {
  job: OptimizeJob;
  onClose: () => void;
  onView: () => void;
}) {
  return (
    <div className="fixed bottom-4 right-4 z-50 w-80 max-w-[calc(100vw-2rem)] rounded-lg border bg-card p-3 shadow-lg">
      <div className="flex items-start gap-2">
        <div className="mt-0.5 shrink-0">
          {job.state === "succeeded" ? (
            <CheckCircle2 className="size-4 text-emerald-500" />
          ) : (
            <XCircle className="size-4 text-destructive" />
          )}
        </div>
        <div className="min-w-0 flex-1">
          <p className="text-sm font-medium">
            {job.state === "succeeded" ? "Optimize finished" : "Optimize failed"}
          </p>
          <p className="mt-0.5 text-xs text-muted-foreground">
            {job.optimizer} · {job.student_model.replace("databricks-", "")}
            {typeof job.score === "number" && ` · score ${job.score.toFixed(3)}`}
          </p>
          <div className="mt-2 flex gap-1.5">
            <Button size="sm" variant="default" onClick={onView} className="h-7 px-2 text-xs">
              View
            </Button>
            <Button size="sm" variant="ghost" onClick={onClose} className="h-7 px-2 text-xs">
              Dismiss
            </Button>
          </div>
        </div>
        <button onClick={onClose} className="shrink-0 text-muted-foreground hover:text-foreground">
          <X className="size-3.5" />
        </button>
      </div>
    </div>
  );
}
