import { createContext, useCallback, useContext, useMemo, useState } from "react";
import { api } from "@/api";
import { usePolling } from "@/lib/use-polling";
import { useToast } from "@/lib/toast-context";

// One row per workbench-submitted Databricks Job run. Derived entirely
// from /api/runs/active so we don't need every tab to register manually
// — the backend filters Jobs runs by SP + run_name prefix and reports the
// state authoritatively. Tab-specific UIs (Playground's optimize, Train's
// progress, Library's ingest badge) still listen via this hook, but they
// no longer have to push state in: the tracker is read-only-from-frontend.
export interface RunRow {
  run_id: number;
  name: string;
  kind: "ingest" | "cache" | "optimize" | "train" | "repin" | "local-inference" | "other";
  life_cycle_state: string;
  result_state: string;
  state_message: string | null;
  error_detail?: string;
  start_time: number;
  end_time: number;
  run_url: string | null;
}

interface TrackerCtx {
  runs: RunRow[];
  loading: boolean;
  refresh: () => void;
  // Per-run dismissal so completed jobs don't pile up in the popover after
  // the user has acknowledged them. Persisted in localStorage so reloads
  // don't re-surface old runs.
  dismissed: Set<number>;
  dismiss: (run_id: number) => void;
}

const Ctx = createContext<TrackerCtx | null>(null);

const DISMISSED_KEY = "vlmwb.runs.dismissed.v1";

function loadDismissed(): Set<number> {
  try {
    const raw = localStorage.getItem(DISMISSED_KEY);
    return raw ? new Set(JSON.parse(raw) as number[]) : new Set();
  } catch {
    return new Set();
  }
}

function saveDismissed(s: Set<number>) {
  try {
    localStorage.setItem(DISMISSED_KEY, JSON.stringify(Array.from(s).slice(-100)));
  } catch { /* quota — silent */ }
}

export function RunsTrackerProvider({ children }: { children: React.ReactNode }) {
  const [runs, setRuns] = useState<RunRow[]>([]);
  const [loading, setLoading] = useState(true);
  const [dismissed, setDismissed] = useState<Set<number>>(() => loadDismissed());
  const [prevRuns, setPrevRuns] = useState<Map<number, RunRow>>(new Map());
  const { addToast } = useToast();

  // Check if any runs are active (not terminal)
  const hasActive = runs.some(isRunActive);

  // Use 15s when active, 60s when idle
  const interval = hasActive ? 15000 : 60000;

  const fetchOnce = useCallback(async () => {
    try {
      const r = await api.listRuns();

      // Check for transitions: active → completed/failed
      const newPrevMap = new Map<number, RunRow>();
      for (const run of r.runs) {
        const prev = prevRuns.get(run.run_id);
        newPrevMap.set(run.run_id, run);

        if (prev && isRunActive(prev) && !isRunActive(run)) {
          // Run just finished
          const variant = isRunFailed(run) ? "destructive" : "success";
          const title = isRunFailed(run)
            ? `${run.kind} run failed: ${run.name}`
            : `${run.kind} run completed: ${run.name}`;
          const description = run.state_message || (run.error_detail ? `Error: ${run.error_detail}` : undefined);
          addToast({
            title,
            description,
            variant,
            duration: 6000,
          });
        }
      }

      setRuns(r.runs);
      setPrevRuns(newPrevMap);
    } catch {
      /* silent — header pill just stays at its last state */
    } finally {
      setLoading(false);
    }
  }, [prevRuns, addToast]);

  // Use polling hook with dynamic interval
  usePolling(fetchOnce, interval, () => false);

  const dismiss = useCallback((run_id: number) => {
    setDismissed((cur) => {
      const next = new Set(cur);
      next.add(run_id);
      saveDismissed(next);
      return next;
    });
  }, []);

  const value = useMemo(() => ({
    runs, loading, refresh: fetchOnce, dismissed, dismiss,
  }), [runs, loading, fetchOnce, dismissed, dismiss]);

  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}

export function useRunsTracker(): TrackerCtx {
  const ctx = useContext(Ctx);
  if (!ctx) throw new Error("useRunsTracker must be used inside RunsTrackerProvider");
  return ctx;
}

// Convenience: a run is "active" if it's not in a terminal state.
export function isRunActive(r: RunRow): boolean {
  return !["TERMINATED", "INTERNAL_ERROR", "SKIPPED", "BLOCKED"].includes(r.life_cycle_state);
}

export function isRunFailed(r: RunRow): boolean {
  if (isRunActive(r)) return false;
  return r.result_state === "FAILED" || r.life_cycle_state === "INTERNAL_ERROR";
}

export function elapsedFor(r: RunRow): string {
  const end = r.end_time || Date.now();
  const sec = Math.max(0, Math.round((end - r.start_time) / 1000));
  if (sec < 60) return `${sec}s`;
  const m = Math.floor(sec / 60);
  const s = sec % 60;
  return `${m}m ${s}s`;
}
