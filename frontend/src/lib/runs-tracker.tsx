import { createContext, useCallback, useContext, useEffect, useMemo, useState } from "react";
import { api } from "@/api";

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

const POLL_MS = 15000;

export function RunsTrackerProvider({ children }: { children: React.ReactNode }) {
  const [runs, setRuns] = useState<RunRow[]>([]);
  const [loading, setLoading] = useState(true);
  const [dismissed, setDismissed] = useState<Set<number>>(() => loadDismissed());

  const fetchOnce = useCallback(async () => {
    try {
      const r = await api.listRuns();
      setRuns(r.runs);
    } catch {
      /* silent — header pill just stays at its last state */
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchOnce();
    const t = setInterval(fetchOnce, POLL_MS);
    return () => clearInterval(t);
  }, [fetchOnce]);

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
