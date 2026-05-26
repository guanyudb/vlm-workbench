import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState } from "react";
import { api } from "@/api";

export interface OptimizeJob {
  run_id: string;
  databricks_run_id: string;
  optimizer: "gepa" | "dspy";
  student_model: string;
  teacher_model: string;
  snapshot_id: string;
  snapshot_name?: string;
  // Lifecycle tracked client-side so the user sees the run from kick-off
  // → "running" until the backend reports a terminal state.
  state: "running" | "succeeded" | "failed";
  result_state?: string | null;
  state_message?: string | null;
  run_page_url?: string | null;
  best_prompt?: string | null;
  score?: number | null;
  history?: number[] | null;
  started_at: number;
  finished_at?: number;
  // Surfaced when we successfully poll once after completion. We use this as
  // the trigger to fire the toast notification exactly once.
  notified?: boolean;
}

interface TrackerCtx {
  jobs: OptimizeJob[];
  startJob: (job: Omit<OptimizeJob, "state" | "started_at">) => void;
  dismissJob: (run_id: string) => void;
  markNotified: (run_id: string) => void;
}

const Ctx = createContext<TrackerCtx | null>(null);

const STORAGE_KEY = "vlmwb.optimizeJobs.v1";

function loadJobs(): OptimizeJob[] {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    if (!Array.isArray(parsed)) return [];
    return parsed.filter((j) => j && typeof j.run_id === "string");
  } catch {
    return [];
  }
}

function saveJobs(jobs: OptimizeJob[]) {
  try {
    // Keep only last 20 to bound storage
    localStorage.setItem(STORAGE_KEY, JSON.stringify(jobs.slice(0, 20)));
  } catch { /* quota — silent */ }
}

export function OptimizeTrackerProvider({ children }: { children: React.ReactNode }) {
  const [jobs, setJobs] = useState<OptimizeJob[]>(() => loadJobs());
  // Track per-job consecutive error counts so a transient blip doesn't kill the run.
  const errorCountsRef = useRef<Record<string, number>>({});

  const persist = useCallback((next: OptimizeJob[]) => {
    setJobs(next);
    saveJobs(next);
  }, []);

  const startJob = useCallback((job: Omit<OptimizeJob, "state" | "started_at">) => {
    persist([
      { ...job, state: "running", started_at: Date.now() },
      ...jobs.filter((j) => j.run_id !== job.run_id),
    ]);
  }, [jobs, persist]);

  const dismissJob = useCallback((run_id: string) => {
    persist(jobs.filter((j) => j.run_id !== run_id));
  }, [jobs, persist]);

  const markNotified = useCallback((run_id: string) => {
    persist(jobs.map((j) => (j.run_id === run_id ? { ...j, notified: true } : j)));
  }, [jobs, persist]);

  // Background poller — runs while there's at least one in-flight job. Resilient
  // to transient errors (502/network blips): keeps polling unless the same run
  // errors >5 times in a row.
  useEffect(() => {
    const running = jobs.filter((j) => j.state === "running");
    if (running.length === 0) return;

    let stopped = false;
    const poll = async () => {
      if (stopped) return;
      const updates: Record<string, Partial<OptimizeJob>> = {};
      await Promise.all(running.map(async (j) => {
        try {
          const s = await api.optimizeStatus(j.run_id, j.databricks_run_id);
          errorCountsRef.current[j.run_id] = 0;
          const life = s.life_cycle_state;
          if (life === "TERMINATED" || life === "INTERNAL_ERROR" || life === "SKIPPED") {
            const ok = life === "TERMINATED" && s.result_state === "SUCCESS";
            updates[j.run_id] = {
              state: ok ? "succeeded" : "failed",
              result_state: s.result_state,
              state_message: s.state_message,
              run_page_url: s.run_page_url,
              best_prompt: s.best_prompt,
              score: s.score,
              history: s.history,
              finished_at: Date.now(),
            };
          } else {
            updates[j.run_id] = {
              result_state: s.result_state,
              state_message: s.state_message,
              run_page_url: s.run_page_url,
            };
          }
        } catch {
          errorCountsRef.current[j.run_id] = (errorCountsRef.current[j.run_id] ?? 0) + 1;
          if ((errorCountsRef.current[j.run_id] ?? 0) >= 8) {
            updates[j.run_id] = {
              state: "failed",
              state_message: "polling failed repeatedly (proxy 502 / network)",
              finished_at: Date.now(),
            };
          }
        }
      }));
      if (stopped) return;
      if (Object.keys(updates).length > 0) {
        setJobs((cur) => {
          const next = cur.map((j) => updates[j.run_id] ? { ...j, ...updates[j.run_id] } : j);
          saveJobs(next);
          return next;
        });
      }
    };
    poll();
    const interval = setInterval(poll, 8000);
    return () => { stopped = true; clearInterval(interval); };
    // We re-subscribe whenever the list of running jobs changes.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [jobs.filter((j) => j.state === "running").map((j) => j.run_id).join(",")]);

  const value = useMemo<TrackerCtx>(() => ({ jobs, startJob, dismissJob, markNotified }), [jobs, startJob, dismissJob, markNotified]);
  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}

export function useOptimizeTracker(): TrackerCtx {
  const ctx = useContext(Ctx);
  if (!ctx) throw new Error("useOptimizeTracker must be used inside OptimizeTrackerProvider");
  return ctx;
}
