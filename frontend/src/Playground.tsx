import { useEffect, useMemo, useRef, useState } from "react";
import { Loader2, Play, RefreshCcw, Square, Sparkles, AlertCircle, Check, Save, Film, Trash2, Wand2, ExternalLink } from "lucide-react";
import {
  api,
  type FrameEntry,
  type FrameLabel,
  type LocalModelEntry,
  type ModelEntry,
  type RunResultRow,
  type SnapshotSummary,
} from "@/api";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Checkbox } from "@/components/ui/checkbox";
import { Label } from "@/components/ui/label";
import { ScrollArea } from "@/components/ui/scroll-area";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle, DialogTrigger,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import { useOptimizeTracker } from "@/lib/optimize-tracker";
import { usePersistentState, setStringCodec, mapStringCodec } from "@/lib/use-persistent-state";
import { cn } from "@/lib/utils";

type FrameSource = "extracted" | "eval" | "labeled";

const DEFAULT_PROMPT = `Identify the surgical instrument by its visible visual features:

PROBE vs SHAVER (most common confusion):
- probe = thin, narrow solid metal rod, uniform diameter end-to-end, tip may have a tiny hook or blunt end; NO opening, NO rotating head
- shaver = wider hollow tube with a distinct rectangular or oval side-opening (aspiration window) near the tip; shaft is noticeably thicker than a probe

Other instruments:
- burr = round/spherical or oval abrasive head at tip
- grasper or biter = two metal jaws that open/close at a hinge
- anchor_driver = ribbed or screw-threaded shaft
- electrocautery = smooth wand with flat, angled, or hook-shaped distal tip
- cannula = transparent or yellow plastic hollow tube

Vocabulary: probe, shaver, burr, grasper, biter, suture_passer, anchor_driver, electrocautery, cannula, scissors, drill_guide, trocar, knot_pusher, rasp, other_metal_tool, no_instrument_visible.
Respond with strict JSON: {"instruments": [{"class": "<one of vocab>", "confidence": 0.0-1.0, "evidence": "<short text>"}], "anatomy": "<short text>", "tissue_condition": "<short text>"}`;

const RECOMMENDED_MODELS = new Set([
  "databricks-claude-sonnet-4-5",
  "databricks-claude-haiku-4-5",
  "databricks-gpt-5-2",
  "databricks-llama-4-maverick",
]);

// Accept both shapes:
//   {"instrument": "shaver"}              (Qwen/MedGemma default, Studio's prompt)
//   {"instruments": [{"class": "..."}]}   (the default Playground prompt)
// When the model returned a parseable object but neither field carries an
// instrument name (empty string, null, or both missing), treat that as
// "no_instrument_visible" — a valid vocabulary entry — instead of leaving the
// cell as "unparsed". Returning null reserves that for "the model didn't
// produce parseable JSON at all".
function primaryClass(parsed: any): string | null {
  if (!parsed || typeof parsed !== "object") return null;
  if (typeof parsed.instrument === "string") {
    return parsed.instrument.trim() || "no_instrument_visible";
  }
  const inst = parsed.instruments;
  if (Array.isArray(inst)) {
    if (inst.length === 0) return "no_instrument_visible";
    if (typeof inst[0] === "object") {
      const c = inst[0]?.class;
      if (typeof c === "string") return c.trim() || "no_instrument_visible";
    }
  }
  // Parsed object exists but has no instrument-shaped key — treat as nothing visible.
  return "no_instrument_visible";
}

function confidence(parsed: any): number | null {
  if (!parsed || typeof parsed !== "object") return null;
  if (typeof parsed.confidence === "number") return parsed.confidence;
  const inst = parsed.instruments;
  if (Array.isArray(inst) && inst.length && typeof inst[0] === "object") {
    const c = inst[0]?.confidence;
    if (typeof c === "number") return c;
  }
  return null;
}

export default function Playground({
  onSendToStudio,
  pendingPrompt,
  onConsumePendingPrompt,
}: {
  onSendToStudio?: (snapshotId: string) => void;
  pendingPrompt?: string | null;
  onConsumePendingPrompt?: () => void;
}) {
  const optimizeTracker = useOptimizeTracker();
  // ── data ─────────────────────────────────────────────────────────────
  const [models, setModels] = useState<ModelEntry[] | null>(null);
  const [localModels, setLocalModels] = useState<LocalModelEntry[]>([]);
  const [frames, setFrames] = useState<FrameEntry[] | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  // Ground-truth labels keyed by frame_path. Populated lazily — whenever the
  // visible frame set changes we look up labels for those paths and cache.
  // Used to (a) render the gold-label chip on each frame card, (b) compute
  // accuracy chips in the results matrix.
  const [labels, setLabels] = useState<Map<string, FrameLabel>>(new Map());

  // ── selection state (persists across tab switch and page reload) ─────
  const [frameSource, setFrameSource] = usePersistentState<FrameSource>("vlmwb.pg.frameSource", "eval");
  const [videoFilter, setVideoFilter] = usePersistentState<string>("vlmwb.pg.videoFilter", "__all__");
  // Subsample interval in seconds (default 30). 0 = "show everything".
  const [intervalSec, setIntervalSec] = usePersistentState<number>("vlmwb.pg.intervalSec", 30);
  const [selectedFrames, setSelectedFrames] = usePersistentState<Set<string>>(
    "vlmwb.pg.selectedFrames", new Set(), setStringCodec,
  );
  const [selectedModels, setSelectedModels] = usePersistentState<Set<string>>(
    "vlmwb.pg.selectedModels", new Set(RECOMMENDED_MODELS), setStringCodec,
  );
  // Local (open-weight, serverless GPU) models. These take ~5-15 min cold.
  const [selectedLocalModels, setSelectedLocalModels] = usePersistentState<Set<string>>(
    "vlmwb.pg.selectedLocalModels", new Set(), setStringCodec,
  );
  const [prompt, setPrompt] = usePersistentState<string>("vlmwb.pg.prompt", DEFAULT_PROMPT);
  // Live workspace default — replaces the locally-inlined DEFAULT_PROMPT
  // when /api/task-config is reachable. Used by the "Reset" button so it
  // restores the workspace's current task config, not an old code constant.
  const [workspaceDefaultPrompt, setWorkspaceDefaultPrompt] = useState<string>(DEFAULT_PROMPT);
  useEffect(() => {
    api.getTaskConfig()
      .then((c) => { if (c.rendered_prompt) setWorkspaceDefaultPrompt(c.rendered_prompt); })
      .catch(() => { /* keep code-constant fallback */ });
  }, []);
  // Frame preview / lightbox
  const [previewIdx, setPreviewIdx] = useState<number | null>(null);

  // ── run state ────────────────────────────────────────────────────────
  // `running` is intentionally NOT persisted — a fresh page load is never
  // "in the middle of an HTTP POST"; the AI Gateway run either completed or
  // got cut off. Local GPU jobs survive refresh via `localRuns` below, and
  // the polling effect will flip `running` back on if any are still in
  // flight.
  const [running, setRunning] = useState(false);
  const [results, setResults] = usePersistentState<Map<string, RunResultRow>>(
    "vlmwb.pg.results", new Map(), mapStringCodec<RunResultRow>(),
  );
  const [progress, setProgress] = usePersistentState<{ done: number; total: number } | null>(
    "vlmwb.pg.progress", null,
  );
  const [runError, setRunError] = useState<string | null>(null);
  const [runDuration, setRunDuration] = usePersistentState<number | null>(
    "vlmwb.pg.runDuration", null,
  );
  const cancelRef = useRef<(() => void) | null>(null);

  // Local-model jobs. Keyed by model_name. Persists to localStorage so a
  // page refresh while a GPU job is running picks polling back up where it
  // left off. The poller effect below re-subscribes whenever the set of
  // running run IDs changes (including on first mount).
  type LocalRun = {
    runId: string;
    databricksRunId: string;
    modelName: string;
    framePaths: string[];
    state: "submitting" | "running" | "succeeded" | "failed";
    runPageUrl?: string | null;
    error?: string;
    startedAt: number;
    finishedAt?: number;
  };
  const [localRuns, setLocalRuns] = usePersistentState<Record<string, LocalRun>>(
    "vlmwb.pg.localRuns", {},
  );

  // On mount, scrub any stale "submitting" entries — they can't be resumed
  // because we never got back a databricks_run_id. Mark them failed so the
  // UI shows something instead of an indefinite spinner.
  useEffect(() => {
    setLocalRuns((cur) => {
      let changed = false;
      const next: Record<string, LocalRun> = {};
      for (const [k, v] of Object.entries(cur)) {
        if (v.state === "submitting" && !v.databricksRunId) {
          next[k] = { ...v, state: "failed", error: "submission lost on refresh", finishedAt: Date.now() };
          changed = true;
        } else {
          next[k] = v;
        }
      }
      return changed ? next : cur;
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // ── snapshots ────────────────────────────────────────────────────────
  const [snapshots, setSnapshots] = useState<SnapshotSummary[]>([]);
  const [saveOpen, setSaveOpen] = useState(false);
  const [saveName, setSaveName] = useState("");
  const [saveBestModel, setSaveBestModel] = useState<string>("");
  const [saving, setSaving] = useState(false);
  const [saveError, setSaveError] = useState<string | null>(null);
  const [saveSuccess, setSaveSuccess] = useState<string | null>(null);

  const refreshSnapshots = () => api.listSnapshots(20).then(setSnapshots).catch(() => {});
  useEffect(() => { refreshSnapshots(); }, []);

  // ── optimize ─────────────────────────────────────────────────────────
  const [optimizeOpen, setOptimizeOpen] = useState(false);
  const [optSnapshotId, setOptSnapshotId] = useState<string>("");
  const [optTeacher, setOptTeacher] = useState<string>("databricks-claude-sonnet-4-5");
  const [optStudent, setOptStudent] = useState<string>("databricks-claude-haiku-4-5");
  const [optType, setOptType] = useState<"gepa" | "dspy">("gepa");
  const [optRounds, setOptRounds] = useState<number>(5);
  const [optSubmitting, setOptSubmitting] = useState(false);
  const [optRun, setOptRun] = useState<{
    run_id: string;
    databricks_run_id: string;
    output_dir: string;
    student_model: string;
    teacher_model: string;
    n_eval_examples: number;
    optimizer: string;
  } | null>(null);
  const [optStatus, setOptStatus] = useState<{
    life_cycle_state: string | null;
    result_state: string | null;
    state_message: string | null;
    run_page_url: string | null;
    best_prompt: string | null;
    score: number | null;
    history: number[] | null;
  } | null>(null);
  const [optError, setOptError] = useState<string | null>(null);

  const openOptimizeDialog = (snapshotId?: string) => {
    setOptError(null);
    setOptStatus(null);
    setOptRun(null);
    if (snapshotId) {
      setOptSnapshotId(snapshotId);
      // Try to populate teacher/student from the snapshot's models. If any
      // frame in this snapshot already has a gold label, default the teacher
      // to "Gold labels" — supervised > pseudo-labels whenever we can.
      api.getSnapshot(snapshotId).then(async (snap) => {
        const models = snap.model_names || [];
        try {
          const gold = await api.listLabels(snap.frame_paths);
          if (gold.length > 0) {
            setOptTeacher("__gold__");
            if (snap.best_model && models.includes(snap.best_model)) {
              setOptStudent(snap.best_model);
            } else if (models.length > 0) {
              setOptStudent(models[0]);
            }
            return;
          }
        } catch { /* fall through */ }
        // No gold labels — keep the original model-as-teacher heuristic.
        if (snap.best_model && models.includes(snap.best_model)) {
          setOptTeacher(snap.best_model);
          const other = models.find((m) => m !== snap.best_model);
          if (other) setOptStudent(other);
        } else if (models.length >= 2) {
          setOptTeacher(models[0]);
          setOptStudent(models[1]);
        }
      }).catch(() => {});
    } else if (snapshots.length > 0) {
      setOptSnapshotId(snapshots[0].id);
    }
    setOptimizeOpen(true);
  };

  const submitOptimize = async () => {
    if (!optSnapshotId || !optTeacher || !optStudent) return;
    // When teacher is the gold-labels sentinel, the "different from student"
    // rule doesn't apply — gold labels are independent of any model.
    if (optTeacher !== "__gold__" && optTeacher === optStudent) {
      setOptError("Teacher and student must be different models");
      return;
    }
    setOptSubmitting(true);
    setOptError(null);
    try {
      const r = await api.optimize({
        snapshot_id: optSnapshotId,
        optimizer: optType,
        teacher_model: optTeacher,
        student_model: optStudent,
        n_rounds: optRounds,
      });
      const snap = snapshots.find((s) => s.id === optSnapshotId);
      // Hand the run off to the global tracker. The dialog can be closed —
      // the navbar indicator + toast will surface completion.
      optimizeTracker.startJob({
        run_id: r.run_id,
        databricks_run_id: r.databricks_run_id,
        optimizer: r.optimizer as "gepa" | "dspy",
        student_model: r.student_model,
        teacher_model: r.teacher_model,
        snapshot_id: optSnapshotId,
        snapshot_name: snap?.name,
      });
      setOptRun(r);
      // Auto-close the dialog so the user isn't stuck staring at it. The
      // background poller in OptimizeTrackerProvider takes over.
      setOptimizeOpen(false);
    } catch (e) {
      setOptError((e as Error).message);
    } finally {
      setOptSubmitting(false);
    }
  };

  // The dialog mirrors the live job from the global tracker so the user can
  // re-open it and still see status. Polling itself runs in the provider,
  // surviving dialog close + page navigation.
  const trackedJob = optRun
    ? optimizeTracker.jobs.find((j) => j.run_id === optRun.run_id) ?? null
    : null;
  useEffect(() => {
    if (!trackedJob) return;
    setOptStatus({
      life_cycle_state:
        trackedJob.state === "running" ? "RUNNING" :
        trackedJob.state === "succeeded" ? "TERMINATED" : "INTERNAL_ERROR",
      result_state: trackedJob.result_state ?? null,
      state_message: trackedJob.state_message ?? null,
      run_page_url: trackedJob.run_page_url ?? null,
      best_prompt: trackedJob.best_prompt ?? null,
      score: trackedJob.score ?? null,
      history: trackedJob.history ?? null,
    });
  }, [trackedJob?.state, trackedJob?.best_prompt, trackedJob?.state_message, trackedJob?.run_page_url]);

  const applyOptimizedPrompt = () => {
    if (optStatus?.best_prompt) {
      setPrompt(optStatus.best_prompt);
      setOptimizeOpen(false);
    }
  };

  // Accept a prompt staged from the global indicator (e.g. user clicked
  // Apply on a finished optimize job from a different page).
  useEffect(() => {
    if (pendingPrompt) {
      setPrompt(pendingPrompt);
      onConsumePendingPrompt?.();
    }
  }, [pendingPrompt, onConsumePendingPrompt]);

  // ── load models + frames on mount ────────────────────────────────────
  useEffect(() => {
    api
      .models()
      .then(setModels)
      .catch((e) => setLoadError(`models: ${e.message}`));
    api.localModels().then(setLocalModels).catch(() => setLocalModels([]));
  }, []);

  // Background poller for any local-model jobs still running. Tolerates
  // transient errors (proxy 502 etc.) — only marks failed after 8 in a row.
  const localErrCountsRef = useRef<Record<string, number>>({});
  useEffect(() => {
    const inflight = Object.values(localRuns).filter((r) => r.state === "running" || r.state === "submitting");
    if (inflight.length === 0) return;
    let stopped = false;
    const tick = async () => {
      if (stopped) return;
      await Promise.all(inflight.map(async (r) => {
        if (r.state !== "running") return;
        try {
          const s = await api.localRunStatus(r.runId, r.databricksRunId);
          localErrCountsRef.current[r.runId] = 0;
          if (s.life_cycle_state === "TERMINATED" && s.result_state === "SUCCESS") {
            // Fold rows into the results Map. The notebook writes `frame` as
            // the full Volume path, but the results matrix indexes by basename
            // (`${model}::${basename}`) for parity with AI Gateway rows — so
            // normalize here, otherwise cells stay stuck on "waiting on GPU…"
            // even after results.json lands.
            if (Array.isArray(s.results)) {
              setResults((cur) => {
                const next = new Map(cur);
                for (const row of s.results!) {
                  const base = (row.frame || "").split("/").pop() || row.frame;
                  next.set(`${r.modelName}::${base}`, { ...row, model: r.modelName, frame: base });
                }
                return next;
              });
              setProgress((cur) => cur ? { ...cur, done: cur.done + s.results!.filter((x) => x.ok).length } : cur);
            }
            setLocalRuns((cur) => ({ ...cur, [r.modelName]: { ...cur[r.modelName], state: "succeeded", runPageUrl: s.run_page_url, finishedAt: Date.now() } }));
          } else if (s.life_cycle_state === "INTERNAL_ERROR" || s.life_cycle_state === "SKIPPED" ||
                     (s.life_cycle_state === "TERMINATED" && s.result_state !== "SUCCESS")) {
            setLocalRuns((cur) => ({ ...cur, [r.modelName]: { ...cur[r.modelName], state: "failed", runPageUrl: s.run_page_url, error: s.state_message ?? "job failed", finishedAt: Date.now() } }));
          } else {
            setLocalRuns((cur) => ({ ...cur, [r.modelName]: { ...cur[r.modelName], runPageUrl: s.run_page_url } }));
          }
        } catch {
          localErrCountsRef.current[r.runId] = (localErrCountsRef.current[r.runId] ?? 0) + 1;
          if ((localErrCountsRef.current[r.runId] ?? 0) >= 8) {
            setLocalRuns((cur) => ({ ...cur, [r.modelName]: { ...cur[r.modelName], state: "failed", error: "polling failed repeatedly", finishedAt: Date.now() } }));
          }
        }
      }));
    };
    tick();
    const interval = setInterval(tick, 8000);
    return () => { stopped = true; clearInterval(interval); };
    // re-subscribe whenever the set of inflight run IDs changes
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [Object.values(localRuns).filter((r) => r.state === "running" || r.state === "submitting").map((r) => r.runId).join(",")]);

  // Keep `running` in sync with whether any local GPU job is still in
  // flight. Also handles the refresh-recovery case: if localRuns came back
  // from localStorage with "running" entries, this flips `running` on so
  // the UI shows the spinning indicators and disables the Run button.
  useEffect(() => {
    const inflight = Object.values(localRuns).some((r) => r.state === "running" || r.state === "submitting");
    if (inflight && !running) setRunning(true);
    if (!inflight && running && cancelRef.current === null) setRunning(false);
  }, [localRuns, running]);

  useEffect(() => {
    setFrames(null);
    api
      .frames({ source: frameSource })
      .then(setFrames)
      .catch((e) => setLoadError(`frames: ${e.message}`));
  }, [frameSource]);

  // Fetch labels for every loaded frame path so we can render gold-truth
  // chips + accuracy stats. Cheap: a single batched call keyed by paths.
  useEffect(() => {
    if (!frames || frames.length === 0) { setLabels(new Map()); return; }
    api.listLabels(frames.map((f) => f.path))
      .then((rows) => {
        const m = new Map<string, FrameLabel>();
        for (const r of rows) m.set(r.frame_path, r);
        setLabels(m);
      })
      .catch(() => setLabels(new Map()));
  }, [frames]);

  const visibleFrames = useMemo(() => {
    if (!frames) return [];
    let f = frames;
    if (videoFilter !== "__all__") f = f.filter((x) => x.video === videoFilter);
    // Subsample to roughly intervalSec apart, per video. (Selection state is
    // preserved across interval changes — the Set isn't cleared — but a
    // selected frame won't override the interval. If it doesn't pass the
    // filter, it's hidden but still selected.)
    if (intervalSec > 0) {
      const byVideo = new Map<string, FrameEntry[]>();
      for (const x of f) {
        const k = x.video || "__none__";
        if (!byVideo.has(k)) byVideo.set(k, []);
        byVideo.get(k)!.push(x);
      }
      const out: FrameEntry[] = [];
      for (const list of byVideo.values()) {
        list.sort((a, b) => (a.timestamp_s ?? 0) - (b.timestamp_s ?? 0));
        let lastT = -Infinity;
        for (const x of list) {
          const t = x.timestamp_s ?? 0;
          if (t - lastT >= intervalSec) {
            out.push(x);
            lastT = t;
          }
        }
      }
      return out;
    }
    return f;
  }, [frames, videoFilter, intervalSec]);

  const videoOptions = useMemo(() => {
    if (!frames) return [];
    const set = new Set<string>();
    for (const f of frames) if (f.video) set.add(f.video);
    return Array.from(set).sort();
  }, [frames]);

  const visionModels = useMemo(
    () => (models ?? []).filter((m) => m.vision && m.ready),
    [models]
  );

  // ── handlers ──────────────────────────────────────────────────────────
  const toggleFrame = (path: string) => {
    setSelectedFrames((prev) => {
      const next = new Set(prev);
      next.has(path) ? next.delete(path) : next.add(path);
      return next;
    });
  };
  const toggleModel = (name: string) => {
    setSelectedModels((prev) => {
      const next = new Set(prev);
      next.has(name) ? next.delete(name) : next.add(name);
      return next;
    });
  };
  const toggleLocalModel = (name: string) => {
    setSelectedLocalModels((prev) => {
      const next = new Set(prev);
      next.has(name) ? next.delete(name) : next.add(name);
      return next;
    });
  };
  const selectAllVisibleFrames = () => {
    setSelectedFrames(new Set(visibleFrames.map((f) => f.path)));
  };
  const clearFrames = () => setSelectedFrames(new Set());
  const setRecommendedModels = () => {
    const ready = new Set(visionModels.map((m) => m.name));
    setSelectedModels(new Set(Array.from(RECOMMENDED_MODELS).filter((m) => ready.has(m))));
  };

  const startRun = () => {
    if (running) return;
    if (selectedFrames.size === 0) return;
    if (selectedModels.size === 0 && selectedLocalModels.size === 0) return;
    setRunning(true);
    setResults(new Map());
    const totalCells =
      selectedFrames.size * (selectedModels.size + selectedLocalModels.size);
    setProgress({ done: 0, total: totalCells });
    setRunError(null);
    setRunDuration(null);
    setLocalRuns({});
    const t0 = performance.now();

    // 1) AI Gateway models — run in one batched call, results land all at once.
    if (selectedModels.size > 0) {
      cancelRef.current = api.playgroundRun(
        {
          frame_paths: Array.from(selectedFrames),
          model_names: Array.from(selectedModels),
          prompt,
        },
        {
          onResponse: (resp) => {
            setResults((cur) => {
              const next = new Map(cur);
              for (const r of resp.results) {
                next.set(`${r.model}::${r.frame}`, r);
              }
              return next;
            });
            setProgress((cur) => cur ? { ...cur, done: cur.done + resp.successful } : cur);
            setRunDuration(resp.elapsed_s);
            cancelRef.current = null;
            // If there are no local jobs, we're done.
            const anyLocal = Object.values(localRuns).some((r) => r.state === "running" || r.state === "submitting");
            if (!anyLocal && selectedLocalModels.size === 0) setRunning(false);
          },
          onError: (err) => {
            setRunError(err.message);
            cancelRef.current = null;
            setRunDuration(Number(((performance.now() - t0) / 1000).toFixed(2)));
            // Don't kill local jobs already submitted.
            const anyLocal = Object.values(localRuns).some((r) => r.state === "running" || r.state === "submitting");
            if (!anyLocal) setRunning(false);
          },
        }
      );
    } else {
      cancelRef.current = null;
    }

    // 2) Local models — submit each as its own GPU job.
    const framePaths = Array.from(selectedFrames);
    Array.from(selectedLocalModels).forEach(async (modelName) => {
      setLocalRuns((cur) => ({
        ...cur,
        [modelName]: {
          runId: "(submitting)",
          databricksRunId: "",
          modelName,
          framePaths,
          state: "submitting",
          startedAt: Date.now(),
        },
      }));
      try {
        const r = await api.localRunSubmit({ model_name: modelName, frame_paths: framePaths, prompt });
        setLocalRuns((cur) => ({
          ...cur,
          [modelName]: {
            ...cur[modelName],
            runId: r.run_id,
            databricksRunId: r.databricks_run_id,
            state: "running",
          },
        }));
      } catch (e) {
        setLocalRuns((cur) => ({
          ...cur,
          [modelName]: {
            ...cur[modelName],
            state: "failed",
            error: (e as Error).message,
            finishedAt: Date.now(),
          },
        }));
      }
    });
  };

  const stopRun = () => {
    cancelRef.current?.();
    cancelRef.current = null;
    setRunning(false);
  };

  // Union of AI Gateway + Local models that actually ran in this batch.
  // Used to populate the snapshot's model_names + the best-model dropdown.
  const allRunModels = useMemo(
    () => [...Array.from(selectedModels), ...Array.from(selectedLocalModels)],
    [selectedModels, selectedLocalModels],
  );

  const openSaveDialog = () => {
    if (results.size === 0) return;
    // Top vote across results = candidate "best model"
    const counts = new Map<string, number>();
    for (const r of results.values()) {
      if (!r.ok) continue;
      const cls = primaryClass(r.parsed);
      if (cls) {
        counts.set(r.model, (counts.get(r.model) || 0) + 1);
      }
    }
    const first = allRunModels[0] ?? "";
    setSaveBestModel(first);
    setSaveName(`${first.replace(/^databricks-/, "") || "snapshot"} ${new Date().toLocaleString()}`);
    setSaveError(null);
    setSaveSuccess(null);
    setSaveOpen(true);
  };

  const submitSave = async () => {
    if (!saveName.trim() || running) return;
    setSaving(true);
    setSaveError(null);
    try {
      const resp = await api.saveSnapshot({
        name: saveName.trim(),
        frame_paths: Array.from(selectedFrames),
        // Include both AI Gateway and Local models so the snapshot's
        // `model_names` faithfully reflects everything that ran.
        model_names: allRunModels,
        prompt,
        best_model: saveBestModel || null,
        results: Array.from(results.values()),
        elapsed_s: runDuration ?? undefined,
      });
      setSaveSuccess(resp.id);
      await refreshSnapshots();
      setTimeout(() => setSaveOpen(false), 600);
    } catch (e) {
      setSaveError((e as Error).message);
    } finally {
      setSaving(false);
    }
  };

  const loadSnapshot = async (id: string) => {
    try {
      const s = await api.getSnapshot(id);
      setSelectedFrames(new Set(s.frame_paths));
      setSelectedModels(new Set(s.model_names));
      setPrompt(s.prompt);
      // Re-hydrate results so the matrix immediately shows what was saved
      const m = new Map<string, RunResultRow>();
      for (const r of s.results) m.set(`${r.model}::${r.frame}`, r);
      setResults(m);
      setRunDuration(s.elapsed_s ?? null);
      setProgress({ done: s.results.length, total: s.results.length });
    } catch (e) {
      setRunError(`Failed to load snapshot: ${(e as Error).message}`);
    }
  };

  const sendToStudio = (id: string) => {
    if (onSendToStudio) onSendToStudio(id);
  };

  const removeSnapshot = async (id: string) => {
    try {
      await api.deleteSnapshot(id);
      await refreshSnapshots();
    } catch { /* ignore */ }
  };

  // ── derived for the results matrix ───────────────────────────────────
  const runFrames = useMemo(() => Array.from(selectedFrames).sort(), [selectedFrames]);
  const runModels = useMemo(
    () => [...Array.from(selectedModels), ...Array.from(selectedLocalModels)].sort(),
    [selectedModels, selectedLocalModels],
  );
  const canRun =
    !running &&
    selectedFrames.size > 0 &&
    (selectedModels.size > 0 || selectedLocalModels.size > 0);

  return (
    <div className="grid gap-6 lg:grid-cols-[minmax(0,360px)_1fr]">
      {/* ─── LEFT COLUMN: selection panel ─────────────────────────────── */}
      <div className="space-y-4 lg:sticky lg:top-20 lg:self-start">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between">
            <CardTitle className="text-base">Frames</CardTitle>
            <div className="flex items-center gap-2">
              <Badge variant="secondary">{selectedFrames.size} / {visibleFrames.length} selected</Badge>
            </div>
          </CardHeader>
          <CardContent className="space-y-3">
            <div className="flex flex-col gap-1.5">
              <Label className="text-xs">Source</Label>
              <Select value={frameSource} onValueChange={(v) => setFrameSource(v as FrameSource)}>
                <SelectTrigger className="h-8">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="eval">Eval set (curated 31)</SelectItem>
                  <SelectItem value="extracted">Smart-extracted (per video)</SelectItem>
                  <SelectItem value="labeled">Labeled (ground truth)</SelectItem>
                </SelectContent>
              </Select>
            </div>

            <div className="flex flex-col gap-1.5">
              <Label className="text-xs">
                Sample every <span className="font-mono">{intervalSec === 0 ? "frame" : `${intervalSec}s`}</span>
              </Label>
              <Select value={String(intervalSec)} onValueChange={(v) => setIntervalSec(Number(v))}>
                <SelectTrigger className="h-8">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="0">All frames</SelectItem>
                  <SelectItem value="5">5s</SelectItem>
                  <SelectItem value="10">10s</SelectItem>
                  <SelectItem value="15">15s</SelectItem>
                  <SelectItem value="30">30s (default)</SelectItem>
                  <SelectItem value="60">60s</SelectItem>
                  <SelectItem value="120">120s</SelectItem>
                  <SelectItem value="300">5 min</SelectItem>
                </SelectContent>
              </Select>
            </div>

            <div className="flex flex-col gap-1.5">
              <Label className="text-xs">Video</Label>
              <Select
                value={videoFilter}
                onValueChange={setVideoFilter}
                disabled={videoOptions.length === 0}
              >
                <SelectTrigger className="h-8" title={
                  videoOptions.length === 0
                    ? "No per-video attribution in this frame source — switch source or check filename pattern"
                    : undefined
                }>
                  <SelectValue placeholder={videoOptions.length === 0 ? "(no video metadata)" : undefined} />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="__all__">All videos</SelectItem>
                  {videoOptions.map((v) => (
                    <SelectItem key={v} value={v}>{v}</SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            <div className="flex gap-2">
              <Button variant="outline" size="sm" onClick={selectAllVisibleFrames} disabled={!visibleFrames.length}>
                Select all
              </Button>
              <Button
                variant="outline"
                size="sm"
                onClick={() => setSelectedFrames(new Set(visibleFrames.filter((f) => labels.has(f.path)).map((f) => f.path)))}
                disabled={!visibleFrames.some((f) => labels.has(f.path))}
                title="Pick only frames with ground-truth labels"
              >
                Labeled only
              </Button>
              <Button variant="outline" size="sm" onClick={clearFrames} disabled={!selectedFrames.size}>
                Clear
              </Button>
            </div>

            {!frames ? (
              <div className="flex items-center gap-2 text-xs text-muted-foreground">
                <Loader2 className="size-3 animate-spin" /> loading frames…
              </div>
            ) : (
              <ScrollArea className="h-80 rounded border">
                <div className="grid grid-cols-2 gap-2 p-2">
                  {visibleFrames.map((f, i) => {
                    const sel = selectedFrames.has(f.path);
                    return (
                      <div
                        key={f.path}
                        className={cn(
                          "group relative aspect-square overflow-hidden rounded border-2 transition-colors",
                          sel ? "border-primary" : "border-transparent hover:border-border"
                        )}
                        title={f.name}
                      >
                        <img
                          src={api.frameImageUrl(f.path)}
                          alt={f.name}
                          loading="lazy"
                          className="h-full w-full cursor-zoom-in object-cover"
                          onClick={() => setPreviewIdx(i)}
                        />
                        {/* Selection toggle as a corner overlay so the click-to-zoom isn't hijacked */}
                        <button
                          onClick={(e) => { e.stopPropagation(); toggleFrame(f.path); }}
                          className={cn(
                            "absolute right-1 top-1 grid size-6 place-items-center rounded-full shadow transition-colors",
                            sel
                              ? "bg-primary text-primary-foreground"
                              : "bg-background/80 text-muted-foreground opacity-0 group-hover:opacity-100"
                          )}
                          title={sel ? "Deselect" : "Select"}
                        >
                          <Check className="size-3.5" strokeWidth={3} />
                        </button>
                        {f.timestamp_s != null && (
                          <span className="absolute bottom-0 left-0 right-0 truncate bg-black/70 px-1.5 py-0.5 text-[11px] text-white">
                            {f.timestamp_s.toFixed(1)}s
                            {sel && <span className="ml-1 font-medium">· selected</span>}
                          </span>
                        )}
                        {labels.has(f.path) && (
                          <span className="absolute left-1 top-1 rounded bg-emerald-600/90 px-1.5 py-0.5 text-[10px] font-medium text-white shadow"
                                title={`Ground truth: ${labels.get(f.path)!.instrument}`}>
                            {labels.get(f.path)!.instrument}
                          </span>
                        )}
                      </div>
                    );
                  })}
                  {visibleFrames.length === 0 && (
                    <p className="col-span-3 p-4 text-center text-xs text-muted-foreground">
                      No frames in this source/video.
                    </p>
                  )}
                </div>
              </ScrollArea>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between">
            <CardTitle className="text-base">Models</CardTitle>
            <div className="flex items-center gap-2">
              <Badge variant="secondary">
                {selectedModels.size + selectedLocalModels.size} selected
              </Badge>
            </div>
          </CardHeader>
          <CardContent className="space-y-3">
            <div className="flex gap-2">
              <Button variant="outline" size="sm" onClick={setRecommendedModels}>
                <Sparkles className="size-3.5" /> Recommended
              </Button>
              <Button
                variant="outline"
                size="sm"
                onClick={() => { api.models(true).then(setModels); api.localModels().then(setLocalModels).catch(() => {}); }}
                title="Re-fetch endpoints + local models"
              >
                <RefreshCcw className="size-3.5" />
              </Button>
            </div>

            {/* AI Gateway section */}
            <div>
              <div className="mb-1 flex items-center justify-between text-[11px] font-medium uppercase tracking-wide text-muted-foreground">
                <span>AI Gateway</span>
                <span>{selectedModels.size} of {visionModels.length}</span>
              </div>
              {!models ? (
                <div className="flex items-center gap-2 text-xs text-muted-foreground">
                  <Loader2 className="size-3 animate-spin" /> loading endpoints…
                </div>
              ) : (
                <ScrollArea className="h-44 rounded border">
                  <div className="space-y-1 p-1.5">
                    {visionModels.map((m) => {
                      const sel = selectedModels.has(m.name);
                      const recommended = RECOMMENDED_MODELS.has(m.name);
                      return (
                        <button
                          key={m.name}
                          onClick={() => toggleModel(m.name)}
                          className={cn(
                            "flex w-full items-center gap-2 rounded px-2 py-1.5 text-left text-xs transition-colors",
                            sel ? "bg-primary/10" : "hover:bg-accent"
                          )}
                        >
                          <Checkbox checked={sel} tabIndex={-1} aria-hidden className="pointer-events-none" />
                          <span className="flex-1 truncate font-mono">{m.name.replace(/^databricks-/, "")}</span>
                          {recommended && <Badge variant="secondary" className="text-[10px]">⭐</Badge>}
                        </button>
                      );
                    })}
                    {visionModels.length === 0 && (
                      <p className="p-2 text-xs text-muted-foreground">No vision-capable endpoints found.</p>
                    )}
                  </div>
                </ScrollArea>
              )}
            </div>

            {/* Local models section */}
            <div>
              <div className="mb-1 flex items-center justify-between text-[11px] font-medium uppercase tracking-wide text-muted-foreground">
                <span>Local · Serverless GPU</span>
                <span>{selectedLocalModels.size} of {localModels.length}</span>
              </div>
              <div className="rounded border">
                <div className="space-y-1 p-1.5">
                  {localModels.length === 0 ? (
                    <p className="p-2 text-xs text-muted-foreground">No local models cached. Drop a manifest in the Volume to add one.</p>
                  ) : localModels.map((m) => {
                    const sel = selectedLocalModels.has(m.name);
                    return (
                      <button
                        key={m.name}
                        onClick={() => m.ready && toggleLocalModel(m.name)}
                        disabled={!m.ready}
                        className={cn(
                          "flex w-full items-center gap-2 rounded px-2 py-1.5 text-left text-xs transition-colors",
                          sel ? "bg-primary/10" : "hover:bg-accent",
                          !m.ready && "opacity-50 cursor-not-allowed"
                        )}
                        title={m.notes ?? undefined}
                      >
                        <Checkbox checked={sel} disabled={!m.ready} tabIndex={-1} aria-hidden className="pointer-events-none" />
                        <span className="flex-1 truncate font-mono">{m.display_name}</span>
                        <Badge variant="outline" className="text-[10px] font-normal">{m.accelerator.replace("GPU_", "")}</Badge>
                        {!m.ready && <Badge variant="outline" className="text-[10px] font-normal text-muted-foreground">caching…</Badge>}
                      </button>
                    );
                  })}
                </div>
              </div>
              {selectedLocalModels.size > 0 && (
                <p className="mt-1 text-[11px] text-muted-foreground">
                  Local models cold-start in ~5–15 min on first invocation.
                  Job runs in the background — feel free to keep working.
                </p>
              )}
            </div>
          </CardContent>
        </Card>
      </div>

      {/* ─── RIGHT COLUMN: prompt + run + results ─────────────────────── */}
      <div className="space-y-4">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between">
            <CardTitle className="text-base">Prompt</CardTitle>
            <div className="flex items-center gap-2">
              <Button variant="ghost" size="sm" onClick={() => setPrompt(workspaceDefaultPrompt)}>
                Reset to default
              </Button>
            </div>
          </CardHeader>
          <CardContent>
            <Textarea
              rows={12}
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              className="text-xs"
            />
          </CardContent>
        </Card>

        <Card>
          <CardContent className="flex flex-wrap items-center gap-3 p-4">
            <Button onClick={startRun} disabled={!canRun} variant="default" className="gap-2">
              {running ? (<><Loader2 className="size-4 animate-spin" /> Running…</>) : (<><Play className="size-4" /> Run</>)}
            </Button>
            {running && (
              <Button variant="outline" onClick={stopRun} className="gap-2">
                <Square className="size-4" /> Stop
              </Button>
            )}
            {progress && (
              <span className="text-xs text-muted-foreground">
                {progress.done} / {progress.total} done
                {runDuration != null && !running && ` · ${runDuration}s`}
              </span>
            )}
            {runError && (
              <span className="flex items-center gap-1 text-xs text-destructive">
                <AlertCircle className="size-3.5" /> {runError}
              </span>
            )}
            <Button
              variant="outline"
              onClick={openSaveDialog}
              disabled={running || results.size === 0}
              className="gap-2"
              title="Save this run as a snapshot — reusable in Studio + Compare"
            >
              <Save className="size-4" /> Save snapshot
            </Button>
            <Button
              variant="ghost"
              onClick={() => {
                setResults(new Map());
                setProgress(null);
                setRunDuration(null);
                setLocalRuns({});
              }}
              disabled={running || (results.size === 0 && Object.keys(localRuns).length === 0)}
              className="gap-2 text-muted-foreground"
              title="Clear the persisted results matrix"
            >
              <Trash2 className="size-3.5" /> Clear
            </Button>
            <Button
              variant="outline"
              onClick={() => openOptimizeDialog(undefined)}
              disabled={snapshots.length === 0}
              className="gap-2"
              title="Optimize a prompt for one model in a snapshot, using another model's outputs as gold labels"
            >
              <Wand2 className="size-4" /> Optimize prompt…
            </Button>
            <span className="ml-auto text-xs text-muted-foreground">
              {selectedFrames.size} frame(s) × {selectedModels.size} model(s) = {selectedFrames.size * selectedModels.size} call(s)
            </span>
          </CardContent>
        </Card>

        {/* Recent snapshots strip */}
        {snapshots.length > 0 && (
          <Card>
            <CardHeader className="flex flex-row items-center justify-between pb-2">
              <CardTitle className="text-sm">Recent snapshots</CardTitle>
              <span className="text-xs text-muted-foreground">{snapshots.length}</span>
            </CardHeader>
            <CardContent className="p-3 pt-0">
              <ScrollArea className="max-h-48">
                <ul className="space-y-1">
                  {snapshots.map((s) => (
                    <li
                      key={s.id}
                      className="flex items-center gap-2 rounded border bg-card/40 px-2 py-1.5 text-xs"
                    >
                      <div className="min-w-0 flex-1">
                        <div className="truncate font-medium">{s.name}</div>
                        <div className="truncate text-[10px] text-muted-foreground">
                          {s.n_frames} frames · {s.n_models} models
                          {s.best_model ? ` · best: ${s.best_model.replace(/^databricks-/, "")}` : ""}
                          {" · "}{new Date(s.created_at).toLocaleString()}
                        </div>
                      </div>
                      <Button size="sm" variant="ghost" onClick={() => loadSnapshot(s.id)} title="Load into Playground">
                        Load
                      </Button>
                      <Button
                        size="sm"
                        variant="ghost"
                        onClick={() => openOptimizeDialog(s.id)}
                        className="gap-1"
                        title="Optimize a prompt using this snapshot's predictions"
                      >
                        <Wand2 className="size-3" />
                      </Button>
                      <Button
                        size="sm"
                        variant="default"
                        onClick={() => sendToStudio(s.id)}
                        className="gap-1"
                        title="Open this combo in Studio for full-video review"
                      >
                        <Film className="size-3" /> Studio
                      </Button>
                      <Button size="sm" variant="ghost" onClick={() => removeSnapshot(s.id)} title="Delete">
                        <Trash2 className="size-3.5" />
                      </Button>
                    </li>
                  ))}
                </ul>
              </ScrollArea>
            </CardContent>
          </Card>
        )}

        {/* Optimize dialog */}
        <Dialog open={optimizeOpen} onOpenChange={setOptimizeOpen}>
          <DialogContent className="max-w-xl">
            <DialogHeader>
              <DialogTitle>Optimize prompt</DialogTitle>
              <DialogDescription>
                Submits a Databricks Job that runs <span className="font-mono">gepa</span> or
                <span className="font-mono"> dspy</span> on a snapshot. The teacher model's
                outputs become pseudo-labels; the student model's prompt gets optimized to match.
              </DialogDescription>
            </DialogHeader>

            {!optRun && (
              <div className="space-y-3">
                <div className="space-y-1.5">
                  <Label className="text-xs">Snapshot</Label>
                  <Select value={optSnapshotId} onValueChange={setOptSnapshotId}>
                    <SelectTrigger>
                      <SelectValue placeholder="Pick a snapshot" />
                    </SelectTrigger>
                    <SelectContent>
                      {snapshots.map((s) => (
                        <SelectItem key={s.id} value={s.id}>{s.name}</SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
                <div className="grid grid-cols-2 gap-3">
                  <div className="space-y-1.5">
                    <Label className="text-xs">
                      Teacher (provides labels)
                      {optTeacher === "__gold__" && (
                        <span className="ml-1.5 text-[10px] text-emerald-600 dark:text-emerald-400">
                          supervised
                        </span>
                      )}
                    </Label>
                    <Select value={optTeacher} onValueChange={setOptTeacher}>
                      <SelectTrigger><SelectValue /></SelectTrigger>
                      <SelectContent>
                        <SelectItem value="__gold__">
                          Gold labels (ground truth)
                        </SelectItem>
                        {visionModels.map((m) => (
                          <SelectItem key={m.name} value={m.name}>
                            {m.name.replace(/^databricks-/, "")}
                          </SelectItem>
                        ))}
                        {localModels.filter((m) => m.ready).map((m) => (
                          <SelectItem key={m.name} value={m.name}>
                            {m.display_name} (local)
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                    <p className="text-[10px] text-muted-foreground">
                      {optTeacher === "__gold__"
                        ? "Labels you saved in the Label tab become the gold answer for each frame."
                        : "This model's predictions become pseudo-labels."}
                    </p>
                  </div>
                  <div className="space-y-1.5">
                    <Label className="text-xs">Student (prompt being optimized)</Label>
                    <Select value={optStudent} onValueChange={setOptStudent}>
                      <SelectTrigger><SelectValue /></SelectTrigger>
                      <SelectContent>
                        {visionModels.map((m) => (
                          <SelectItem key={m.name} value={m.name}>
                            {m.name.replace(/^databricks-/, "")}
                          </SelectItem>
                        ))}
                        {localModels.filter((m) => m.ready).map((m) => (
                          <SelectItem key={m.name} value={m.name}>
                            {m.display_name} (local)
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </div>
                </div>
                <div className="grid grid-cols-2 gap-3">
                  <div className="space-y-1.5">
                    <Label className="text-xs">Optimizer</Label>
                    <Select value={optType} onValueChange={(v) => setOptType(v as "gepa" | "dspy")}>
                      <SelectTrigger><SelectValue /></SelectTrigger>
                      <SelectContent>
                        <SelectItem value="gepa">GEPA (reflective)</SelectItem>
                        <SelectItem value="dspy">DSPy (BootstrapFewShot)</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                  {optType === "gepa" && (
                    <div className="space-y-1.5">
                      <Label className="text-xs">Rounds</Label>
                      <Input
                        type="number"
                        min={1}
                        max={10}
                        value={optRounds}
                        onChange={(e) => setOptRounds(Number(e.target.value || 1))}
                      />
                    </div>
                  )}
                </div>
                {optError && (
                  <p className="flex items-center gap-1 text-xs text-destructive">
                    <AlertCircle className="size-3.5" /> {optError}
                  </p>
                )}
              </div>
            )}

            {optRun && (
              <div className="space-y-3">
                <div className="rounded border bg-muted/40 p-2 text-xs">
                  <div><span className="text-muted-foreground">run:</span> <span className="font-mono">{optRun.run_id}</span></div>
                  <div><span className="text-muted-foreground">student:</span> {optRun.student_model.replace(/^databricks-/, "")}</div>
                  <div><span className="text-muted-foreground">teacher:</span> {optRun.teacher_model.replace(/^databricks-/, "")}</div>
                  <div><span className="text-muted-foreground">eval examples:</span> {optRun.n_eval_examples}</div>
                </div>
                <div className="rounded border p-2 text-sm">
                  <div className="flex items-center justify-between">
                    <span>
                      <span className="text-muted-foreground">status: </span>
                      <span className="font-mono">
                        {optStatus?.life_cycle_state ?? "submitting…"}
                        {optStatus?.result_state && ` / ${optStatus.result_state}`}
                      </span>
                    </span>
                    {optStatus?.run_page_url && (
                      <a
                        href={optStatus.run_page_url}
                        target="_blank"
                        rel="noreferrer"
                        className="flex items-center gap-1 text-xs text-primary hover:underline"
                      >
                        <ExternalLink className="size-3" /> open in workspace
                      </a>
                    )}
                  </div>
                  {optStatus?.history && optStatus.history.length > 0 && (
                    <div className="mt-2 text-xs text-muted-foreground">
                      round-by-round score: {optStatus.history.map((s) => `${(s * 100).toFixed(0)}%`).join(" → ")}
                    </div>
                  )}
                  {optStatus?.score != null && (
                    <div className="mt-1 text-xs">
                      <span className="text-muted-foreground">final: </span>
                      <span className="font-medium">{(optStatus.score * 100).toFixed(1)}%</span>
                    </div>
                  )}
                  {optStatus?.state_message && (
                    <div className="mt-1 truncate text-[11px] text-muted-foreground">{optStatus.state_message}</div>
                  )}
                </div>
                {optStatus?.best_prompt && (
                  <div className="space-y-1.5">
                    <Label className="text-xs">Optimized prompt</Label>
                    <Textarea readOnly value={optStatus.best_prompt} rows={10} className="text-[11px]" />
                  </div>
                )}
                {optError && (
                  <p className="flex items-center gap-1 text-xs text-destructive">
                    <AlertCircle className="size-3.5" /> {optError}
                  </p>
                )}
              </div>
            )}

            <DialogFooter>
              {!optRun ? (
                <>
                  <Button variant="outline" onClick={() => setOptimizeOpen(false)} disabled={optSubmitting}>
                    Cancel
                  </Button>
                  <Button onClick={submitOptimize} disabled={optSubmitting || !optSnapshotId} className="gap-2">
                    {optSubmitting ? <><Loader2 className="size-4 animate-spin" /> Submitting…</> : <><Wand2 className="size-4" /> Run optimizer</>}
                  </Button>
                </>
              ) : (
                <>
                  <Button variant="outline" onClick={() => setOptimizeOpen(false)}>Close</Button>
                  <Button
                    onClick={applyOptimizedPrompt}
                    disabled={!optStatus?.best_prompt}
                    className="gap-2"
                  >
                    <Check className="size-4" /> Apply this prompt
                  </Button>
                </>
              )}
            </DialogFooter>
          </DialogContent>
        </Dialog>

        {/* Save snapshot dialog */}
        <Dialog open={saveOpen} onOpenChange={setSaveOpen}>
          <DialogContent>
            <DialogHeader>
              <DialogTitle>Save snapshot</DialogTitle>
              <DialogDescription>
                Persists this (frames + models + prompt + results) to Lakebase. Reload it any time
                from Playground or apply it as Studio's analysis config.
              </DialogDescription>
            </DialogHeader>
            <div className="space-y-3">
              <div className="space-y-1.5">
                <Label className="text-xs">Name</Label>
                <Input
                  autoFocus
                  value={saveName}
                  onChange={(e) => setSaveName(e.target.value)}
                  placeholder="e.g. sonnet-45-gepa-v1"
                />
              </div>
              <div className="space-y-1.5">
                <Label className="text-xs">Best model (used by Studio)</Label>
                <Select value={saveBestModel} onValueChange={setSaveBestModel}>
                  <SelectTrigger>
                    <SelectValue placeholder="Pick the model whose outputs you trust most" />
                  </SelectTrigger>
                  <SelectContent>
                    {allRunModels.map((m) => (
                      <SelectItem key={m} value={m}>
                        {m.replace(/^databricks-/, "")}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
              {saveError && (
                <p className="flex items-center gap-1 text-xs text-destructive">
                  <AlertCircle className="size-3.5" /> {saveError}
                </p>
              )}
              {saveSuccess && (
                <p className="flex items-center gap-1 text-xs text-emerald-600">
                  <Check className="size-3.5" /> saved
                </p>
              )}
            </div>
            <DialogFooter>
              <Button variant="outline" onClick={() => setSaveOpen(false)} disabled={saving}>
                Cancel
              </Button>
              <Button onClick={submitSave} disabled={saving || !saveName.trim()}>
                {saving ? <><Loader2 className="size-4 animate-spin" /> Saving…</> : "Save"}
              </Button>
            </DialogFooter>
          </DialogContent>
        </Dialog>

        {loadError && (
          <Card>
            <CardContent className="flex items-center gap-2 p-4 text-sm text-destructive">
              <AlertCircle className="size-4" /> {loadError}
            </CardContent>
          </Card>
        )}

        {/* Results matrix */}
        {(runFrames.length > 0 && runModels.length > 0) && (
          <Card>
            <CardHeader>
              <CardTitle className="text-base">Results matrix</CardTitle>
            </CardHeader>
            <CardContent className="p-0">
              <ResultsMatrix
                frames={runFrames}
                models={runModels}
                results={results}
                localRuns={localRuns}
                labels={labels}
              />
            </CardContent>
          </Card>
        )}
      </div>

      <FramePreviewModal
        frames={visibleFrames}
        index={previewIdx}
        selected={selectedFrames}
        onClose={() => setPreviewIdx(null)}
        onIndexChange={setPreviewIdx}
        onToggleSelect={toggleFrame}
      />
    </div>
  );
}

// ── Frame preview modal (click-to-zoom) ─────────────────────────────────

function FramePreviewModal({
  frames,
  index,
  selected,
  onClose,
  onIndexChange,
  onToggleSelect,
}: {
  frames: FrameEntry[];
  index: number | null;
  selected: Set<string>;
  onClose: () => void;
  onIndexChange: (i: number) => void;
  onToggleSelect: (path: string) => void;
}) {
  const open = index != null && index >= 0 && index < frames.length;
  // Keyboard navigation
  useEffect(() => {
    if (!open) return;
    const handler = (e: KeyboardEvent) => {
      if (index == null) return;
      if (e.key === "ArrowRight" && index < frames.length - 1) {
        e.preventDefault(); onIndexChange(index + 1);
      } else if (e.key === "ArrowLeft" && index > 0) {
        e.preventDefault(); onIndexChange(index - 1);
      } else if (e.key === " ") {
        e.preventDefault();
        const f = frames[index];
        if (f) onToggleSelect(f.path);
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [open, index, frames, onIndexChange, onToggleSelect]);

  if (!open || index == null) return null;
  const f = frames[index];
  const isSelected = selected.has(f.path);

  return (
    <Dialog open={open} onOpenChange={(o) => !o && onClose()}>
      <DialogContent className="max-w-5xl">
        <DialogHeader>
          <DialogTitle className="font-mono text-sm">
            {f.name}
            {f.timestamp_s != null && (
              <span className="ml-2 text-muted-foreground">@ {f.timestamp_s.toFixed(2)}s</span>
            )}
          </DialogTitle>
          <DialogDescription className="text-xs">
            {f.video || "(unknown source)"} · frame {index + 1} of {frames.length}
            {" · "}
            <span className="text-muted-foreground">←/→ navigate · space to (de)select · esc to close</span>
          </DialogDescription>
        </DialogHeader>
        <div className="relative">
          <img
            src={api.frameImageUrl(f.path)}
            alt={f.name}
            className="max-h-[70vh] w-full rounded border bg-black object-contain"
          />
        </div>
        <DialogFooter className="flex flex-row items-center justify-between gap-2 sm:justify-between">
          <div className="flex gap-1">
            <Button
              variant="outline"
              size="sm"
              disabled={index === 0}
              onClick={() => onIndexChange(Math.max(0, index - 1))}
            >
              ← Prev
            </Button>
            <Button
              variant="outline"
              size="sm"
              disabled={index >= frames.length - 1}
              onClick={() => onIndexChange(Math.min(frames.length - 1, index + 1))}
            >
              Next →
            </Button>
          </div>
          <Button
            onClick={() => onToggleSelect(f.path)}
            variant={isSelected ? "default" : "outline"}
            className="gap-2"
          >
            <Check className="size-4" /> {isSelected ? "Selected" : "Select"}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

function ResultsMatrix({
  frames,
  models,
  results,
  localRuns,
  labels,
}: {
  frames: string[];
  models: string[];
  results: Map<string, RunResultRow>;
  localRuns?: Record<string, { state: "submitting" | "running" | "succeeded" | "failed"; error?: string; runPageUrl?: string | null; startedAt: number }>;
  labels?: Map<string, FrameLabel>;
}) {
  // Per-model accuracy across frames that have a gold label. Only counts
  // frames where the model produced a parseable predicted class.
  const perModelAcc = useMemo(() => {
    const out: Record<string, { correct: number; total: number }> = {};
    for (const m of models) out[m] = { correct: 0, total: 0 };
    for (const path of frames) {
      const fname = path.split("/").pop() || path;
      const gold = labels?.get(path)?.instrument;
      if (!gold) continue;
      for (const m of models) {
        const r = results.get(`${m}::${fname}`);
        if (!r?.ok) continue;
        const pred = primaryClass(r.parsed);
        if (!pred) continue;
        out[m].total += 1;
        if (pred === gold) out[m].correct += 1;
      }
    }
    return out;
  }, [frames, models, results, labels]);

  const labeledCount = useMemo(
    () => frames.filter((p) => labels?.has(p)).length,
    [frames, labels],
  );

  return (
    <div className="overflow-auto">
      <table className="w-full border-collapse text-xs">
        <thead className="sticky top-0 z-10 bg-card">
          <tr>
            <th className="sticky left-0 z-20 w-32 border-b border-r bg-card p-2 text-left font-medium">
              <div className="flex flex-col gap-0.5">
                <span>Frame</span>
                {labeledCount > 0 && (
                  <span className="text-[10px] font-normal text-muted-foreground">
                    {labeledCount}/{frames.length} labeled
                  </span>
                )}
              </div>
            </th>
            {models.map((m) => {
              const local = localRuns?.[m];
              const acc = perModelAcc[m];
              return (
                <th key={m} className="border-b p-2 text-left font-medium">
                  <div className="flex flex-col gap-0.5">
                    <span className="font-mono">{m.replace(/^databricks-/, "")}</span>
                    {local && local.state !== "succeeded" && (
                      <span className={cn(
                        "text-[10px] font-normal",
                        local.state === "failed" ? "text-destructive" : "text-amber-600 dark:text-amber-400"
                      )}>
                        {local.state === "submitting" && "submitting…"}
                        {local.state === "running" && `GPU running · ${Math.round((Date.now() - local.startedAt) / 1000)}s`}
                        {local.state === "failed" && "failed"}
                      </span>
                    )}
                    {acc && acc.total > 0 && (
                      <span className="text-[10px] font-normal text-emerald-600 dark:text-emerald-400">
                        {acc.correct}/{acc.total} vs gold · {((acc.correct / acc.total) * 100).toFixed(0)}%
                      </span>
                    )}
                  </div>
                </th>
              );
            })}
          </tr>
        </thead>
        <tbody>
          {frames.map((path) => {
            const fname = path.split("/").pop() || path;
            const gold = labels?.get(path)?.instrument;
            return (
              <tr key={path} className="border-b">
                <td className="sticky left-0 z-10 w-32 border-r bg-card p-2 align-top">
                  <div className="flex flex-col gap-1.5">
                    <img
                      src={api.frameImageUrl(path)}
                      alt={fname}
                      className="aspect-square w-28 rounded border object-cover"
                    />
                    <span className="truncate text-[10px] text-muted-foreground" title={fname}>
                      {fname}
                    </span>
                    {gold && (
                      <Badge variant="outline" className="self-start text-[10px] font-normal text-emerald-700 dark:text-emerald-400">
                        gold: {gold}
                      </Badge>
                    )}
                  </div>
                </td>
                {models.map((m) => {
                  const r = results.get(`${m}::${fname}`);
                  const local = localRuns?.[m];
                  return (
                    <td key={m} className="min-w-56 border-l p-2 align-top">
                      {r ? (
                        <ResultCell row={r} goldLabel={gold} />
                      ) : local && (local.state === "running" || local.state === "submitting") ? (
                        <div className="flex items-center gap-1.5 text-[11px] text-muted-foreground">
                          <Loader2 className="size-3 animate-spin" /> waiting on GPU…
                        </div>
                      ) : local && local.state === "failed" ? (
                        <div className="text-[11px] text-destructive" title={local.error}>job failed</div>
                      ) : (
                        <ResultCell row={r} goldLabel={gold} />
                      )}
                    </td>
                  );
                })}
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function ResultCell({ row, goldLabel }: { row?: RunResultRow; goldLabel?: string }) {
  if (!row) {
    return (
      <div className="flex items-center gap-1.5 text-muted-foreground">
        <Loader2 className="size-3 animate-spin" /> waiting…
      </div>
    );
  }
  if (!row.ok) {
    return (
      <div className="space-y-1">
        <div className="flex items-center gap-1 text-destructive">
          <AlertCircle className="size-3" /> error
        </div>
        <div className="text-[10px] text-muted-foreground">{row.error}</div>
      </div>
    );
  }
  const cls = primaryClass(row.parsed);
  const conf = confidence(row.parsed);
  const isCorrect = goldLabel && cls ? goldLabel === cls : null;
  return (
    <div className="space-y-1.5">
      <div className="flex items-center gap-1.5">
        {cls ? (
          <Badge>{cls}</Badge>
        ) : (
          <Badge variant="secondary">unparsed</Badge>
        )}
        {isCorrect === true && (
          <span className="text-emerald-600 dark:text-emerald-400" title={`matches gold: ${goldLabel}`}>✓</span>
        )}
        {isCorrect === false && (
          <span className="text-destructive" title={`gold: ${goldLabel}`}>✗</span>
        )}
        {conf != null && (
          <span className="text-[10px] text-muted-foreground">{(conf * 100).toFixed(0)}%</span>
        )}
        <span className="ml-auto text-[10px] text-muted-foreground">{row.elapsed_s}s</span>
      </div>
      {row.parsed && typeof row.parsed === "object" && (
        <details className="text-[11px] text-muted-foreground">
          <summary className="cursor-pointer">JSON</summary>
          <pre className="mt-1 max-h-40 overflow-auto rounded bg-muted p-1.5 font-mono text-[10px]">{JSON.stringify(row.parsed, null, 2)}</pre>
        </details>
      )}
      {!row.parsed && row.raw && (
        <details className="text-[11px] text-muted-foreground">
          <summary className="cursor-pointer">raw output</summary>
          <pre className="mt-1 max-h-40 overflow-auto whitespace-pre-wrap rounded bg-muted p-1.5 font-mono text-[10px]">{row.raw.slice(0, 500)}</pre>
        </details>
      )}
    </div>
  );
}
