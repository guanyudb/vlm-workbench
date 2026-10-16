import { useEffect, useRef, useState } from "react";
import {
  AlertCircle, CheckCircle2, ChevronDown, ChevronRight, ExternalLink, Loader2,
  RefreshCcw, Sparkles, X, XCircle,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Checkbox } from "@/components/ui/checkbox";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import {
  Select, SelectContent, SelectItem, SelectTrigger, SelectValue,
} from "@/components/ui/select";
import {
  api,
  type FinetuneRunSummary,
  type LocalModelEntry,
  type SnapshotSummary,
} from "@/api";
import { usePersistentState } from "@/lib/use-persistent-state";
import { usePolling } from "@/lib/use-polling";
import { cn } from "@/lib/utils";

const DEFAULT_FT_PROMPT = `Identify the surgical instrument by its visible visual features. Respond with strict JSON: {"instrument": "<class>", "anatomy": "<short>", "tissue_condition": "<short>"} where <class> is one of: probe, shaver, burr, grasper, biter, suture_passer, anchor_driver, electrocautery, cannula, scissors, drill_guide, trocar, knot_pusher, rasp, other_metal_tool, no_instrument_visible.`;

// Advanced-parameter defaults. Mirrors FineTuneRequest in app.py; used both
// as initial values and to flag which hidden settings the user has changed.
const DEFAULTS = {
  lr: 0.0002, loraAlpha: 32, loraDropout: 0.05, batch: 1, gradAccum: 8,
  warmup: 0.03, wd: 0.0, maxLen: 1024,
};

// Persisted across refresh so a running fine-tune keeps polling. Same pattern
// as the optimizer tracker — Lakebase + Volume hold the source of truth, but
// localStorage lets the UI keep showing the spinning state during a refresh.
type ActiveFT = {
  run_id: string;
  databricks_run_id: string;
  uc_model_name: string;
  base_model_name: string;
  n_train: number;
  state: "submitting" | "running" | "succeeded" | "failed";
  result_state?: string | null;
  state_message?: string | null;
  error_detail?: string | null;
  run_page_url?: string | null;
  result?: any;
  started_at: number;
  finished_at?: number;
};

// Serverless GPU options for fine-tune jobs. Values must match the Jobs API
// exactly (`GPU_<N>x<TYPE>`); a malformed name silently runs on CPU.
const ACCELERATORS = [
  { value: "GPU_1xA10",  label: "1× A10 (24 GB) · recommended for 8B bases" },
  { value: "GPU_1xH100", label: "1× H100 (80 GB) · 12B+ bases, may queue" },
  { value: "GPU_8xH100", label: "8× H100 · rarely needed for LoRA" },
];

const toggleIn = <T,>(arr: T[], v: T): T[] =>
  arr.includes(v) ? arr.filter((x) => x !== v) : [...arr, v];

const fmtElapsed = (ms: number) => {
  const s = Math.max(0, Math.round(ms / 1000));
  if (s < 60) return `${s}s`;
  const m = Math.floor(s / 60);
  return `${m}m ${s % 60}s`;
};

export default function Train() {
  // UC layout for label-rendering only. Pulled from /api/health so the
  // workbench shows the *actual* registered name regardless of workspace.
  const [ucCatalog, setUcCatalog] = useState<string>("<catalog>");
  const [ucSchema, setUcSchema] = useState<string>("<schema>");
  const [mlflowExperimentPath, setMlflowExperimentPath] = useState<string | null>(null);
  useEffect(() => {
    api.health().then((h) => {
      setUcCatalog(h.uc_catalog);
      setUcSchema(h.uc_schema);
      setMlflowExperimentPath(h.mlflow_experiment_path || null);
    }).catch(() => {});
  }, []);
  const [localModels, setLocalModels] = useState<LocalModelEntry[]>([]);
  const [labelStats, setLabelStats] = useState<{ total: number; by_instrument: { instrument: string; n: number }[] } | null>(null);
  const [pastRuns, setPastRuns] = useState<FinetuneRunSummary[]>([]);
  const [pastLoading, setPastLoading] = useState(false);
  const [snapshots, setSnapshots] = useState<SnapshotSummary[]>([]);
  const [error, setError] = useState<string | null>(null);

  // ── form state (persisted) ────────────────────────────────────────
  const [baseModel, setBaseModel] = usePersistentState<string>("vlmwb.train.baseModel", "qwen3-vl-8b");
  const [ucModelName, setUcModelName] = usePersistentState<string>("vlmwb.train.ucName", "");
  // Data source for labels. A single text input drives this:
  //   blank          → Lakebase (live) — every saved label, mutates
  //   catalog.schema.table → that Delta table — versioned snapshot
  // The source mode is derived so the UI can't drift from what gets sent.
  const [deltaTable, setDeltaTable] = usePersistentState<string>("vlmwb.train.deltaTable", "");
  const [deltaVersion, setDeltaVersion] = usePersistentState<string>("vlmwb.train.deltaVersion", "");
  const dataSource: "lakebase" | "delta" = deltaTable.trim() ? "delta" : "lakebase";
  // Filters (all AND-composed; empty = no restriction)
  const [filterSnapshots, setFilterSnapshots] = usePersistentState<string[]>("vlmwb.train.snapshots.v2", []);
  const [filterInstruments, setFilterInstruments] = usePersistentState<string[]>("vlmwb.train.instruments", []);
  const [filterVideos, setFilterVideos] = usePersistentState<string[]>("vlmwb.train.videos", []);
  const [labeledSince, setLabeledSince] = usePersistentState<string>("vlmwb.train.since", "");
  // "snapshots" restricts training to frames in the picked snapshots — how the
  // demo keeps held-out frames out of training. Defaults to whatever an
  // existing persisted snapshot selection implies.
  const [scope, setScope] = usePersistentState<"all" | "snapshots">(
    "vlmwb.train.scope", filterSnapshots.length ? "snapshots" : "all",
  );
  const [trainPrompt, setTrainPrompt] = usePersistentState<string>("vlmwb.train.prompt", DEFAULT_FT_PROMPT);
  // Hyperparams. Epochs + capacity are always visible; the rest sit behind
  // the "Output name, prompt & advanced" toggle. Capacity maps to LoRA r.
  const [numEpochs, setNumEpochs] = usePersistentState<number>("vlmwb.train.epochs", 3);
  const [capacity, setCapacity] = usePersistentState<"S" | "M" | "L">("vlmwb.train.capacity", "M");
  const [learningRate, setLearningRate] = usePersistentState<number>("vlmwb.train.lr", DEFAULTS.lr);
  const [loraAlpha, setLoraAlpha] = usePersistentState<number>("vlmwb.train.loraAlpha", DEFAULTS.loraAlpha);
  const [loraDropout, setLoraDropout] = usePersistentState<number>("vlmwb.train.loraDropout", DEFAULTS.loraDropout);
  const [batchSize, setBatchSize] = usePersistentState<number>("vlmwb.train.batch", DEFAULTS.batch);
  const [gradAccum, setGradAccum] = usePersistentState<number>("vlmwb.train.gradAccum", DEFAULTS.gradAccum);
  const [warmupRatio, setWarmupRatio] = usePersistentState<number>("vlmwb.train.warmup", DEFAULTS.warmup);
  const [weightDecay, setWeightDecay] = usePersistentState<number>("vlmwb.train.wd", DEFAULTS.wd);
  const [maxLength, setMaxLength] = usePersistentState<number>("vlmwb.train.maxLen", DEFAULTS.maxLen);
  const [accelerator, setAccelerator] = usePersistentState<string>("vlmwb.train.accel", "GPU_1xA10");
  // Capacity preset → LoRA rank. Higher = more learnable params + more
  // overfit risk on small datasets. 16 is a reasonable middle for the
  // typical workbench dataset size (10–200 labels).
  const loraR = capacity === "S" ? 8 : capacity === "L" ? 32 : 16;

  const useSnapshots = dataSource === "lakebase" && scope === "snapshots";
  const needsSnapshot = useSnapshots && filterSnapshots.length === 0;
  const extraFilterCount =
    (filterInstruments.length ? 1 : 0) + (filterVideos.length ? 1 : 0) +
    (labeledSince ? 1 : 0) + (deltaTable.trim() ? 1 : 0);
  // Hidden settings that differ from their defaults, so a collapsed section
  // never silently changes what gets trained.
  const customized = [
    ucModelName.trim() && "output name",
    trainPrompt.trim() !== DEFAULT_FT_PROMPT && "prompt",
    (learningRate !== DEFAULTS.lr || loraAlpha !== DEFAULTS.loraAlpha ||
      loraDropout !== DEFAULTS.loraDropout || batchSize !== DEFAULTS.batch ||
      gradAccum !== DEFAULTS.gradAccum || warmupRatio !== DEFAULTS.warmup ||
      weightDecay !== DEFAULTS.wd || maxLength !== DEFAULTS.maxLen) && "parameters",
  ].filter(Boolean) as string[];
  const [filtersOpen, setFiltersOpen] = useState(extraFilterCount > 0);
  const [advancedOpen, setAdvancedOpen] = useState(false);

  // Label selection shared by the live preview and the submit, so the count
  // shown is exactly what the job trains on.
  const labelSelection = () => {
    const parsedVer = deltaVersion.trim() ? parseInt(deltaVersion.trim(), 10) : undefined;
    return {
      data_source: dataSource,
      delta_table: dataSource === "delta" ? deltaTable.trim() : undefined,
      delta_version: dataSource === "delta" && parsedVer !== undefined && !Number.isNaN(parsedVer) ? parsedVer : undefined,
      // snapshot_ids is Lakebase-only
      snapshot_ids: useSnapshots && filterSnapshots.length ? filterSnapshots : undefined,
      video_names: filterVideos.length ? filterVideos : undefined,
      labeled_since: labeledSince || undefined,
    };
  };

  // Live preview: how many labels actually match the current selection
  const [preview, setPreview] = useState<{ total: number; by_instrument: { instrument: string; n: number }[] } | null>(null);
  const [previewErr, setPreviewErr] = useState<string | null>(null);
  useEffect(() => {
    if (needsSnapshot) { setPreview(null); setPreviewErr(null); return; }
    const t = setTimeout(() => {
      api.finetunePreview({
        ...labelSelection(),
        instruments: filterInstruments.length ? filterInstruments : undefined,
      }).then((p) => { setPreview(p); setPreviewErr(null); })
        .catch((e) => { setPreview(null); setPreviewErr((e as Error).message); });
    }, 300);  // debounce filter clicks
    return () => clearTimeout(t);
  }, [
    dataSource, deltaTable, deltaVersion, scope,
    filterInstruments.join(","),
    filterSnapshots.join(","),
    filterVideos.join(","),
    labeledSince,
  ]);

  // ── active runs (persisted) ───────────────────────────────────────
  const [activeRuns, setActiveRuns] = usePersistentState<ActiveFT[]>("vlmwb.train.active", []);
  const errCountsRef = useRef<Record<string, number>>({});
  const [submitting, setSubmitting] = useState(false);

  const refreshRuns = () => {
    setPastLoading(true);
    api.finetuneRuns(25).then(setPastRuns).catch(() => {}).finally(() => setPastLoading(false));
  };
  const refreshAll = () => {
    api.localModels().then(setLocalModels).catch(() => {});
    api.labelsStats().then(setLabelStats).catch(() => {});
    api.listSnapshots(30).then(setSnapshots).catch(() => {});
    refreshRuns();
  };
  useEffect(refreshAll, []);

  // Poller for in-flight fine-tune jobs. Tolerant of transient 502s — only
  // gives up after 10 consecutive errors per run.
  const hasInflightFT = activeRuns.some((r) => r.state === "running" || r.state === "submitting");
  usePolling(
    async () => {
      const inflight = activeRuns.filter((r) => r.state === "running" || r.state === "submitting");
      const updates: Record<string, Partial<ActiveFT>> = {};
      await Promise.all(inflight.map(async (r) => {
        if (r.state !== "running" || !r.databricks_run_id) return;
        try {
          const s = await api.finetuneStatus(r.run_id, r.databricks_run_id);
          errCountsRef.current[r.run_id] = 0;
          const life = s.life_cycle_state;
          if (life === "TERMINATED" || life === "INTERNAL_ERROR" || life === "SKIPPED") {
            const ok = life === "TERMINATED" && s.result_state === "SUCCESS";
            updates[r.run_id] = {
              state: ok ? "succeeded" : "failed",
              result_state: s.result_state,
              state_message: s.state_message,
              error_detail: s.error_detail,
              run_page_url: s.run_page_url,
              result: s.result,
              finished_at: Date.now(),
            };
          } else {
            updates[r.run_id] = {
              result_state: s.result_state,
              state_message: s.state_message,
              run_page_url: s.run_page_url,
            };
          }
        } catch {
          errCountsRef.current[r.run_id] = (errCountsRef.current[r.run_id] ?? 0) + 1;
          if ((errCountsRef.current[r.run_id] ?? 0) >= 10) {
            updates[r.run_id] = { state: "failed", state_message: "polling failed repeatedly", finished_at: Date.now() };
          }
        }
      }));
      if (Object.keys(updates).length === 0) return;
      setActiveRuns((cur) => cur.map((r) => updates[r.run_id] ? { ...r, ...updates[r.run_id] } : r));
      // If anything just succeeded, refresh local models so the new fine-tuned
      // model shows up in Playground's Local section immediately.
      if (Object.values(updates).some((u) => u.state === "succeeded")) {
        api.localModels().then(setLocalModels).catch(() => {});
        api.finetuneRuns(25).then(setPastRuns).catch(() => {});
      }
    },
    12000,
    () => !hasInflightFT
  );

  const totalLabels = labelStats?.total ?? 0;
  // The matched count (from preview) is what training will actually receive.
  // Fall back to totalLabels only when nothing narrows the selection and the
  // preview hasn't resolved yet, so the user isn't staring at "0" on first paint.
  const narrowed = useSnapshots || extraFilterCount > 0;
  const matchedLabels = needsSnapshot ? 0 : preview ? preview.total : narrowed ? 0 : totalLabels;
  const canTrain = matchedLabels >= 4 && !!baseModel && !submitting;
  // Mirrors the backend's auto name: ft_<ts>_<base with '/'→'_' and '-' dropped, ≤18 chars>
  const autoName = `ft_<timestamp>_${baseModel.replace(/\//g, "_").replace(/-/g, "").slice(0, 18)}`;

  const submitTrain = async () => {
    if (!canTrain) return;
    setSubmitting(true);
    setError(null);
    try {
      const r = await api.finetuneSubmit({
        ...labelSelection(),
        base_model_name: baseModel,
        uc_model_name: ucModelName.trim() || undefined,
        train_prompt: trainPrompt.trim() || undefined,
        label_filter_instruments: filterInstruments.length ? filterInstruments : undefined,
        num_epochs: numEpochs,
        learning_rate: learningRate,
        lora_r: loraR,
        lora_alpha: loraAlpha,
        lora_dropout: loraDropout,
        per_device_batch_size: batchSize,
        grad_accum_steps: gradAccum,
        warmup_ratio: warmupRatio,
        weight_decay: weightDecay,
        max_length: maxLength,
        accelerator: accelerator,
      });
      setActiveRuns((cur) => [
        {
          run_id: r.run_id,
          databricks_run_id: r.databricks_run_id,
          uc_model_name: r.uc_model_name,
          base_model_name: baseModel,
          n_train: r.n_train,
          state: "running" as const,
          started_at: Date.now(),
        },
        ...cur,
      ].slice(0, 20));
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setSubmitting(false);
    }
  };

  const dismissActive = (id: string) => {
    setActiveRuns((cur) => cur.filter((r) => r.run_id !== id));
  };

  const clearFilters = () => {
    setFilterInstruments([]); setFilterVideos([]); setLabeledSince("");
    setDeltaTable(""); setDeltaVersion("");
  };

  const counting = !needsSnapshot && preview === null && !previewErr;
  const blockedReason =
    needsSnapshot ? "Pick at least one snapshot." :
    matchedLabels < 4 ? (
      counting ? "Counting labels…" :
      labelStats && totalLabels < 4 && dataSource === "lakebase"
        ? "Label at least 4 frames in the Label tab first."
        : `Only ${matchedLabels} labels match. Training needs at least 4.`
    ) :
    !baseModel ? "Pick a base model." : null;

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-semibold">Refine</h1>
        <p className="text-sm text-muted-foreground">
          Fine-tune a local VLM on your labeled frames. The result registers in Unity Catalog and
          shows up as a selectable model in Playground.
        </p>
      </div>

      {/* ── 1 · Which labels ───────────────────────────────────────── */}
      <Card>
        <CardHeader className="flex flex-row items-start justify-between pb-3">
          <div>
            <CardTitle className="text-base">1 · Training labels</CardTitle>
            <CardDescription>Which of your saved labels the model learns from.</CardDescription>
          </div>
          <Button variant="outline" size="sm" onClick={refreshAll}>
            <RefreshCcw className="size-3.5" /> Refresh
          </Button>
        </CardHeader>
        <CardContent className="space-y-4">
          <div>
            <div className="flex flex-wrap items-baseline gap-x-3 gap-y-1">
              <span className={cn(
                "text-3xl font-semibold tabular-nums",
                matchedLabels < 4 && !counting && "text-destructive",
              )}>
                {matchedLabels}
              </span>
              <span className="text-sm text-muted-foreground">
                labels will be used
                {dataSource === "lakebase" && totalLabels > matchedLabels && ` (of ${totalLabels} saved)`}
              </span>
              {matchedLabels >= 4 && matchedLabels < 30 && (
                <span className="text-xs text-amber-600 dark:text-amber-500">small set, expect modest gains</span>
              )}
            </div>
            {previewErr && (
              <p className="mt-2 rounded bg-destructive/10 px-2 py-1 text-[11px] text-destructive">
                Couldn't count matching labels: {previewErr}
              </p>
            )}
            {!needsSnapshot && preview && preview.by_instrument.length > 0 && (
              <div className="mt-2 flex flex-wrap gap-1.5">
                {preview.by_instrument.map((b) => (
                  <Badge key={b.instrument} variant="outline" className="font-normal">
                    {b.instrument} <span className="ml-1 text-muted-foreground">×{b.n}</span>
                  </Badge>
                ))}
              </div>
            )}
          </div>

          {dataSource === "lakebase" ? (
            <div className="space-y-2">
              <div className="flex max-w-md gap-1">
                {([
                  ["all", "All saved labels"],
                  ["snapshots", "Only frames in snapshots"],
                ] as const).map(([v, label]) => (
                  <button
                    key={v}
                    onClick={() => setScope(v)}
                    className={cn(
                      "flex-1 rounded border px-2 py-1 text-xs",
                      scope === v ? "bg-primary text-primary-foreground" : "bg-card hover:bg-accent",
                    )}
                  >
                    {label}
                  </button>
                ))}
              </div>
              {scope === "snapshots" && (
                snapshots.length === 0 ? (
                  <p className="text-[11px] text-muted-foreground">
                    No snapshots yet. Save one from Playground first.
                  </p>
                ) : (
                  <div className="max-h-48 overflow-y-auto rounded-md border">
                    {snapshots.map((s) => (
                      <label
                        key={s.id}
                        className="flex cursor-pointer items-center gap-2 border-b px-2 py-1.5 text-xs last:border-b-0 hover:bg-accent/50"
                      >
                        <Checkbox
                          checked={filterSnapshots.includes(s.id)}
                          onCheckedChange={() => setFilterSnapshots((cur) => toggleIn(cur, s.id))}
                        />
                        <span className="min-w-0 flex-1 truncate">{s.name}</span>
                        <span className="shrink-0 tabular-nums text-muted-foreground">
                          {s.n_frames} frames · {s.created_at?.slice(0, 10)}
                        </span>
                      </label>
                    ))}
                  </div>
                )
              )}
            </div>
          ) : (
            <p className="text-xs text-muted-foreground">
              Reading labels from Delta table <span className="font-mono">{deltaTable.trim()}</span>
              {deltaVersion.trim() && <> at version {deltaVersion.trim()}</>}. Snapshot scoping only
              applies to Lakebase labels.
            </p>
          )}

          <div>
            <button
              onClick={() => setFiltersOpen((o) => !o)}
              className="flex items-center gap-1 text-xs text-muted-foreground hover:text-foreground"
            >
              {filtersOpen ? <ChevronDown className="size-3.5" /> : <ChevronRight className="size-3.5" />}
              More filters
              {extraFilterCount > 0 && (
                <Badge variant="secondary" className="ml-1 h-4 px-1.5 text-[10px] font-normal">
                  {extraFilterCount} active
                </Badge>
              )}
            </button>
            {filtersOpen && (
              <div className="mt-3 space-y-3 rounded-md border bg-muted/30 p-3">
                <div className="space-y-1.5">
                  <Label className="text-[11px] text-muted-foreground">Instrument classes</Label>
                  <div className="flex flex-wrap gap-1">
                    {(labelStats?.by_instrument ?? []).map((b) => (
                      <button
                        key={b.instrument}
                        onClick={() => setFilterInstruments((cur) => toggleIn(cur, b.instrument))}
                        className={cn(
                          "rounded border px-2 py-0.5 text-[11px] transition-colors",
                          filterInstruments.includes(b.instrument)
                            ? "bg-primary text-primary-foreground"
                            : "bg-card hover:bg-accent",
                        )}
                      >
                        {b.instrument} <span className="opacity-60">×{b.n}</span>
                      </button>
                    ))}
                  </div>
                </div>

                <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
                  <div className="space-y-1.5">
                    <Label className="text-[11px] text-muted-foreground">Source videos (comma-separated)</Label>
                    <Input
                      value={filterVideos.join(", ")}
                      placeholder="e.g. knee_arthroscopy_03, lap_chole_01"
                      className="h-8 text-xs"
                      onChange={(e) => setFilterVideos(
                        e.target.value.split(",").map((s) => s.trim()).filter(Boolean)
                      )}
                    />
                  </div>
                  <div className="space-y-1.5">
                    <Label className="text-[11px] text-muted-foreground">Labels added since</Label>
                    <Input
                      type="date"
                      value={labeledSince}
                      className="h-8 text-xs"
                      onChange={(e) => setLabeledSince(e.target.value)}
                    />
                  </div>
                </div>

                <div className="space-y-1.5">
                  <Label className="text-[11px] text-muted-foreground">
                    Read from a Delta table instead of Lakebase
                  </Label>
                  <div className="flex gap-2">
                    <Input
                      value={deltaTable}
                      placeholder={`${ucCatalog}.${ucSchema}.frame_labels_delta`}
                      className="h-8 flex-1 font-mono text-xs"
                      onChange={(e) => setDeltaTable(e.target.value)}
                    />
                    {deltaTable.trim() && (
                      <Input
                        type="number" min={0} step={1}
                        value={deltaVersion}
                        placeholder="latest"
                        className="h-8 w-24 text-xs"
                        title="Delta time-travel: leave blank for latest, or pin to a specific version"
                        onChange={(e) => setDeltaVersion(e.target.value)}
                      />
                    )}
                  </div>
                </div>

                {extraFilterCount > 0 && (
                  <button
                    onClick={clearFilters}
                    className="text-[11px] text-muted-foreground underline hover:text-foreground"
                  >
                    Clear filters
                  </button>
                )}
              </div>
            )}
          </div>
        </CardContent>
      </Card>

      {/* ── 2 · Model + training ───────────────────────────────────── */}
      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-base">2 · Model & training</CardTitle>
          <CardDescription>
            Runs a LoRA fine-tune as a serverless GPU job. Progress shows under Runs below.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
            <div className="space-y-1.5">
              <Label className="text-xs">Base model</Label>
              <Select value={baseModel} onValueChange={setBaseModel}>
                <SelectTrigger><SelectValue placeholder="Pick a local model" /></SelectTrigger>
                <SelectContent>
                  {localModels.filter((m) => m.ready).map((m) => (
                    <SelectItem key={m.name} value={m.name}>
                      {m.display_name}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
            <div className="space-y-1.5">
              <Label className="text-xs">GPU</Label>
              <Select value={accelerator} onValueChange={setAccelerator}>
                <SelectTrigger><SelectValue /></SelectTrigger>
                <SelectContent>
                  {ACCELERATORS.map((a) => (
                    <SelectItem key={a.value} value={a.value}>{a.label}</SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
            <div className="space-y-1.5">
              <Label className="text-xs">Epochs · {numEpochs}</Label>
              <input
                type="range"
                min={1} max={10} step={1}
                value={numEpochs}
                onChange={(e) => setNumEpochs(Number(e.target.value))}
                className="w-full"
              />
              <p className="text-[10px] text-muted-foreground">
                3 is a good default. 6+ risks overfitting a small label set.
              </p>
            </div>
            <div className="space-y-1.5">
              <Label className="text-xs">Capacity</Label>
              <div className="flex gap-1">
                {(["S", "M", "L"] as const).map((c) => (
                  <button
                    key={c}
                    onClick={() => setCapacity(c)}
                    className={cn(
                      "flex-1 rounded border px-2 py-1 text-xs",
                      capacity === c ? "bg-primary text-primary-foreground" : "bg-card hover:bg-accent",
                    )}
                  >
                    {c === "S" ? "Small" : c === "M" ? "Medium" : "Large"}
                  </button>
                ))}
              </div>
              <p className="text-[10px] text-muted-foreground">
                LoRA rank 8 / 16 / 32. Smaller overfits less on few labels.
              </p>
            </div>
          </div>

          <div>
            <button
              onClick={() => setAdvancedOpen((o) => !o)}
              className="flex items-center gap-1 text-xs text-muted-foreground hover:text-foreground"
            >
              {advancedOpen ? <ChevronDown className="size-3.5" /> : <ChevronRight className="size-3.5" />}
              Output name, prompt & advanced parameters
              {customized.length > 0 && (
                <Badge variant="secondary" className="ml-1 h-4 px-1.5 text-[10px] font-normal">
                  custom {customized.join(", ")}
                </Badge>
              )}
            </button>
            {advancedOpen && (
              <div className="mt-3 space-y-4 rounded-md border bg-muted/30 p-3">
                <div className="space-y-1.5">
                  <Label className="text-[11px] text-muted-foreground">UC model name</Label>
                  <Input
                    value={ucModelName}
                    placeholder={`auto: ${autoName}`}
                    className="h-8 text-xs"
                    onChange={(e) => setUcModelName(e.target.value)}
                  />
                </div>

                <div className="space-y-1.5">
                  <div className="flex items-baseline justify-between">
                    <Label className="text-[11px] text-muted-foreground">Training prompt</Label>
                    {trainPrompt.trim() !== DEFAULT_FT_PROMPT && (
                      <button
                        onClick={() => setTrainPrompt(DEFAULT_FT_PROMPT)}
                        className="text-[11px] text-muted-foreground underline hover:text-foreground"
                      >
                        Reset to default
                      </button>
                    )}
                  </div>
                  <Textarea
                    rows={4}
                    value={trainPrompt}
                    onChange={(e) => setTrainPrompt(e.target.value)}
                    className="text-xs"
                  />
                  <p className="text-[10px] text-muted-foreground">
                    Use the same prompt at inference. The model learns to answer in this JSON shape.
                  </p>
                </div>

                <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
                  <div className="space-y-1">
                    <Label className="text-[11px] text-muted-foreground">Learning rate</Label>
                    <Input
                      type="number" step={0.00001} min={0.00001} max={0.001}
                      value={learningRate} className="h-8 text-xs"
                      onChange={(e) => setLearningRate(Number(e.target.value) || DEFAULTS.lr)}
                    />
                  </div>
                  <div className="space-y-1">
                    <Label className="text-[11px] text-muted-foreground">LoRA alpha</Label>
                    <Input
                      type="number" min={1} max={256} step={1}
                      value={loraAlpha} className="h-8 text-xs"
                      onChange={(e) => setLoraAlpha(Number(e.target.value) || DEFAULTS.loraAlpha)}
                    />
                  </div>
                  <div className="space-y-1">
                    <Label className="text-[11px] text-muted-foreground">LoRA dropout</Label>
                    <Input
                      type="number" step={0.01} min={0} max={0.5}
                      value={loraDropout} className="h-8 text-xs"
                      onChange={(e) => setLoraDropout(Number(e.target.value) || 0)}
                    />
                  </div>
                  <div className="space-y-1">
                    <Label className="text-[11px] text-muted-foreground">Batch / device</Label>
                    <Input
                      type="number" min={1} max={32} step={1}
                      value={batchSize} className="h-8 text-xs"
                      onChange={(e) => setBatchSize(Number(e.target.value) || 1)}
                    />
                  </div>
                  <div className="space-y-1">
                    <Label className="text-[11px] text-muted-foreground">
                      Grad accum · effective batch {batchSize * gradAccum}
                    </Label>
                    <Input
                      type="number" min={1} max={64} step={1}
                      value={gradAccum} className="h-8 text-xs"
                      onChange={(e) => setGradAccum(Number(e.target.value) || 1)}
                    />
                  </div>
                  <div className="space-y-1">
                    <Label className="text-[11px] text-muted-foreground">Warmup ratio</Label>
                    <Input
                      type="number" step={0.01} min={0} max={0.5}
                      value={warmupRatio} className="h-8 text-xs"
                      onChange={(e) => setWarmupRatio(Number(e.target.value) || 0)}
                    />
                  </div>
                  <div className="space-y-1">
                    <Label className="text-[11px] text-muted-foreground">Weight decay</Label>
                    <Input
                      type="number" step={0.01} min={0} max={0.5}
                      value={weightDecay} className="h-8 text-xs"
                      onChange={(e) => setWeightDecay(Number(e.target.value) || 0)}
                    />
                  </div>
                  <div className="space-y-1">
                    <Label className="text-[11px] text-muted-foreground">Max sequence length</Label>
                    <Input
                      type="number" min={256} max={4096} step={128}
                      value={maxLength} className="h-8 text-xs"
                      onChange={(e) => setMaxLength(Number(e.target.value) || DEFAULTS.maxLen)}
                    />
                  </div>
                </div>
              </div>
            )}
          </div>

          {error && (
            <p className="flex items-start gap-1 text-xs text-destructive">
              <AlertCircle className="mt-0.5 size-3.5 shrink-0" />
              <span className="break-words">{error}</span>
            </p>
          )}

          <div className="flex flex-wrap items-center gap-3 border-t pt-4">
            <Button onClick={submitTrain} disabled={!canTrain} className="gap-2">
              {submitting
                ? <><Loader2 className="size-4 animate-spin" /> Submitting…</>
                : <><Sparkles className="size-4" /> Fine-tune on {matchedLabels} labels</>}
            </Button>
            <span className="text-xs text-muted-foreground">
              {blockedReason ?? (
                <>Registers as <span className="font-mono">{ucCatalog}.{ucSchema}.{ucModelName.trim() || autoName}</span></>
              )}
            </span>
          </div>
        </CardContent>
      </Card>

      {/* ── Runs: in-flight first, then MLflow history ─────────────── */}
      <Card>
        <CardHeader className="flex flex-row items-start justify-between pb-3">
          <div>
            <CardTitle className="text-base">Runs</CardTitle>
            <CardDescription>
              Finished models appear in Playground under Local.
              {mlflowExperimentPath && <> History is from MLflow experiment <span className="font-mono">{mlflowExperimentPath}</span>.</>}
            </CardDescription>
          </div>
          <Button variant="ghost" size="sm" onClick={refreshRuns} title="Refresh run history">
            <RefreshCcw className="size-3.5" />
          </Button>
        </CardHeader>
        <CardContent className="space-y-2">
          {activeRuns.map((r) => (
            <ActiveRunCard key={r.run_id} run={r} onDismiss={() => dismissActive(r.run_id)} />
          ))}
          {activeRuns.length > 0 && pastRuns.length > 0 && (
            <p className="pt-2 text-[11px] font-medium uppercase tracking-wide text-muted-foreground">History</p>
          )}
          {pastLoading ? (
            <p className="flex items-center gap-2 text-xs text-muted-foreground">
              <Loader2 className="size-3 animate-spin" /> loading…
            </p>
          ) : pastRuns.length === 0 ? (
            activeRuns.length === 0 && <p className="text-xs text-muted-foreground">No runs yet.</p>
          ) : (
            <div className="space-y-1.5">
              {pastRuns.map((r) => (
                <div key={r.mlflow_run_id} className="flex flex-wrap items-center gap-2 rounded border p-2 text-xs">
                  <span className="font-medium">{r.run_name || r.mlflow_run_id.slice(0, 8)}</span>
                  <Badge variant="outline" className="font-normal">{r.base_model || "—"}</Badge>
                  {r.n_train && <span className="text-muted-foreground">{r.n_train} frames</span>}
                  {r.final_train_loss != null && (
                    <span className="text-muted-foreground">loss {r.final_train_loss.toFixed(3)}</span>
                  )}
                  {r.train_elapsed_s != null && (
                    <span className="text-muted-foreground">{fmtElapsed(r.train_elapsed_s * 1000)}</span>
                  )}
                  <Badge
                    variant={r.status === "FINISHED" ? "default" : r.status === "FAILED" ? "destructive" : "secondary"}
                    className="ml-auto font-normal"
                  >
                    {r.status?.toLowerCase() || "—"}
                  </Badge>
                </div>
              ))}
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}

function ActiveRunCard({ run, onDismiss }: { run: ActiveFT; onDismiss: () => void }) {
  const now = Date.now();
  const elapsedMs = (run.finished_at ?? now) - run.started_at;
  return (
    <div className="rounded-md border p-3">
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0 flex-1">
          <div className="flex flex-wrap items-center gap-1.5">
            {run.state === "running" || run.state === "submitting" ? (
              <Loader2 className="size-3.5 animate-spin text-primary" />
            ) : run.state === "succeeded" ? (
              <CheckCircle2 className="size-3.5 text-emerald-500" />
            ) : (
              <XCircle className="size-3.5 text-destructive" />
            )}
            <span className="font-mono text-xs">{run.uc_model_name}</span>
            <Badge variant="outline" className="font-normal">{run.base_model_name}</Badge>
            <span className="text-xs text-muted-foreground">{run.n_train} frames</span>
            <span className="ml-1 text-xs text-muted-foreground">· {fmtElapsed(elapsedMs)}</span>
          </div>
          {run.state === "succeeded" && run.result && (
            <div className="mt-1 text-xs">
              <span className="text-emerald-600 dark:text-emerald-400">
                ✓ registered to <span className="font-mono">{run.result.uc_model}</span>
                {run.result.train_loss != null && (
                  <> · loss {Number(run.result.train_loss).toFixed(3)}</>
                )}
              </span>
              {" "}
              <span className="text-muted-foreground">
                — open Playground → Local section to test it.
              </span>
            </div>
          )}
          {run.state === "failed" && (
            <details className="mt-1">
              <summary className="cursor-pointer text-xs text-destructive font-medium">
                {run.state_message || "Job failed"}
              </summary>
              {run.error_detail && (
                <p className="mt-1 text-xs text-destructive whitespace-pre-wrap break-words">{run.error_detail}</p>
              )}
            </details>
          )}
          {(run.state === "running" || run.state === "submitting") && (
            <p className="mt-1 text-xs text-muted-foreground">
              {run.state_message || "GPU acquiring + loading model + training …"}
            </p>
          )}
        </div>
        <div className="flex shrink-0 flex-col items-end gap-1.5">
          {run.run_page_url && (
            <a
              href={run.run_page_url}
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center gap-1 text-[11px] text-muted-foreground hover:text-foreground"
            >
              <ExternalLink className="size-3" /> Job
            </a>
          )}
          {run.result?.mlflow_url && (
            <a
              href={run.result.mlflow_url}
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center gap-1 text-[11px] text-muted-foreground hover:text-foreground"
            >
              <ExternalLink className="size-3" /> MLflow
            </a>
          )}
          {run.state !== "running" && run.state !== "submitting" && (
            <Button size="sm" variant="ghost" onClick={onDismiss} className="h-6 gap-1 px-2 text-xs">
              <X className="size-3" /> Dismiss
            </Button>
          )}
        </div>
      </div>
    </div>
  );
}
