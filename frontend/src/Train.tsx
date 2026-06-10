import { useEffect, useMemo, useRef, useState } from "react";
import {
  AlertCircle, CheckCircle2, ExternalLink, Loader2, Play, RefreshCcw,
  Settings, Sparkles, Wand2, X, XCircle,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
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
import { cn } from "@/lib/utils";

const DEFAULT_FT_PROMPT = `Identify the surgical instrument by its visible visual features. Respond with strict JSON: {"instrument": "<class>", "anatomy": "<short>", "tissue_condition": "<short>"} where <class> is one of: probe, shaver, burr, grasper, biter, suture_passer, anchor_driver, electrocautery, cannula, scissors, drill_guide, trocar, knot_pusher, rasp, other_metal_tool, no_instrument_visible.`;

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
  run_page_url?: string | null;
  result?: any;
  started_at: number;
  finished_at?: number;
};

// Serverless GPU options surfaced for fine-tune jobs. Names must match the
// Jobs API exactly (`GPU_<N>x<TYPE>` — no `Nx` prefix → CPU silent
// downgrade). H100 was added back per the user's workspace
// enablement (e2-demo-field-eng has GPU_1xH100 available); availability
// elsewhere depends on the workspace's serverless-GPU SKU list.
const ACCELERATORS = [
  { value: "GPU_1xA10",  label: "1× A10 · 24 GB · cheap (best for 8B base)" },
  { value: "GPU_1xH100", label: "1× H100 · 80 GB · fast (best for 12B+ base)" },
  { value: "GPU_8xH100", label: "8× H100 · 640 GB · fastest (overkill for LoRA)" },
];

export default function Train() {
  // UC layout for label-rendering only. Pulled from /api/health so the
  // workbench shows the *actual* registered name regardless of workspace.
  const [ucCatalog, setUcCatalog] = useState<string>("<catalog>");
  const [ucSchema, setUcSchema] = useState<string>("<schema>");
  useEffect(() => {
    api.health().then((h) => { setUcCatalog(h.uc_catalog); setUcSchema(h.uc_schema); }).catch(() => {});
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
  // Filters (all AND-composed; empty = no restriction)
  const [filterSnapshots, setFilterSnapshots] = usePersistentState<string[]>("vlmwb.train.snapshots.v2", []);
  const [filterInstruments, setFilterInstruments] = usePersistentState<string[]>("vlmwb.train.instruments", []);
  const [filterVideos, setFilterVideos] = usePersistentState<string[]>("vlmwb.train.videos", []);
  const [labeledSince, setLabeledSince] = usePersistentState<string>("vlmwb.train.since", "");
  const [trainPrompt, setTrainPrompt] = usePersistentState<string>("vlmwb.train.prompt", DEFAULT_FT_PROMPT);
  // Hyperparams. Beginner panel: epochs + capacity preset. Advanced collapses
  // the rest. Capacity preset maps to LoRA r so users don't see "rank".
  const [numEpochs, setNumEpochs] = usePersistentState<number>("vlmwb.train.epochs", 3);
  const [capacity, setCapacity] = usePersistentState<"S" | "M" | "L">("vlmwb.train.capacity", "M");
  const [learningRate, setLearningRate] = usePersistentState<number>("vlmwb.train.lr", 0.0002);
  const [loraAlpha, setLoraAlpha] = usePersistentState<number>("vlmwb.train.loraAlpha", 32);
  const [loraDropout, setLoraDropout] = usePersistentState<number>("vlmwb.train.loraDropout", 0.05);
  const [batchSize, setBatchSize] = usePersistentState<number>("vlmwb.train.batch", 1);
  const [gradAccum, setGradAccum] = usePersistentState<number>("vlmwb.train.gradAccum", 8);
  const [warmupRatio, setWarmupRatio] = usePersistentState<number>("vlmwb.train.warmup", 0.03);
  const [weightDecay, setWeightDecay] = usePersistentState<number>("vlmwb.train.wd", 0.0);
  const [maxLength, setMaxLength] = usePersistentState<number>("vlmwb.train.maxLen", 1024);
  const [accelerator, setAccelerator] = usePersistentState<string>("vlmwb.train.accel", "GPU_1xA10");
  const [advancedOpen, setAdvancedOpen] = useState(false);
  // Capacity preset → LoRA rank. Higher = more learnable params + more
  // overfit risk on small datasets. 16 is a reasonable middle for the
  // typical workbench dataset size (10–200 labels).
  const loraR = capacity === "S" ? 8 : capacity === "L" ? 32 : 16;

  // Live preview: how many labels actually match the current filters
  const [preview, setPreview] = useState<{ total: number; by_instrument: { instrument: string; n: number }[] } | null>(null);
  useEffect(() => {
    const t = setTimeout(() => {
      api.finetunePreview({
        instruments: filterInstruments.length ? filterInstruments : undefined,
        snapshot_ids: filterSnapshots.length ? filterSnapshots : undefined,
        video_names: filterVideos.length ? filterVideos : undefined,
        labeled_since: labeledSince || undefined,
      }).then(setPreview).catch(() => setPreview(null));
    }, 300);  // debounce filter clicks
    return () => clearTimeout(t);
  }, [
    filterInstruments.join(","),
    filterSnapshots.join(","),
    filterVideos.join(","),
    labeledSince,
  ]);

  // ── active runs (persisted) ───────────────────────────────────────
  const [activeRuns, setActiveRuns] = usePersistentState<ActiveFT[]>("vlmwb.train.active", []);
  const errCountsRef = useRef<Record<string, number>>({});
  const [submitting, setSubmitting] = useState(false);

  const refreshAll = () => {
    api.localModels().then(setLocalModels).catch(() => {});
    api.labelsStats().then(setLabelStats).catch(() => {});
    api.listSnapshots(30).then(setSnapshots).catch(() => {});
    setPastLoading(true);
    api.finetuneRuns(25).then(setPastRuns).catch(() => {}).finally(() => setPastLoading(false));
  };
  useEffect(refreshAll, []);

  // Poller for in-flight fine-tune jobs. Tolerant of transient 502s — only
  // gives up after 10 consecutive errors per run.
  useEffect(() => {
    const inflight = activeRuns.filter((r) => r.state === "running" || r.state === "submitting");
    if (inflight.length === 0) return;
    let stopped = false;
    const poll = async () => {
      if (stopped) return;
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
      if (stopped || Object.keys(updates).length === 0) return;
      setActiveRuns((cur) => cur.map((r) => updates[r.run_id] ? { ...r, ...updates[r.run_id] } : r));
      // If anything just succeeded, refresh local models so the new fine-tuned
      // model shows up in Playground's Local section immediately.
      if (Object.values(updates).some((u) => u.state === "succeeded")) {
        api.localModels().then(setLocalModels).catch(() => {});
        api.finetuneRuns(25).then(setPastRuns).catch(() => {});
      }
    };
    poll();
    const interval = setInterval(poll, 12000);
    return () => { stopped = true; clearInterval(interval); };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeRuns.filter((r) => r.state === "running" || r.state === "submitting").map((r) => r.run_id).join(",")]);

  const totalLabels = labelStats?.total ?? 0;
  // The matched count (from preview) is what training will actually receive.
  // Fall back to totalLabels when no filters are set and the preview hasn't
  // resolved yet, so the user isn't staring at "0" on first paint.
  const matchedLabels =
    (preview?.total ?? null) !== null
      ? preview!.total
      : (filterInstruments.length || filterSnapshots.length || filterVideos.length || labeledSince
          ? 0
          : totalLabels);
  const canTrain = matchedLabels >= 4 && !!baseModel && !submitting;

  // Derived: every distinct video that has at least one labeled frame.
  // Source the list from extracted_frames index via the videos field on the
  // labels stats response — but we don't have that endpoint yet. Cheap
  // workaround: parse the frame_paths in `preview` if present; otherwise
  // derive from snapshots' frame_paths. (The chosen UI doesn't NEED a
  // strict list — users can type a video name and we'll do a LIKE match
  // server-side. So we surface the list as a convenience but accept any
  // string.) For Phase 1 we just expose snapshot names + a free-form video
  // field; videos filter is "tag-input" style.
  const submitTrain = async () => {
    if (!canTrain) return;
    setSubmitting(true);
    setError(null);
    try {
      const r = await api.finetuneSubmit({
        base_model_name: baseModel,
        uc_model_name: ucModelName.trim() || undefined,
        train_prompt: trainPrompt.trim() || undefined,
        label_filter_instruments: filterInstruments.length ? filterInstruments : undefined,
        snapshot_ids: filterSnapshots.length ? filterSnapshots : undefined,
        video_names: filterVideos.length ? filterVideos : undefined,
        labeled_since: labeledSince || undefined,
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
          state: "running",
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

  // ── Toggle helpers for multi-select filters ─────────────────────────
  const toggleIn = <T,>(arr: T[], v: T): T[] =>
    arr.includes(v) ? arr.filter((x) => x !== v) : [...arr, v];
  const TOGGLED = (arr: string[], v: string) =>
    arr.includes(v) ? "bg-primary text-primary-foreground" : "bg-card hover:bg-accent";

  const dismissActive = (id: string) => {
    setActiveRuns((cur) => cur.filter((r) => r.run_id !== id));
  };

  const elapsed = (ms: number) => {
    const s = Math.max(0, Math.round(ms / 1000));
    if (s < 60) return `${s}s`;
    const m = Math.floor(s / 60);
    return `${m}m ${s % 60}s`;
  };

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-semibold">Refine</h1>
        <p className="text-sm text-muted-foreground">
          Fine-tune a local VLM on your labeled frames. The result registers in Unity Catalog and
          shows up as a selectable model in Playground.
        </p>
      </div>

      {/* Eligibility / data summary */}
      <Card>
        <CardHeader className="flex flex-row items-start justify-between pb-3">
          <div>
            <CardTitle className="text-base">Training data</CardTitle>
            <CardDescription>
              Sourced live from the labels table — every saved label is included.
            </CardDescription>
          </div>
          <Button variant="outline" size="sm" onClick={refreshAll}>
            <RefreshCcw className="size-3.5" /> Refresh
          </Button>
        </CardHeader>
        <CardContent>
          <div className="flex items-baseline gap-4">
            <span className="text-3xl font-semibold tabular-nums">{totalLabels}</span>
            <span className="text-sm text-muted-foreground">labeled frames</span>
            {totalLabels > 0 && totalLabels < 4 && (
              <span className="text-xs text-destructive">need ≥ 4 to start</span>
            )}
            {totalLabels >= 4 && totalLabels < 30 && (
              <span className="text-xs text-amber-600">small dataset — gains will be modest</span>
            )}
          </div>
          {labelStats && labelStats.by_instrument.length > 0 && (
            <div className="mt-3 flex flex-wrap gap-1.5">
              {labelStats.by_instrument.map((b) => (
                <Badge key={b.instrument} variant="outline" className="font-normal">
                  {b.instrument} <span className="ml-1 text-muted-foreground">×{b.n}</span>
                </Badge>
              ))}
            </div>
          )}
        </CardContent>
      </Card>

      {/* Start training */}
      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-base">Start fine-tune</CardTitle>
          <CardDescription>
            Base model + a label set → a LoRA-fine-tuned variant registered to UC. Typical wall-clock:
            6–15 min on 1×A10 with ~30 frames, 8–20 min on 8×H100.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-3">
          <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
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
              <p className="text-[10px] text-muted-foreground">
                Only ready local models can be the base. Add more via the setup_cache notebook.
              </p>
            </div>
            <div className="space-y-1.5">
              <Label className="text-xs">UC model name (optional)</Label>
              <Input
                value={ucModelName}
                placeholder="auto (vlmwb_ft_<ts>)"
                onChange={(e) => setUcModelName(e.target.value)}
              />
              <p className="text-[10px] text-muted-foreground">
                Registers as <span className="font-mono">{ucCatalog}.{ucSchema}.{ucModelName || "vlmwb_ft_<ts>"}</span>.
              </p>
            </div>
          </div>

          {/* ── Label filters ─────────────────────────────────────────
              Filters compose with AND. Empty = no restriction. Live preview
              shows the exact count training will consume so the user knows
              what they're committing to before clicking Start. */}
          <div className="rounded-md border bg-muted/30 p-3 space-y-3">
            <div className="flex items-baseline justify-between">
              <Label className="text-xs font-medium">Label filters</Label>
              <div className="text-xs">
                {preview ? (
                  <span className={cn(
                    preview.total < 4 ? "text-destructive" :
                    preview.total < 30 ? "text-amber-600 dark:text-amber-500" :
                    "text-muted-foreground",
                  )}>
                    <span className="font-medium tabular-nums">{preview.total}</span> labels matched
                  </span>
                ) : (
                  <span className="text-muted-foreground">…</span>
                )}
              </div>
            </div>

            <div className="space-y-1.5">
              <Label className="text-[11px] text-muted-foreground">Snapshots (multi-select)</Label>
              <div className="flex flex-wrap gap-1">
                {snapshots.length === 0 && (
                  <span className="text-[11px] text-muted-foreground/70">No snapshots yet — save one from Playground first to use this filter.</span>
                )}
                {snapshots.map((s) => (
                  <button
                    key={s.id}
                    onClick={() => setFilterSnapshots((cur) => toggleIn(cur, s.id))}
                    className={cn(
                      "rounded border px-2 py-0.5 text-[11px] transition-colors",
                      TOGGLED(filterSnapshots, s.id),
                    )}
                    title={`${s.n_frames} frames · ${s.created_at?.slice(0, 10)}`}
                  >
                    {s.name}
                  </button>
                ))}
              </div>
            </div>

            <div className="space-y-1.5">
              <Label className="text-[11px] text-muted-foreground">Instrument classes (multi-select)</Label>
              <div className="flex flex-wrap gap-1">
                {(labelStats?.by_instrument ?? []).map((b) => (
                  <button
                    key={b.instrument}
                    onClick={() => setFilterInstruments((cur) => toggleIn(cur, b.instrument))}
                    className={cn(
                      "rounded border px-2 py-0.5 text-[11px] transition-colors",
                      TOGGLED(filterInstruments, b.instrument),
                    )}
                  >
                    {b.instrument} <span className="opacity-60">×{b.n}</span>
                  </button>
                ))}
              </div>
            </div>

            <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
              <div className="space-y-1.5">
                <Label className="text-[11px] text-muted-foreground">Source videos (comma-separated, optional)</Label>
                <Input
                  value={filterVideos.join(", ")}
                  placeholder="e.g. VinayVideoThing1, knee_arthroscopy_03"
                  className="h-8 text-xs"
                  onChange={(e) => setFilterVideos(
                    e.target.value.split(",").map((s) => s.trim()).filter(Boolean)
                  )}
                />
              </div>
              <div className="space-y-1.5">
                <Label className="text-[11px] text-muted-foreground">Labels added since (optional)</Label>
                <Input
                  type="date"
                  value={labeledSince}
                  className="h-8 text-xs"
                  onChange={(e) => setLabeledSince(e.target.value)}
                />
              </div>
            </div>

            {(filterSnapshots.length || filterInstruments.length || filterVideos.length || labeledSince) ? (
              <button
                onClick={() => { setFilterSnapshots([]); setFilterInstruments([]); setFilterVideos([]); setLabeledSince(""); }}
                className="text-[11px] text-muted-foreground hover:text-foreground underline"
              >
                Clear all filters
              </button>
            ) : null}
          </div>

          <div>
            <Label className="text-xs">Training prompt (what the model sees during fine-tune)</Label>
            <Textarea
              rows={5}
              value={trainPrompt}
              onChange={(e) => setTrainPrompt(e.target.value)}
              className="mt-1 text-xs"
            />
            <p className="mt-1 text-[10px] text-muted-foreground">
              Should match the prompt you'll use at inference. The fine-tuned model is trained to
              respond with strict JSON, so anatomy + tissue_condition are populated from your labels.
            </p>
          </div>

          {/* ── Hyperparams — Beginner pane always visible ────────────
              Two knobs that materially change outcomes for the typical
              user: epochs (controls fit vs overfit) and capacity (LoRA
              rank, abstracted as Small/Medium/Large). Everything else
              lives behind the Advanced toggle. */}
          <div className="rounded-md border bg-muted/30 p-3 space-y-3">
            <Label className="text-xs font-medium">Training parameters</Label>
            <div className="grid grid-cols-1 gap-3 md:grid-cols-3">
              <div className="space-y-1.5">
                <Label className="text-[11px] text-muted-foreground">Epochs · {numEpochs}</Label>
                <input
                  type="range"
                  min={1} max={10} step={1}
                  value={numEpochs}
                  onChange={(e) => setNumEpochs(Number(e.target.value))}
                  className="w-full"
                />
                <p className="text-[10px] text-muted-foreground">
                  1–2: light touch · 3: default · 6+: risk of overfitting on small data
                </p>
              </div>
              <div className="space-y-1.5">
                <Label className="text-[11px] text-muted-foreground">Capacity</Label>
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
                  LoRA rank · S=8 (least overfit), M=16, L=32 (most expressive)
                </p>
              </div>
              <div className="space-y-1.5">
                <Label className="text-[11px] text-muted-foreground">Accelerator</Label>
                <Select value={accelerator} onValueChange={setAccelerator}>
                  <SelectTrigger className="h-8 text-xs"><SelectValue /></SelectTrigger>
                  <SelectContent>
                    {ACCELERATORS.map((a) => (
                      <SelectItem key={a.value} value={a.value}>{a.label}</SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
            </div>

            <div>
              <button
                onClick={() => setAdvancedOpen((o) => !o)}
                className="flex items-center gap-1.5 text-[11px] text-muted-foreground hover:text-foreground"
              >
                <Settings className="size-3" />
                {advancedOpen ? "Hide" : "Show"} advanced parameters
              </button>
              {advancedOpen && (
                <div className="mt-3 grid grid-cols-2 gap-3 md:grid-cols-4">
                  <div className="space-y-1">
                    <Label className="text-[11px] text-muted-foreground">Learning rate</Label>
                    <Input
                      type="number" step={0.00001} min={0.00001} max={0.001}
                      value={learningRate} className="h-8 text-xs"
                      onChange={(e) => setLearningRate(Number(e.target.value) || 2e-4)}
                    />
                  </div>
                  <div className="space-y-1">
                    <Label className="text-[11px] text-muted-foreground">LoRA alpha</Label>
                    <Input
                      type="number" min={1} max={256} step={1}
                      value={loraAlpha} className="h-8 text-xs"
                      onChange={(e) => setLoraAlpha(Number(e.target.value) || 32)}
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
                      onChange={(e) => setMaxLength(Number(e.target.value) || 1024)}
                    />
                  </div>
                </div>
              )}
            </div>
          </div>

          {error && (
            <p className="flex items-center gap-1 text-xs text-destructive">
              <AlertCircle className="size-3.5" /> {error}
            </p>
          )}

          <div className="flex items-center gap-3 border-t pt-3">
            <Button onClick={submitTrain} disabled={!canTrain} className="gap-2">
              {submitting
                ? <><Loader2 className="size-4 animate-spin" /> Submitting…</>
                : <><Sparkles className="size-4" /> Start fine-tune on {matchedLabels} labels</>}
            </Button>
            {matchedLabels < 4 && totalLabels < 4 && (
              <span className="text-xs text-muted-foreground">
                Label at least 4 frames in the Label tab first.
              </span>
            )}
            {matchedLabels < 4 && totalLabels >= 4 && (
              <span className="text-xs text-muted-foreground">
                Current filters match {matchedLabels} / {totalLabels} labels — relax filters to train.
              </span>
            )}
          </div>
        </CardContent>
      </Card>

      {/* Active runs */}
      {activeRuns.length > 0 && (
        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-base">Active runs</CardTitle>
            <CardDescription>
              Live + recently-finished fine-tunes. Successful runs auto-appear in Playground's Local section.
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-2">
            {activeRuns.map((r) => (
              <ActiveRunCard
                key={r.run_id}
                run={r}
                onDismiss={() => dismissActive(r.run_id)}
                elapsedFmt={elapsed}
              />
            ))}
          </CardContent>
        </Card>
      )}

      {/* Past runs (from MLflow) */}
      <Card>
        <CardHeader className="flex flex-row items-center justify-between pb-3">
          <div>
            <CardTitle className="text-base">Run history</CardTitle>
            <CardDescription>From MLflow experiment {`/Users/guanyu.chen@databricks.com/vlm-workbench`}</CardDescription>
          </div>
          <Button
            variant="ghost"
            size="sm"
            onClick={() => { setPastLoading(true); api.finetuneRuns(25).then(setPastRuns).finally(() => setPastLoading(false)); }}
          >
            <RefreshCcw className="size-3.5" />
          </Button>
        </CardHeader>
        <CardContent>
          {pastLoading ? (
            <p className="flex items-center gap-2 text-xs text-muted-foreground">
              <Loader2 className="size-3 animate-spin" /> loading…
            </p>
          ) : pastRuns.length === 0 ? (
            <p className="text-xs text-muted-foreground">No past runs yet.</p>
          ) : (
            <div className="space-y-1.5">
              {pastRuns.map((r) => (
                <div key={r.mlflow_run_id} className="flex flex-wrap items-center gap-2 rounded border p-2 text-xs">
                  <span className="font-medium">{r.run_name || r.mlflow_run_id.slice(0, 8)}</span>
                  <Badge variant="outline" className="font-normal">{r.base_model || "—"}</Badge>
                  {r.n_train && <span className="text-muted-foreground">n_train={r.n_train}</span>}
                  {r.lora_r && <span className="text-muted-foreground">r={r.lora_r}</span>}
                  {r.final_train_loss != null && (
                    <span className="text-muted-foreground">loss={r.final_train_loss.toFixed(3)}</span>
                  )}
                  {r.train_elapsed_s != null && (
                    <span className="text-muted-foreground">{elapsed(r.train_elapsed_s * 1000)}</span>
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

function ActiveRunCard({
  run, onDismiss, elapsedFmt,
}: {
  run: ActiveFT;
  onDismiss: () => void;
  elapsedFmt: (ms: number) => string;
}) {
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
            <span className="ml-1 text-xs text-muted-foreground">· {elapsedFmt(elapsedMs)}</span>
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
            <p className="mt-1 text-xs text-destructive">
              {run.state_message || "Job failed — see Databricks Job page for details."}
            </p>
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
