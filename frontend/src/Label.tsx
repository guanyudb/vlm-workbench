import { useEffect, useMemo, useRef, useState } from "react";
import { Check, CheckCircle2, ChevronLeft, ChevronRight, Database, Loader2, RefreshCcw, Save, Trash2, Wand2, X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { api, type FrameLabel, type RunResultRow, type SnapshotSummary } from "@/api";
import { cn } from "@/lib/utils";

const INSTRUMENT_VOCAB = [
  "probe", "shaver", "burr", "grasper", "biter", "suture_passer",
  "anchor_driver", "electrocautery", "cannula", "scissors", "drill_guide",
  "trocar", "knot_pusher", "rasp", "other_metal_tool", "no_instrument_visible",
];

// Hotkey order — first 9 vocab entries map to digits 1–9.
const HOTKEY_VOCAB = INSTRUMENT_VOCAB.slice(0, 9);

function extractInstrument(parsed: any): string | null {
  if (!parsed) return null;
  if (typeof parsed.instrument === "string") return parsed.instrument;
  const list = parsed.instruments;
  if (Array.isArray(list) && list.length && typeof list[0] === "object") {
    return list[0].class ?? null;
  }
  return null;
}
function extractAnatomy(parsed: any): string | null {
  if (!parsed) return null;
  return parsed.anatomy || null;
}
function extractTissue(parsed: any): string | null {
  if (!parsed) return null;
  return parsed.tissue_condition || null;
}
function frameBasename(p: string): string {
  return p.split("/").filter(Boolean).pop() || p;
}

type StagedFrame = {
  frame_path: string;
  predicted: { instrument: string | null; anatomy: string | null; tissue_condition: string | null; evidence: string | null; model: string };
  existing: FrameLabel | null;
  // Working values — start at predicted (or existing if present); user edits.
  draft: {
    instrument: string;
    anatomy: string;
    tissue_condition: string;
    notes: string;
  };
  status: "pending" | "saved" | "skipped";
};

export default function LabelTab() {
  // ── snapshot picker ─────────────────────────────────────────────────
  const [snapshots, setSnapshots] = useState<SnapshotSummary[]>([]);
  const [selectedSnapshotId, setSelectedSnapshotId] = useState<string>("");
  const [bootstrapping, setBootstrapping] = useState(false);
  const [bootError, setBootError] = useState<string | null>(null);

  // ── workspace ───────────────────────────────────────────────────────
  const [staged, setStaged] = useState<StagedFrame[]>([]);
  const [idx, setIdx] = useState(0);
  const [stats, setStats] = useState<{ total: number; by_instrument: { instrument: string; n: number }[] } | null>(null);

  const refreshStats = () => api.labelsStats().then(setStats).catch(() => {});

  // Sync to Delta — mirrors the Lakebase labels into a Delta table in UC for
  // long-term storage. One button; runs synchronously (<10s for hundreds).
  const [syncing, setSyncing] = useState(false);
  const [syncResult, setSyncResult] = useState<string | null>(null);
  const syncToDelta = async () => {
    setSyncing(true);
    setSyncResult(null);
    try {
      const r = await api.labelsSyncToDelta();
      const ds = r.genai_dataset ? `  ·  GenAI dataset: ${r.genai_dataset}` : "";
      setSyncResult(`Synced ${r.rows_synced} rows → ${r.delta_table}${ds}`);
    } catch (e) {
      setSyncResult(`Sync failed: ${(e as Error).message}`);
    } finally {
      setSyncing(false);
      setTimeout(() => setSyncResult(null), 8000);
    }
  };

  useEffect(() => {
    api.listSnapshots(40).then(setSnapshots).catch(() => {});
    refreshStats();
  }, []);

  const bootstrap = async () => {
    if (!selectedSnapshotId) return;
    setBootstrapping(true);
    setBootError(null);
    try {
      const snap = await api.getSnapshot(selectedSnapshotId);
      const sourceModel = snap.best_model || snap.model_names[0];
      // Build a per-frame view of the chosen model's predictions
      const byFrame = new Map<string, RunResultRow>();
      for (const r of snap.results) {
        if (r.model === sourceModel) byFrame.set(r.frame, r);
      }
      const existingArr = await api.listLabels(snap.frame_paths);
      const existingByPath = new Map(existingArr.map((l) => [l.frame_path, l]));
      const rows: StagedFrame[] = snap.frame_paths.map((path) => {
        const r = byFrame.get(frameBasename(path)) || byFrame.get(path);
        const existing = existingByPath.get(path) || null;
        const predInstrument = r ? extractInstrument(r.parsed) : null;
        const predAnatomy = r ? extractAnatomy(r.parsed) : null;
        const predTissue = r ? extractTissue(r.parsed) : null;
        const evidence = (r?.parsed as any)?.instruments?.[0]?.evidence ?? (r?.parsed as any)?.evidence ?? null;
        return {
          frame_path: path,
          predicted: {
            instrument: predInstrument,
            anatomy: predAnatomy,
            tissue_condition: predTissue,
            evidence,
            model: sourceModel,
          },
          existing,
          // Seed draft from existing if present, else from the prediction.
          draft: {
            instrument: existing?.instrument ?? predInstrument ?? "",
            anatomy: existing?.anatomy ?? predAnatomy ?? "",
            tissue_condition: existing?.tissue_condition ?? predTissue ?? "",
            notes: existing?.notes ?? "",
          },
          status: "pending",
        };
      });
      setStaged(rows);
      setIdx(0);
    } catch (e) {
      setBootError((e as Error).message);
    } finally {
      setBootstrapping(false);
    }
  };

  const current = staged[idx];
  const updateDraft = (patch: Partial<StagedFrame["draft"]>) => {
    setStaged((cur) => cur.map((s, i) => i === idx ? { ...s, draft: { ...s.draft, ...patch } } : s));
  };

  // After save/skip, advance: prefer the next frame whose status is still
  // "pending" (so you don't re-visit ones you already saved), and fall back
  // to plain idx+1 if everything ahead is already done. We compute the next
  // index from the *just-updated* array via functional setStaged — using the
  // outer `staged` ref would be stale and would miss the row we just saved.
  const advanceFrom = (curIdx: number, updated: StagedFrame[]) => {
    const nextPending = updated.findIndex((s, i) => i > curIdx && s.status === "pending");
    if (nextPending >= 0) return nextPending;
    if (curIdx + 1 < updated.length) return curIdx + 1;
    return curIdx; // stay put — at the end
  };

  const saveCurrent = async () => {
    if (!current) return;
    if (!current.draft.instrument) return;
    const myIdx = idx;
    const draft = current.draft;
    const framePath = current.frame_path;
    const predictedModel = current.predicted.model;
    try {
      const saved = await api.upsertLabel({
        frame_path: framePath,
        instrument: draft.instrument,
        anatomy: draft.anatomy,
        tissue_condition: draft.tissue_condition,
        notes: draft.notes,
        source: `snapshot:${selectedSnapshotId}:${predictedModel}`,
      });
      setStaged((cur) => {
        const next = cur.map((s, i) =>
          i === myIdx ? { ...s, status: "saved" as const, existing: saved } : s
        );
        setIdx(advanceFrom(myIdx, next));
        return next;
      });
      refreshStats();
    } catch (e) {
      alert(`Save failed: ${(e as Error).message}`);
    }
  };

  const skipCurrent = () => {
    const myIdx = idx;
    setStaged((cur) => {
      const next = cur.map((s, i) => i === myIdx ? { ...s, status: "skipped" as const } : s);
      setIdx(advanceFrom(myIdx, next));
      return next;
    });
  };

  const deleteCurrent = async () => {
    if (!current || !current.existing) return;
    if (!confirm(`Delete the existing label for ${frameBasename(current.frame_path)}?`)) return;
    try {
      await api.deleteLabel(current.frame_path);
      setStaged((cur) => cur.map((s, i) => i === idx ? { ...s, existing: null, status: "pending" } : s));
      refreshStats();
    } catch (e) {
      alert(`Delete failed: ${(e as Error).message}`);
    }
  };

  // Keyboard shortcuts. Number keys → instrument vocab[0..8]; enter → save+next; → ← navigate; s → skip
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (!current) return;
      // Don't hijack when typing in inputs (except for ctrl+enter to save)
      const tag = (e.target as HTMLElement)?.tagName?.toLowerCase();
      const inField = tag === "input" || tag === "textarea";
      if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) {
        e.preventDefault(); saveCurrent(); return;
      }
      if (inField) return;
      if (e.key === "ArrowRight") { e.preventDefault(); setIdx((i) => Math.min(staged.length - 1, i + 1)); }
      else if (e.key === "ArrowLeft") { e.preventDefault(); setIdx((i) => Math.max(0, i - 1)); }
      else if (e.key === "Enter") { e.preventDefault(); saveCurrent(); }
      else if (e.key === "s" || e.key === "S") { e.preventDefault(); skipCurrent(); }
      else if (/^[1-9]$/.test(e.key)) {
        const v = HOTKEY_VOCAB[parseInt(e.key, 10) - 1];
        if (v) { e.preventDefault(); updateDraft({ instrument: v }); }
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [current?.frame_path, current?.draft.instrument, staged]);

  const counts = useMemo(() => {
    let pending = 0, saved = 0, skipped = 0;
    for (const s of staged) {
      if (s.status === "saved") saved += 1;
      else if (s.status === "skipped") skipped += 1;
      else pending += 1;
    }
    return { pending, saved, skipped };
  }, [staged]);

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-semibold">Label</h1>
        <p className="text-sm text-muted-foreground">
          Bootstrap labels from a Playground snapshot, then verify and save as ground truth.
        </p>
      </div>

      {/* Stats + library overview */}
      <Card>
        <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-3">
          <div>
            <CardTitle className="text-base">Library</CardTitle>
            <CardDescription>{stats?.total ?? 0} labeled frames</CardDescription>
          </div>
          <div className="flex items-center gap-2">
            <Button
              variant="outline"
              size="sm"
              onClick={syncToDelta}
              disabled={syncing || (stats?.total ?? 0) === 0}
              className="gap-1.5"
              title="Mirror these labels to a Delta table in Unity Catalog for long-term storage + downstream Spark/training jobs"
            >
              {syncing ? <Loader2 className="size-3.5 animate-spin" /> : <Database className="size-3.5" />}
              Sync to Delta
            </Button>
            <Button variant="outline" size="sm" onClick={refreshStats}>
              <RefreshCcw className="size-3.5" /> Refresh
            </Button>
          </div>
        </CardHeader>
        <CardContent>
          {stats && stats.by_instrument.length > 0 ? (
            <div className="flex flex-wrap gap-1.5">
              {stats.by_instrument.map((b) => (
                <Badge key={b.instrument} variant="outline" className="font-normal">
                  {b.instrument} <span className="ml-1 text-muted-foreground">×{b.n}</span>
                </Badge>
              ))}
            </div>
          ) : (
            <p className="text-xs text-muted-foreground">No labels yet — bootstrap below.</p>
          )}
          {syncResult && (
            <p className="mt-2 text-[11px] text-muted-foreground">{syncResult}</p>
          )}
        </CardContent>
      </Card>

      {/* Bootstrap section */}
      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-base">Bootstrap from snapshot</CardTitle>
          <CardDescription>
            Pull predictions from a snapshot's best model. You'll verify each one before it lands.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-3">
          <div className="flex flex-wrap items-end gap-3">
            <div className="min-w-72 flex-1">
              <Label className="text-xs">Snapshot</Label>
              <Select value={selectedSnapshotId} onValueChange={setSelectedSnapshotId}>
                <SelectTrigger><SelectValue placeholder="Pick a snapshot…" /></SelectTrigger>
                <SelectContent>
                  {snapshots.map((s) => (
                    <SelectItem key={s.id} value={s.id}>
                      {s.name} · {s.n_frames}f
                      {s.best_model && ` · best: ${s.best_model.replace("databricks-", "")}`}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
            <Button onClick={bootstrap} disabled={!selectedSnapshotId || bootstrapping} className="gap-2">
              {bootstrapping ? <Loader2 className="size-4 animate-spin" /> : <Wand2 className="size-4" />}
              Stage labels
            </Button>
          </div>
          {bootError && <p className="text-xs text-destructive">{bootError}</p>}
        </CardContent>
      </Card>

      {/* Per-frame verify view */}
      {staged.length > 0 && current && (
        <Card>
          <CardHeader className="pb-3">
            <div className="flex flex-wrap items-center justify-between gap-3">
              <div>
                <CardTitle className="text-base">Verify · {idx + 1} of {staged.length}</CardTitle>
                <CardDescription className="font-mono text-xs">{frameBasename(current.frame_path)}</CardDescription>
              </div>
              <div className="flex items-center gap-1.5 text-xs">
                <Badge variant="default">{counts.saved} saved</Badge>
                <Badge variant="outline">{counts.pending} pending</Badge>
                <Badge variant="outline">{counts.skipped} skipped</Badge>
              </div>
            </div>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 gap-4 md:grid-cols-[minmax(0,360px)_1fr]">
              {/* Frame image */}
              <div className="space-y-2">
                <img
                  src={api.frameImageUrl(current.frame_path)}
                  alt={frameBasename(current.frame_path)}
                  className="w-full rounded border bg-black object-contain"
                />
                <div className="flex items-center justify-between text-[11px] text-muted-foreground">
                  <span>← / → navigate · 1-9 instrument · enter save · s skip</span>
                </div>
              </div>

              {/* Form + prediction */}
              <div className="space-y-3">
                {/* Prediction summary card */}
                <div className="rounded-md border border-dashed bg-muted/30 p-3 text-xs">
                  <div className="mb-1.5 flex items-center gap-1.5 text-muted-foreground">
                    <Wand2 className="size-3" /> predicted by {current.predicted.model.replace("databricks-", "")}
                  </div>
                  <div className="flex flex-wrap items-center gap-x-3 gap-y-1">
                    <span>
                      <span className="text-muted-foreground">instrument:</span>{" "}
                      <Badge variant="secondary" className="font-normal">{current.predicted.instrument || "—"}</Badge>
                    </span>
                    {current.predicted.anatomy && (
                      <span>
                        <span className="text-muted-foreground">anatomy:</span> {current.predicted.anatomy}
                      </span>
                    )}
                    {current.predicted.tissue_condition && (
                      <span>
                        <span className="text-muted-foreground">tissue:</span> {current.predicted.tissue_condition}
                      </span>
                    )}
                  </div>
                  {current.predicted.evidence && (
                    <p className="mt-1.5 text-muted-foreground italic">"{current.predicted.evidence}"</p>
                  )}
                  {current.existing && (
                    <div className="mt-2 border-t pt-2 text-emerald-600 dark:text-emerald-400">
                      <CheckCircle2 className="mr-1 inline size-3" />
                      already labeled as <span className="font-medium">{current.existing.instrument}</span>
                      <span className="ml-1 text-muted-foreground">(source: {current.existing.source})</span>
                    </div>
                  )}
                </div>

                {/* Editable form */}
                <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
                  <div>
                    <Label className="text-xs">Instrument</Label>
                    <Select value={current.draft.instrument || undefined} onValueChange={(v) => updateDraft({ instrument: v })}>
                      <SelectTrigger><SelectValue placeholder="Pick…" /></SelectTrigger>
                      <SelectContent>
                        {INSTRUMENT_VOCAB.map((v, i) => (
                          <SelectItem key={v} value={v}>
                            {i < 9 && <span className="mr-1.5 text-[10px] text-muted-foreground">{i + 1}</span>}
                            {v}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </div>
                  <div>
                    <Label className="text-xs">Anatomy</Label>
                    <Input
                      value={current.draft.anatomy}
                      onChange={(e) => updateDraft({ anatomy: e.target.value })}
                      placeholder="(optional)"
                    />
                  </div>
                  <div>
                    <Label className="text-xs">Tissue condition</Label>
                    <Input
                      value={current.draft.tissue_condition}
                      onChange={(e) => updateDraft({ tissue_condition: e.target.value })}
                      placeholder="(optional)"
                    />
                  </div>
                  <div className="md:col-span-2">
                    <Label className="text-xs">Notes</Label>
                    <Textarea
                      rows={2}
                      value={current.draft.notes}
                      onChange={(e) => updateDraft({ notes: e.target.value })}
                      placeholder="(optional)"
                    />
                  </div>
                </div>

                <div className="flex flex-wrap gap-2 pt-1">
                  <Button onClick={saveCurrent} variant="default" className="gap-2" disabled={!current.draft.instrument}>
                    <Save className="size-4" /> Save & next
                  </Button>
                  <Button onClick={skipCurrent} variant="outline">
                    Skip
                  </Button>
                  {current.existing && (
                    <Button onClick={deleteCurrent} variant="ghost" className="gap-2 text-destructive">
                      <Trash2 className="size-3.5" /> Delete existing
                    </Button>
                  )}
                  <div className="flex-1" />
                  <Button variant="outline" disabled={idx === 0} onClick={() => setIdx((i) => Math.max(0, i - 1))}>
                    <ChevronLeft className="size-4" />
                  </Button>
                  <Button variant="outline" disabled={idx >= staged.length - 1} onClick={() => setIdx((i) => Math.min(staged.length - 1, i + 1))}>
                    <ChevronRight className="size-4" />
                  </Button>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Strip thumbnail navigator */}
      {staged.length > 1 && (
        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-base">Frames</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="flex flex-wrap gap-1.5">
              {staged.map((s, i) => (
                <button
                  key={s.frame_path}
                  onClick={() => setIdx(i)}
                  className={cn(
                    "relative h-12 w-16 overflow-hidden rounded border transition-all",
                    i === idx ? "border-primary ring-2 ring-primary/30" : "border-border hover:border-primary/60",
                  )}
                  title={frameBasename(s.frame_path)}
                >
                  <img src={api.frameImageUrl(s.frame_path)} className="h-full w-full object-cover" />
                  <span className={cn(
                    "absolute right-0.5 top-0.5 inline-block size-2 rounded-full",
                    s.status === "saved" ? "bg-emerald-500" :
                    s.status === "skipped" ? "bg-muted-foreground" : "bg-amber-500"
                  )} />
                </button>
              ))}
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
