import { useEffect, useMemo, useState } from "react";
import { CheckCircle2, ChevronDown, ChevronRight, Loader2, XCircle } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Checkbox } from "@/components/ui/checkbox";
import { Badge } from "@/components/ui/badge";
import { api, type RunResultRow, type SnapshotDetail, type SnapshotSummary } from "@/api";
import { cn } from "@/lib/utils";

function extractInstrument(parsed: any): string {
  if (!parsed || typeof parsed !== "object") return "?";
  if (typeof parsed.instrument === "string") {
    return parsed.instrument.trim() || "no_instrument_visible";
  }
  const list = parsed.instruments;
  if (Array.isArray(list)) {
    if (list.length === 0) return "no_instrument_visible";
    if (typeof list[0] === "object") {
      const c = list[0]?.class;
      if (typeof c === "string") return c.trim() || "no_instrument_visible";
    }
  }
  return "no_instrument_visible";
}

function frameBasename(p: string): string {
  return p.split("/").filter(Boolean).pop() || p;
}

interface ResultKey { snapId: string; model: string; framePath: string; }
function rowKey(k: ResultKey): string { return `${k.snapId}::${k.model}::${k.framePath}`; }

export default function Compare() {
  const [summaries, setSummaries] = useState<SnapshotSummary[]>([]);
  const [loadingList, setLoadingList] = useState(false);
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set());
  const [snapshots, setSnapshots] = useState<Record<string, SnapshotDetail>>({});
  const [loadingDetail, setLoadingDetail] = useState(false);
  const [expandedPrompts, setExpandedPrompts] = useState<Set<string>>(new Set());
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setLoadingList(true);
    api.listSnapshots(100)
      .then(setSummaries)
      .catch((e) => setError(String(e)))
      .finally(() => setLoadingList(false));
  }, []);

  const toggleSnapshot = (id: string) => {
    setSelectedIds((cur) => {
      const next = new Set(cur);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  };

  const loadDetails = async () => {
    setLoadingDetail(true);
    setError(null);
    try {
      const missing = Array.from(selectedIds).filter((id) => !snapshots[id]);
      const fetched = await Promise.all(missing.map((id) => api.getSnapshot(id)));
      setSnapshots((cur) => {
        const next = { ...cur };
        fetched.forEach((d) => { next[d.id] = d; });
        return next;
      });
    } catch (e) {
      setError(String(e));
    } finally {
      setLoadingDetail(false);
    }
  };

  const selectedDetails = Array.from(selectedIds)
    .map((id) => snapshots[id])
    .filter((d): d is SnapshotDetail => Boolean(d));

  // Index results: { snapId -> { framePath -> { model -> row } } }
  const resultIndex = useMemo(() => {
    const idx: Record<string, Record<string, Record<string, RunResultRow>>> = {};
    selectedDetails.forEach((s) => {
      idx[s.id] = {};
      s.results.forEach((r) => {
        if (!idx[s.id][r.frame]) idx[s.id][r.frame] = {};
        idx[s.id][r.frame][r.model] = r;
      });
    });
    return idx;
  }, [selectedDetails]);

  // Frames common to all selected
  const commonFrames = useMemo(() => {
    if (selectedDetails.length === 0) return [];
    const sets = selectedDetails.map((s) => new Set(s.frame_paths));
    const first = sets[0];
    return Array.from(first).filter((f) => sets.every((set) => set.has(f))).sort();
  }, [selectedDetails]);

  // Frames in any selected snapshot
  const allFrames = useMemo(() => {
    const set = new Set<string>();
    selectedDetails.forEach((s) => s.frame_paths.forEach((p) => set.add(p)));
    return Array.from(set).sort();
  }, [selectedDetails]);

  const togglePrompt = (id: string) => {
    setExpandedPrompts((cur) => {
      const next = new Set(cur);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  };

  const detailsReady = selectedDetails.length === selectedIds.size && selectedIds.size > 0;

  // Agreement metric: across each common frame, count distinct best-model predictions across snapshots
  const agreementStats = useMemo(() => {
    if (selectedDetails.length < 2 || commonFrames.length === 0) return null;
    let agree = 0;
    let total = 0;
    commonFrames.forEach((f) => {
      const labelsBySnap: string[] = selectedDetails.map((s) => {
        const best = s.best_model && s.best_model in (resultIndex[s.id]?.[f] || {})
          ? s.best_model
          : Object.keys(resultIndex[s.id]?.[f] || {})[0];
        if (!best) return "";
        const r = resultIndex[s.id]?.[f]?.[best];
        return r?.parsed ? extractInstrument(r.parsed) : "";
      });
      if (labelsBySnap.every((l) => l)) {
        total += 1;
        const uniq = new Set(labelsBySnap);
        if (uniq.size === 1) agree += 1;
      }
    });
    return { agree, total };
  }, [selectedDetails, commonFrames, resultIndex]);

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-semibold">Compare</h1>
        <p className="text-sm text-muted-foreground">
          Pick 2+ snapshots to see how their prompts and models agree on shared frames.
        </p>
      </div>

      {error && (
        <div className="rounded-md border border-destructive/50 bg-destructive/10 px-3 py-2 text-sm text-destructive">
          {error}
        </div>
      )}

      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-base">Snapshots</CardTitle>
          <CardDescription>
            {loadingList ? "loading…" : `${summaries.length} snapshots — ${selectedIds.size} selected`}
          </CardDescription>
        </CardHeader>
        <CardContent>
          {summaries.length === 0 && !loadingList ? (
            <p className="text-sm text-muted-foreground">No snapshots yet — save one from Playground first.</p>
          ) : (
            <div className="grid grid-cols-1 gap-2 md:grid-cols-2">
              {summaries.map((s) => {
                const sel = selectedIds.has(s.id);
                return (
                  <label
                    key={s.id}
                    className={cn(
                      "flex cursor-pointer items-start gap-3 rounded-md border p-3 transition-colors",
                      sel ? "border-primary bg-primary/5" : "hover:bg-accent",
                    )}
                  >
                    <Checkbox checked={sel} onCheckedChange={() => toggleSnapshot(s.id)} className="mt-0.5" />
                    <div className="min-w-0 flex-1">
                      <div className="truncate text-sm font-medium">{s.name}</div>
                      <div className="mt-1 flex flex-wrap items-center gap-1.5 text-xs text-muted-foreground">
                        <span>{s.n_frames} frames</span>
                        <span>·</span>
                        <span>{s.n_models} models</span>
                        {s.best_model && (<><span>·</span><Badge variant="outline" className="font-normal">best: {s.best_model}</Badge></>)}
                      </div>
                      <div className="mt-0.5 text-[11px] text-muted-foreground">
                        {new Date(s.created_at).toLocaleString()}
                      </div>
                    </div>
                  </label>
                );
              })}
            </div>
          )}
          <div className="mt-4 flex items-center justify-between">
            <div className="text-xs text-muted-foreground">
              {selectedIds.size < 2 ? "Pick at least 2 to compare." : `${selectedIds.size} ready`}
            </div>
            <Button onClick={loadDetails} disabled={selectedIds.size < 2 || loadingDetail}>
              {loadingDetail ? <Loader2 className="size-4 animate-spin" /> : "Load comparison"}
            </Button>
          </div>
        </CardContent>
      </Card>

      {detailsReady && (
        <>
          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="text-base">Prompts</CardTitle>
              <CardDescription>
                {selectedDetails.length} snapshots · {commonFrames.length} shared frames · {allFrames.length} total frames
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-3">
              {selectedDetails.map((s) => {
                const open = expandedPrompts.has(s.id);
                return (
                  <div key={s.id} className="rounded-md border p-3">
                    <button
                      onClick={() => togglePrompt(s.id)}
                      className="flex w-full items-start gap-2 text-left"
                    >
                      {open ? <ChevronDown className="mt-0.5 size-4 shrink-0" /> : <ChevronRight className="mt-0.5 size-4 shrink-0" />}
                      <div className="min-w-0 flex-1">
                        <div className="truncate text-sm font-medium">{s.name}</div>
                        <div className="mt-1 flex flex-wrap gap-1">
                          {s.model_names.map((m) => (
                            <Badge
                              key={m}
                              variant={m === s.best_model ? "default" : "outline"}
                              className="text-[10px] font-normal"
                            >
                              {m}
                            </Badge>
                          ))}
                        </div>
                      </div>
                    </button>
                    {open && (
                      <pre className="mt-2 max-h-64 overflow-auto whitespace-pre-wrap rounded border bg-muted/40 p-2 text-xs">
                        {s.prompt}
                      </pre>
                    )}
                  </div>
                );
              })}
            </CardContent>
          </Card>

          {agreementStats && agreementStats.total > 0 && (
            <Card>
              <CardHeader className="pb-3">
                <CardTitle className="text-base">Agreement (best model per snapshot, shared frames)</CardTitle>
                <CardDescription>
                  Counts frames where every snapshot's best model produces the same instrument label.
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-semibold tabular-nums">
                  {agreementStats.agree} / {agreementStats.total}
                  <span className="ml-2 text-base font-normal text-muted-foreground">
                    ({((agreementStats.agree / agreementStats.total) * 100).toFixed(0)}%)
                  </span>
                </div>
              </CardContent>
            </Card>
          )}

          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="text-base">Per-frame predictions</CardTitle>
              <CardDescription>
                Showing {commonFrames.length === 0 ? `union of ${allFrames.length} frames (no shared frames)` : `${commonFrames.length} shared frames`}
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              {(commonFrames.length === 0 ? allFrames : commonFrames).map((framePath) => (
                <FrameRow
                  key={framePath}
                  framePath={framePath}
                  snapshots={selectedDetails}
                  resultIndex={resultIndex}
                />
              ))}
            </CardContent>
          </Card>
        </>
      )}
    </div>
  );
}

function FrameRow({
  framePath,
  snapshots,
  resultIndex,
}: {
  framePath: string;
  snapshots: SnapshotDetail[];
  resultIndex: Record<string, Record<string, Record<string, RunResultRow>>>;
}) {
  // Per snapshot, gather (model, prediction) pairs
  const perSnapshot = snapshots.map((s) => {
    const byModel = resultIndex[s.id]?.[framePath] || {};
    return {
      snap: s,
      rows: s.model_names.map((m) => ({ model: m, row: byModel[m] })),
    };
  });

  // Distinct labels across all (best model from each snap) for the agreement chip
  const bestLabels = perSnapshot.map(({ snap, rows }) => {
    const best = snap.best_model && rows.find((r) => r.model === snap.best_model)?.row
      ? snap.best_model
      : rows[0]?.model;
    const r = best ? (resultIndex[snap.id]?.[framePath]?.[best]) : undefined;
    return r?.parsed ? extractInstrument(r.parsed) : "";
  });
  const allBestPresent = bestLabels.every((l) => l);
  const allAgree = allBestPresent && new Set(bestLabels).size === 1;

  return (
    <div className="rounded-md border">
      <div className="flex items-center justify-between border-b bg-muted/30 px-3 py-2">
        <div className="flex items-center gap-2">
          <span className="font-mono text-xs">{frameBasename(framePath)}</span>
          {allBestPresent && (
            <Badge variant={allAgree ? "default" : "destructive"} className="font-normal">
              {allAgree ? <><CheckCircle2 className="mr-1 size-3" />agree: {bestLabels[0]}</> : <><XCircle className="mr-1 size-3" />disagree</>}
            </Badge>
          )}
        </div>
      </div>
      <div className="grid grid-cols-1 gap-3 p-3 md:grid-cols-[180px_1fr]">
        <div>
          <img
            src={api.frameImageUrl(framePath)}
            alt={frameBasename(framePath)}
            className="h-32 w-full rounded border object-cover"
            loading="lazy"
          />
        </div>
        <div className="space-y-2">
          {perSnapshot.map(({ snap, rows }) => (
            <div key={snap.id} className="rounded border bg-card/40 p-2">
              <div className="mb-1 truncate text-xs font-medium">{snap.name}</div>
              <div className="flex flex-wrap gap-x-3 gap-y-1">
                {rows.map(({ model, row }) => {
                  const label = row?.parsed ? extractInstrument(row.parsed) : (row?.ok ? "?" : "—");
                  const isBest = model === snap.best_model;
                  return (
                    <div key={model} className="flex items-center gap-1.5">
                      <span className={cn("text-[11px]", isBest ? "font-medium" : "text-muted-foreground")}>
                        {model}:
                      </span>
                      <Badge
                        variant={row?.ok ? (isBest ? "default" : "secondary") : "destructive"}
                        className="font-normal"
                      >
                        {label}
                      </Badge>
                    </div>
                  );
                })}
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
