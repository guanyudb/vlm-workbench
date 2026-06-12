import { useEffect, useMemo, useRef, useState } from "react";
import {
  AlertCircle, ChevronDown, ChevronRight, Loader2, Play, Sparkles, Volume2,
  VolumeX, Wand2,
} from "lucide-react";
import {
  api,
  type StudioAnalysis,
  type StudioAudioResponse,
  type VideoEntry,
  type SnapshotSummary,
} from "@/api";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";
import {
  Select, SelectContent, SelectItem, SelectTrigger, SelectValue,
} from "@/components/ui/select";
import { cn } from "@/lib/utils";

function fmtTime(s: number | null | undefined): string {
  if (s == null || isNaN(s)) return "—";
  const m = Math.floor(s / 60);
  const r = Math.floor(s % 60);
  return `${m}:${String(r).padStart(2, "0")}`;
}

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
function extractEvidence(parsed: any): string {
  if (!parsed) return "";
  if (typeof parsed.evidence === "string") return parsed.evidence;
  const list = parsed.instruments;
  if (Array.isArray(list) && list.length && typeof list[0] === "object") {
    return list[0]?.evidence ?? "";
  }
  return "";
}
function extractAnatomy(parsed: any): string {
  if (!parsed) return "";
  return parsed.anatomy || "";
}
function extractTissue(parsed: any): string {
  if (!parsed) return "";
  return parsed.tissue_condition || "";
}

/**
 * Studio v2 — single primary view, controls collapsed into a top bar.
 *
 * Mental model: a user wants to watch a procedure and see *what's happening
 * at each moment*. The per-frame model is the workhorse — anything beyond it
 * (sectioning, validator pass) is optional follow-up, hidden in a collapsed
 * panel.
 *
 * Layout:
 *   ┌──────────────────────────────────────────────────────┐
 *   │ video pick · model · snapshot · Analyze · cache state │
 *   ├─────────────────────────┬─────────────────────────────┤
 *   │       video player      │  per-frame timeline (active)│
 *   │  (sticky, ~9/16 width)  │  thumb + chip + evidence   │
 *   │  + audio (collapsible)  │  click → seek video        │
 *   └─────────────────────────┴─────────────────────────────┘
 *   ▾ Procedure summary (collapsible, optional)
 */
export default function Studio({
  initialSnapshotId,
  onConsumeInitialSnapshot,
}: {
  initialSnapshotId?: string | null;
  onConsumeInitialSnapshot?: () => void;
}) {
  // ── data ─────────────────────────────────────────────────────────────
  const [videos, setVideos] = useState<VideoEntry[] | null>(null);
  const [video, setVideo] = useState<string | null>(null);
  const [analysis, setAnalysis] = useState<StudioAnalysis | null>(null);
  const [analyzing, setAnalyzing] = useState(false);
  const [analyzeError, setAnalyzeError] = useState<string | null>(null);
  const [audio, setAudio] = useState<StudioAudioResponse | null>(null);
  const [audioBusy, setAudioBusy] = useState(false);
  const [audioError, setAudioError] = useState<string | null>(null);
  const [snapshots, setSnapshots] = useState<SnapshotSummary[]>([]);
  const [snapshotId, setSnapshotId] = useState<string | null>(initialSnapshotId ?? null);

  // ── playback state ───────────────────────────────────────────────────
  const [activeFrameIdx, setActiveFrameIdx] = useState<number | null>(null);
  const [activeAudioIdx, setActiveAudioIdx] = useState<number | null>(null);
  const [currentTime, setCurrentTime] = useState(0);

  // ── UI toggles ───────────────────────────────────────────────────────
  const [audioOpen, setAudioOpen] = useState(false);
  const [summaryOpen, setSummaryOpen] = useState(false);
  const [readAloud, setReadAloud] = useState(false);
  const [isPlaying, setIsPlaying] = useState(false);

  // ── refs ─────────────────────────────────────────────────────────────
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const frameRowRefs = useRef<Array<HTMLButtonElement | null>>([]);
  const audioRowRefs = useRef<Array<HTMLButtonElement | null>>([]);
  const cancelRef = useRef<(() => void) | null>(null);
  const lastSpokenFrame = useRef<number | null>(null);
  const [chosenVoice, setChosenVoice] = useState<SpeechSynthesisVoice | null>(null);
  const ttsSupported = typeof window !== "undefined" && typeof window.speechSynthesis !== "undefined";
  const VOICE_PREFERENCE = [
    "Google US English Natural 2",
    "Google US English Natural 1",
    "Google US English Natural",
    "Google US English",
  ];

  // Initial load
  useEffect(() => {
    api.videos().then((v) => {
      setVideos(v);
      if (v.length > 0 && !video) setVideo(v[0].name);
    }).catch(() => {});
    api.listSnapshots(20).then(setSnapshots).catch(() => {});
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Consume incoming snapshot pin
  useEffect(() => {
    if (initialSnapshotId && onConsumeInitialSnapshot) onConsumeInitialSnapshot();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Load cached analysis (auto) + audio (manual)
  useEffect(() => {
    if (!video) return;
    setAnalysis(null);
    setAudio(null);
    setAnalyzeError(null);
    setAudioError(null);
    setActiveFrameIdx(null);
    setActiveAudioIdx(null);
    setCurrentTime(0);
    api.studioAnalysis(video).then(setAnalysis).catch(() => {});
    api.studioAudioGet(video).then((cached) => { if (cached) setAudio(cached); }).catch(() => {});
  }, [video]);

  // Track playback time + active frame/audio segment
  useEffect(() => {
    const v = videoRef.current;
    if (!v) return;
    const handler = () => {
      const t = v.currentTime;
      setCurrentTime(t);
      if (analysis?.per_frame?.length) {
        let best = -1; let bestDist = Infinity;
        analysis.per_frame.forEach((f, i) => {
          const ts = f.timestamp_s ?? 0;
          const d = Math.abs(ts - t);
          if (d < bestDist) { bestDist = d; best = i; }
        });
        setActiveFrameIdx(best >= 0 ? best : null);
      }
      if (audio?.segments?.length) {
        const idx = audio.segments.findIndex((s) => t >= s.start && t < s.end);
        setActiveAudioIdx(idx >= 0 ? idx : null);
      }
    };
    v.addEventListener("timeupdate", handler);
    return () => v.removeEventListener("timeupdate", handler);
  }, [analysis, audio]);

  // Keep active row in view
  useEffect(() => {
    if (activeFrameIdx == null) return;
    frameRowRefs.current[activeFrameIdx]?.scrollIntoView({ block: "nearest", behavior: "smooth" });
  }, [activeFrameIdx]);
  useEffect(() => {
    if (activeAudioIdx == null) return;
    audioRowRefs.current[activeAudioIdx]?.scrollIntoView({ block: "nearest", behavior: "smooth" });
  }, [activeAudioIdx]);

  // Resolve locked TTS voice
  useEffect(() => {
    if (!ttsSupported) return;
    const load = () => {
      const all = window.speechSynthesis.getVoices();
      let pick: SpeechSynthesisVoice | undefined;
      for (const target of VOICE_PREFERENCE) {
        pick = all.find((v) => v.name === target);
        if (pick) break;
      }
      if (!pick) pick = all.find((v) => /google/i.test(v.name) && v.lang === "en-US");
      setChosenVoice(pick ?? null);
    };
    load();
    window.speechSynthesis.addEventListener("voiceschanged", load);
    return () => window.speechSynthesis.removeEventListener("voiceschanged", load);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [ttsSupported]);

  // Track play/pause for narration
  useEffect(() => {
    const v = videoRef.current;
    if (!v) return;
    const onPlay = () => setIsPlaying(true);
    const onPause = () => setIsPlaying(false);
    const onSeeking = () => {
      if (ttsSupported) window.speechSynthesis.cancel();
      lastSpokenFrame.current = null;
    };
    v.addEventListener("play", onPlay);
    v.addEventListener("pause", onPause);
    v.addEventListener("seeking", onSeeking);
    return () => {
      v.removeEventListener("play", onPlay);
      v.removeEventListener("pause", onPause);
      v.removeEventListener("seeking", onSeeking);
    };
  }, [video, ttsSupported]);

  // Narrate the active per-frame description (replaces validator narration).
  useEffect(() => {
    if (!ttsSupported || !readAloud || !isPlaying || activeFrameIdx == null) return;
    if (lastSpokenFrame.current === activeFrameIdx) return;
    const f = analysis?.per_frame?.[activeFrameIdx];
    if (!f) return;
    const cls = extractInstrument(f.parsed);
    const ev = extractEvidence(f.parsed);
    const anat = extractAnatomy(f.parsed);
    const text = [cls, anat && `in the ${anat}`, ev && `— ${ev}`].filter(Boolean).join(" ");
    if (!text) return;
    window.speechSynthesis.cancel();
    const utt = new SpeechSynthesisUtterance(text);
    if (chosenVoice) utt.voice = chosenVoice;
    lastSpokenFrame.current = activeFrameIdx;
    window.speechSynthesis.speak(utt);
  }, [activeFrameIdx, readAloud, isPlaying, analysis, ttsSupported, chosenVoice]);

  useEffect(() => {
    if (!ttsSupported) return;
    if (!readAloud || !isPlaying) {
      window.speechSynthesis.cancel();
      lastSpokenFrame.current = null;
    }
  }, [readAloud, isPlaying, ttsSupported]);
  useEffect(() => () => { if (ttsSupported) window.speechSynthesis.cancel(); }, [ttsSupported]);

  const seekTo = (s: number) => {
    const v = videoRef.current;
    if (v) {
      v.currentTime = s;
      v.play().catch(() => {});
    }
  };

  const triggerAnalyze = (force = false) => {
    if (!video || analyzing) return;
    setAnalyzing(true);
    setAnalyzeError(null);
    cancelRef.current = api.studioAnalyze(
      { video_name: video, force, snapshot_id: snapshotId ?? undefined },
      {
        onResponse: (a) => { setAnalysis(a); setAnalyzing(false); cancelRef.current = null; },
        onError: (e) => { setAnalyzeError(e.message); setAnalyzing(false); cancelRef.current = null; },
      },
    );
  };

  const triggerAudio = (force = false) => {
    if (!video || audioBusy) return;
    setAudioBusy(true);
    setAudioError(null);
    api.studioAudioTranscribe(video, force)
      .then(setAudio)
      .catch((e) => setAudioError(e.message))
      .finally(() => setAudioBusy(false));
  };

  // Derived
  const perFrame = analysis?.per_frame ?? [];
  const summary = analysis?.validator?.parsed?.summary;
  const sections = analysis?.validator?.parsed?.sections ?? [];

  // Active per-frame for the "now" chip
  const active = activeFrameIdx != null ? perFrame[activeFrameIdx] : null;
  const activeLabel = active ? extractInstrument(active.parsed) : null;
  const activeEvidence = active ? extractEvidence(active.parsed) : "";
  const activeAnatomy = active ? extractAnatomy(active.parsed) : "";

  const allInstruments = useMemo(() => {
    const c = new Map<string, number>();
    for (const f of perFrame) {
      const k = extractInstrument(f.parsed);
      if (k && k !== "?") c.set(k, (c.get(k) ?? 0) + 1);
    }
    return Array.from(c.entries()).sort((a, b) => b[1] - a[1]);
  }, [perFrame]);

  return (
    <div className="space-y-4">
      {/* ── Top control bar ─────────────────────────────────────────── */}
      <Card>
        <CardContent className="flex flex-wrap items-center gap-3 p-3">
          <Select value={video ?? ""} onValueChange={setVideo}>
            <SelectTrigger className="h-9 w-64"><SelectValue placeholder="Pick a video…" /></SelectTrigger>
            <SelectContent>
              {(videos ?? []).map((v) => (
                <SelectItem key={v.name} value={v.name}>{v.name}</SelectItem>
              ))}
            </SelectContent>
          </Select>

          <Select
            value={snapshotId ?? "__default__"}
            onValueChange={(v) => setSnapshotId(v === "__default__" ? null : v)}
          >
            <SelectTrigger className="h-9 w-64" title="Override per-frame prompt + model using a saved Playground snapshot">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="__default__">Default model + prompt</SelectItem>
              {snapshots.map((s) => (
                <SelectItem key={s.id} value={s.id}>
                  {s.name}
                  {s.best_model && ` · ${s.best_model.replace("databricks-", "")}`}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>

          <Button onClick={() => triggerAnalyze(false)} disabled={!video || analyzing} className="gap-2">
            {analyzing ? <><Loader2 className="size-4 animate-spin" /> Analyzing…</>
              : analysis ? <><Wand2 className="size-4" /> Re-analyze</>
              : <><Wand2 className="size-4" /> Analyze video</>}
          </Button>
          {analysis && !analyzing && (
            <Button variant="outline" size="sm" onClick={() => triggerAnalyze(true)} title="Bypass cache">
              Force refresh
            </Button>
          )}

          {ttsSupported && analysis && (
            <Button
              variant={readAloud ? "default" : "outline"}
              size="sm"
              onClick={() => setReadAloud((r) => !r)}
              className="gap-2"
              title={chosenVoice ? `Read frame descriptions aloud (${chosenVoice.name})` : "Read aloud"}
            >
              {readAloud ? <Volume2 className="size-4" /> : <VolumeX className="size-4" />}
              {readAloud ? "Narrating" : "Narrate"}
            </Button>
          )}
          {/* Browsers without speechSynthesis (or without the Google US
              voices — Safari/Firefox) used to silently hide the Narrate
              button, which read as a bug. Show a disabled hint instead. */}
          {!ttsSupported && analysis && (
            <Button
              variant="outline"
              size="sm"
              disabled
              className="gap-2 opacity-60"
              title="Narration uses the browser's speech-synthesis voices, which aren't available in this browser. Chrome (desktop) has the best voices."
            >
              <VolumeX className="size-4" /> Narrate (unavailable)
            </Button>
          )}

          <div className="ml-auto flex items-center gap-2 text-xs text-muted-foreground">
            {analysis && (
              <>
                <Badge variant="secondary" className="font-normal">{analysis.n_frames} frames</Badge>
                <Badge variant="secondary" className="font-normal">{fmtTime(analysis.duration_s)}</Badge>
                <span className="font-mono">{analysis.per_frame_model.replace("databricks-", "")}</span>
                {analysis.from_cache && <Badge variant="outline" className="font-normal">cached</Badge>}
              </>
            )}
          </div>
        </CardContent>
      </Card>

      {analyzeError && (
        <Card>
          <CardContent className="flex items-center gap-2 p-3 text-sm text-destructive">
            <AlertCircle className="size-4" /> {analyzeError}
          </CardContent>
        </Card>
      )}

      {/* ── Main two-column: video | timeline ──────────────────────── */}
      <div className="grid gap-4 lg:grid-cols-[minmax(0,9fr)_minmax(0,7fr)]">
        {/* Left: video + "now" chip + audio (collapsible) */}
        <div className="space-y-3 lg:sticky lg:top-20 lg:self-start">
          <div className="overflow-hidden rounded-lg border bg-black">
            {video ? (
              <video
                ref={videoRef}
                key={video}
                controls
                preload="metadata"
                className="aspect-video w-full"
                src={api.videoStreamUrl(video)}
              />
            ) : (
              <div className="flex aspect-video items-center justify-center text-sm text-muted-foreground">
                pick a video
              </div>
            )}
          </div>

          {/* "Now" chip — distilled state of the active frame */}
          {active && (
            <Card>
              <CardContent className="p-3">
                <div className="flex flex-wrap items-center gap-2">
                  <span className="font-mono text-xs text-muted-foreground tabular-nums">
                    {fmtTime(currentTime)}
                  </span>
                  <Badge variant="default">{activeLabel}</Badge>
                  {activeAnatomy && (
                    <span className="text-xs text-muted-foreground">in {activeAnatomy}</span>
                  )}
                </div>
                {activeEvidence && (
                  <p className="mt-1.5 text-xs leading-relaxed text-muted-foreground">"{activeEvidence}"</p>
                )}
              </CardContent>
            </Card>
          )}

          {/* Audio (collapsible) */}
          <CollapsibleHeader
            open={audioOpen}
            onToggle={() => {
              setAudioOpen((o) => !o);
              if (!audioOpen && !audio) triggerAudio(false);
            }}
            label="Audio transcript"
            sub={audio?.segments?.length ? `${audio.segments.length} segments` : audio?.status === "no_audio_stream" ? "no audio" : "off"}
          />
          {audioOpen && (
            <Card>
              <CardContent className="p-2">
                {audioBusy && (
                  <p className="flex items-center gap-2 p-2 text-xs text-muted-foreground">
                    <Loader2 className="size-3.5 animate-spin" /> transcribing…
                  </p>
                )}
                {audioError && (
                  <p className="p-2 text-xs text-destructive">{audioError}</p>
                )}
                {audio?.status === "no_audio_stream" && (
                  <p className="p-2 text-xs text-muted-foreground">
                    This video has no audio track.
                  </p>
                )}
                {audio?.segments && audio.segments.length > 0 && (
                  <ScrollArea className="h-48 rounded border">
                    <ol className="space-y-0.5 p-1.5">
                      {audio.segments.map((s, i) => (
                        <li key={i}>
                          <button
                            ref={(el) => { audioRowRefs.current[i] = el; }}
                            onClick={() => seekTo(s.start)}
                            className={cn(
                              "flex w-full items-start gap-2 rounded p-1 text-left text-xs transition-colors",
                              i === activeAudioIdx ? "bg-primary/15 ring-1 ring-primary/40" : "hover:bg-accent",
                            )}
                          >
                            <span className="mt-0.5 shrink-0 font-mono tabular-nums text-muted-foreground">
                              {fmtTime(s.start)}
                            </span>
                            <span className="flex-1">{s.text || <span className="italic">[silence]</span>}</span>
                          </button>
                        </li>
                      ))}
                    </ol>
                  </ScrollArea>
                )}
              </CardContent>
            </Card>
          )}
        </div>

        {/* Right: per-frame timeline */}
        <div className="space-y-3">
          {!analysis && !analyzing && (
            <Card>
              <CardContent className="space-y-2 p-6 text-center">
                <p className="text-sm">No analysis yet for this video.</p>
                <p className="text-xs text-muted-foreground">
                  Click <span className="font-medium">Analyze video</span> to extract frames and describe each one.
                </p>
              </CardContent>
            </Card>
          )}
          {analyzing && !analysis && (
            <Card>
              <CardContent className="flex items-center gap-2 p-4 text-sm text-muted-foreground">
                <Loader2 className="size-4 animate-spin" /> running per-frame analysis…
              </CardContent>
            </Card>
          )}

          {analysis && (
            <>
              {/* Instrument-frequency summary (replaces validator's sectioning) */}
              {allInstruments.length > 0 && (
                <Card>
                  <CardContent className="flex flex-wrap items-center gap-1.5 p-3">
                    <span className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
                      Seen
                    </span>
                    {allInstruments.map(([k, n]) => (
                      <Badge key={k} variant="outline" className="font-normal">
                        {k} <span className="ml-1 text-muted-foreground">×{n}</span>
                      </Badge>
                    ))}
                  </CardContent>
                </Card>
              )}

              {/* MLflow vs-gold scorecard (only when labels overlap with this video's frames) */}
              {analysis.mlflow && analysis.mlflow.n_gold_overlap > 0 && (
                <Card>
                  <CardContent className="p-3">
                    <div className="flex flex-wrap items-center gap-3">
                      <span className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
                        vs gold
                      </span>
                      <span className="text-2xl font-semibold tabular-nums">
                        {(analysis.mlflow.accuracy_vs_gold * 100).toFixed(0)}%
                      </span>
                      <span className="text-xs text-muted-foreground">
                        on {analysis.mlflow.n_gold_overlap} labeled frame{analysis.mlflow.n_gold_overlap === 1 ? "" : "s"}
                      </span>
                      {analysis.mlflow.mlflow_url && (
                        <a
                          href={analysis.mlflow.mlflow_url}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="ml-auto text-xs text-primary hover:underline"
                        >
                          Open MLflow run →
                        </a>
                      )}
                    </div>
                    {Object.keys(analysis.mlflow.per_class).length > 0 && (
                      <div className="mt-2 flex flex-wrap gap-1.5">
                        {Object.entries(analysis.mlflow.per_class).map(([cls, m]) => (
                          <Badge
                            key={cls}
                            variant="outline"
                            className="font-normal"
                            title={`tp=${m.tp} fp=${m.fp} fn=${m.fn}`}
                          >
                            {cls}: P{(m.precision * 100).toFixed(0)} R{(m.recall * 100).toFixed(0)}
                          </Badge>
                        ))}
                      </div>
                    )}
                  </CardContent>
                </Card>
              )}

              {/* Per-frame timeline */}
              <Card>
                <CardContent className="p-0">
                  <ScrollArea className="h-[34rem]">
                    <ol className="divide-y">
                      {perFrame.map((f, i) => {
                        const cls = extractInstrument(f.parsed);
                        const ev = extractEvidence(f.parsed);
                        const ts = f.timestamp_s ?? 0;
                        const isActive = i === activeFrameIdx;
                        return (
                          <li key={f.frame}>
                            <button
                              ref={(el) => { frameRowRefs.current[i] = el; }}
                              onClick={() => seekTo(ts)}
                              className={cn(
                                "flex w-full items-start gap-3 px-3 py-2 text-left transition-colors",
                                isActive ? "bg-primary/10 ring-1 ring-primary/40" : "hover:bg-accent",
                              )}
                            >
                              {/* Thumb */}
                              {f.path && (
                                <img
                                  src={api.frameImageUrl(f.path)}
                                  className="size-16 shrink-0 rounded border bg-muted object-cover"
                                  loading="lazy"
                                  alt=""
                                />
                              )}
                              <div className="min-w-0 flex-1">
                                <div className="flex items-center gap-2">
                                  <span className="font-mono text-xs tabular-nums text-muted-foreground">
                                    {fmtTime(ts)}
                                  </span>
                                  <Badge variant={isActive ? "default" : "secondary"} className="font-normal">
                                    {cls}
                                  </Badge>
                                  {f.elapsed_s != null && (
                                    <span className="text-[10px] text-muted-foreground">
                                      {f.elapsed_s.toFixed(1)}s
                                    </span>
                                  )}
                                </div>
                                {ev && (
                                  <p className={cn(
                                    "mt-0.5 line-clamp-2 text-xs",
                                    isActive ? "text-foreground" : "text-muted-foreground"
                                  )}>
                                    {ev}
                                  </p>
                                )}
                              </div>
                              {isActive && (
                                <Play className="mt-1 size-3 shrink-0 text-primary" />
                              )}
                            </button>
                          </li>
                        );
                      })}
                    </ol>
                  </ScrollArea>
                </CardContent>
              </Card>
            </>
          )}
        </div>
      </div>

      {/* ── Optional procedure summary (collapsed by default) ───────── */}
      {analysis && (summary || sections.length > 0) && (
        <>
          <CollapsibleHeader
            open={summaryOpen}
            onToggle={() => setSummaryOpen((o) => !o)}
            label="Procedure summary"
            sub={sections.length > 0 ? `${sections.length} sections` : "summary only"}
            icon={<Sparkles className="size-4" />}
          />
          {summaryOpen && (
            <Card>
              <CardContent className="space-y-3 p-4">
                {summary && <p className="text-sm leading-relaxed">{summary}</p>}
                {sections.length > 0 && (
                  <ol className="space-y-1.5">
                    {sections.map((s, i) => (
                      <li key={i} className="rounded border p-2">
                        <button onClick={() => seekTo(s.start_s)} className="flex w-full items-center gap-2 text-left">
                          <Badge variant="secondary" className="font-normal">{s.primary_instrument}</Badge>
                          <span className="text-sm font-medium">{s.title}</span>
                          <span className="ml-auto font-mono text-xs text-muted-foreground tabular-nums">
                            {fmtTime(s.start_s)} – {fmtTime(s.end_s)}
                          </span>
                        </button>
                        <p className="mt-1 text-xs text-muted-foreground">{s.narrative}</p>
                      </li>
                    ))}
                  </ol>
                )}
              </CardContent>
            </Card>
          )}
        </>
      )}
    </div>
  );
}

function CollapsibleHeader({
  open, onToggle, label, sub, icon,
}: {
  open: boolean;
  onToggle: () => void;
  label: string;
  sub?: string;
  icon?: React.ReactNode;
}) {
  return (
    <button
      onClick={onToggle}
      className="flex w-full items-center gap-2 rounded-md border bg-card/60 px-3 py-2 text-sm font-medium hover:bg-accent"
    >
      {open ? <ChevronDown className="size-4" /> : <ChevronRight className="size-4" />}
      {icon}
      <span>{label}</span>
      {sub && <span className="ml-2 text-xs font-normal text-muted-foreground">{sub}</span>}
    </button>
  );
}
