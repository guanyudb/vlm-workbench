import { useEffect, useState } from "react";
import {
  AlertCircle, CheckCircle2, Clock, ImageIcon, Loader2, Play, RefreshCcw, Trash2, Upload, Video, XCircle,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { api, type IngestVideoRow } from "@/api";
import { cn } from "@/lib/utils";

const VIDEOS_INBOX_PATH = "/Volumes/hls_amer_catalog/guanyu_chen/medical_video/videos/inbox/";
const IMAGES_INBOX_PATH = "/Volumes/hls_amer_catalog/guanyu_chen/medical_video/images/";

function fmtBytes(b: number | null | undefined): string {
  if (!b) return "—";
  if (b < 1024) return `${b} B`;
  if (b < 1024 ** 2) return `${(b / 1024).toFixed(1)} KB`;
  if (b < 1024 ** 3) return `${(b / 1024 ** 2).toFixed(1)} MB`;
  return `${(b / 1024 ** 3).toFixed(2)} GB`;
}

function fmtDuration(s: number | null | undefined): string {
  if (s == null) return "—";
  const m = Math.floor(s / 60);
  const r = Math.floor(s % 60);
  return `${m}:${String(r).padStart(2, "0")}`;
}

function StatusBadge({ status }: { status: string }) {
  if (status === "ready") {
    return <Badge variant="default" className="gap-1 font-normal"><CheckCircle2 className="size-3" />ready</Badge>;
  }
  if (status === "processing" || status === "queued") {
    return <Badge variant="secondary" className="gap-1 font-normal"><Loader2 className="size-3 animate-spin" />{status}</Badge>;
  }
  if (status === "error") {
    return <Badge variant="destructive" className="gap-1 font-normal"><XCircle className="size-3" />error</Badge>;
  }
  return <Badge variant="outline" className="gap-1 font-normal"><Clock className="size-3" />{status}</Badge>;
}

export default function Videos() {
  const [videos, setVideos] = useState<IngestVideoRow[]>([]);
  const [loading, setLoading] = useState(false);
  const [submitting, setSubmitting] = useState<string | null>(null);
  const [candidateFps, setCandidateFps] = useState<number>(1.0);
  const [maxFrames, setMaxFrames] = useState<number>(40);
  const [error, setError] = useState<string | null>(null);

  const refresh = () => {
    setLoading(true);
    api.ingestVideos()
      .then((r) => setVideos(r.videos))
      .catch((e) => setError(String(e)))
      .finally(() => setLoading(false));
  };
  useEffect(refresh, []);

  // Auto-refresh every 10s if anything is in-flight, so the UI follows along
  // with the running ingest jobs without a manual refresh.
  useEffect(() => {
    if (!videos.some((v) => v.status === "queued" || v.status === "processing")) return;
    const t = setInterval(refresh, 10000);
    return () => clearInterval(t);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [videos.map((v) => v.status).join(",")]);

  const submitOne = async (row: IngestVideoRow, force = false) => {
    setSubmitting(row.name);
    setError(null);
    try {
      if (row.kind === "image_batch") {
        // Image batches register synchronously — no GPU job needed.
        await api.ingestImages({ batch_name: row.name, force });
      } else {
        const r = await api.ingestSubmit({
          video_name: row.name,
          candidate_fps: candidateFps,
          max_frames: maxFrames,
          force,
        });
        if (r.skipped.length > 0 && r.submitted.length === 0) {
          setError(`Skipped: ${r.skipped[0].reason}`);
        }
      }
      refresh();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setSubmitting(null);
    }
  };

  const submitAllPending = async () => {
    setSubmitting("__all__");
    setError(null);
    try {
      const pending = videos.filter((v) => v.status === "pending");
      const pendingVideos = pending.filter((v) => v.kind === "video");
      const pendingImageBatches = pending.filter((v) => v.kind === "image_batch");
      let submittedAny = false;
      if (pendingVideos.length > 0) {
        const r = await api.ingestSubmit({
          candidate_fps: candidateFps,
          max_frames: maxFrames,
        });
        if (r.submitted.length > 0) submittedAny = true;
        if (r.submitted.length === 0 && r.skipped.length > 0) {
          setError(r.skipped.map((s) => s.reason).join("; "));
        }
      }
      if (pendingImageBatches.length > 0) {
        const r = await api.ingestImages({});
        if (r.registered.length > 0) submittedAny = true;
      }
      if (!submittedAny && pending.length === 0) {
        setError("Nothing pending.");
      }
      refresh();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setSubmitting(null);
    }
  };

  const deleteVideo = async (video_id: string, name: string) => {
    if (!confirm(`Remove '${name}' and its extracted-frames index? (JPGs in the volume are kept.)`)) return;
    try {
      await api.deleteIngestVideo(video_id);
      refresh();
    } catch (e) {
      setError((e as Error).message);
    }
  };

  const pendingCount = videos.filter((v) => v.status === "pending").length;
  const inFlightCount = videos.filter((v) => v.status === "queued" || v.status === "processing").length;
  const readyCount = videos.filter((v) => v.status === "ready").length;
  const errorCount = videos.filter((v) => v.status === "error").length;

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-semibold">Library</h1>
        <p className="text-sm text-muted-foreground">
          Two ways to add data: drop <span className="font-medium">MP4s</span> into{" "}
          <span className="font-mono text-xs">{VIDEOS_INBOX_PATH}</span> and they'll get smart-frame
          extracted, OR drop <span className="font-medium">JPGs/PNGs</span> into a subfolder of{" "}
          <span className="font-mono text-xs">{IMAGES_INBOX_PATH}</span> and they'll register as a
          named batch. Both flow into Playground's "Smart-extracted" frame source.
        </p>
      </div>

      {/* Stats + actions */}
      <Card>
        <CardHeader className="pb-3">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <div>
              <CardTitle className="text-base">Inbox</CardTitle>
              <CardDescription>
                {videos.length} videos · {pendingCount} pending · {inFlightCount} in flight · {readyCount} ready
                {errorCount > 0 && <> · <span className="text-destructive">{errorCount} errored</span></>}
              </CardDescription>
            </div>
            <div className="flex items-center gap-2">
              <Button variant="outline" size="sm" onClick={refresh} disabled={loading}>
                {loading ? <Loader2 className="size-3.5 animate-spin" /> : <RefreshCcw className="size-3.5" />}
                Refresh
              </Button>
              <Button
                onClick={submitAllPending}
                disabled={pendingCount === 0 || submitting === "__all__"}
                className="gap-2"
              >
                {submitting === "__all__" ? <Loader2 className="size-4 animate-spin" /> : <Upload className="size-4" />}
                Ingest all pending ({pendingCount})
              </Button>
            </div>
          </div>
        </CardHeader>
        <CardContent>
          <div className="flex flex-wrap items-end gap-3">
            <div>
              <Label className="text-xs">Frames per second to score</Label>
              <Input
                type="number"
                min={0.1}
                max={5}
                step={0.1}
                value={candidateFps}
                onChange={(e) => setCandidateFps(Number(e.target.value) || 1)}
                className="w-28"
              />
            </div>
            <div>
              <Label className="text-xs">Max smart frames</Label>
              <Input
                type="number"
                min={5}
                max={200}
                step={5}
                value={maxFrames}
                onChange={(e) => setMaxFrames(Number(e.target.value) || 40)}
                className="w-28"
              />
            </div>
            <p className="text-[10px] text-muted-foreground">
              Default 1 fps × 40 frames is a good starting point for arthroscopy videos. Increase
              max_frames for longer videos.
            </p>
          </div>
          {error && (
            <p className="mt-3 flex items-center gap-1 text-xs text-destructive">
              <AlertCircle className="size-3.5" /> {error}
            </p>
          )}
        </CardContent>
      </Card>

      {/* Videos table */}
      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-base">All videos</CardTitle>
        </CardHeader>
        <CardContent className="p-0">
          {videos.length === 0 ? (
            <p className="p-4 text-xs text-muted-foreground">
              {loading ? "loading…" : "No videos found. Drop an MP4 into the inbox above."}
            </p>
          ) : (
            <div className="divide-y">
              {videos.map((v) => (
                <div key={v.name} className="flex flex-wrap items-center gap-3 px-4 py-3">
                  <div className="flex shrink-0 items-center gap-2">
                    {v.kind === "image_batch"
                      ? <ImageIcon className="size-4 text-muted-foreground" />
                      : <Video className="size-4 text-muted-foreground" />}
                  </div>
                  <div className="min-w-0 flex-1">
                    <div className="flex flex-wrap items-center gap-2">
                      <span className="truncate font-mono text-sm">{v.name}</span>
                      <Badge variant="outline" className="font-normal text-[10px]">{v.kind === "image_batch" ? "images" : "video"}</Badge>
                      <StatusBadge status={v.status} />
                      {v.duration_s != null && (
                        <span className="text-xs text-muted-foreground">{fmtDuration(v.duration_s)}</span>
                      )}
                      {v.n_frames_extracted != null && (
                        <span className="text-xs text-muted-foreground">{v.n_frames_extracted} {v.kind === "image_batch" ? "images" : "frames"}</span>
                      )}
                      <span className="text-xs text-muted-foreground">{fmtBytes(v.size_bytes)}</span>
                    </div>
                    {v.status_message && (
                      <p className={cn(
                        "mt-0.5 text-[11px]",
                        v.status === "error" ? "text-destructive" : "text-muted-foreground"
                      )}>
                        {v.status_message}
                      </p>
                    )}
                  </div>
                  <div className="flex items-center gap-1.5">
                    {(v.status === "pending" || v.status === "error") && (
                      <Button
                        size="sm"
                        variant="default"
                        disabled={submitting === v.name}
                        onClick={() => submitOne(v, v.status === "error")}
                        className="gap-1"
                      >
                        {submitting === v.name ? <Loader2 className="size-3.5 animate-spin" /> : <Play className="size-3.5" />}
                        {v.status === "error" ? "Retry" : "Ingest"}
                      </Button>
                    )}
                    {v.status === "ready" && (
                      <Button
                        size="sm"
                        variant="outline"
                        disabled={submitting === v.name}
                        onClick={() => submitOne(v, true)}
                        className="gap-1"
                        title={v.kind === "image_batch" ? "Re-register this image batch" : "Re-extract frames (overwrites existing index)"}
                      >
                        <RefreshCcw className="size-3.5" /> Re-ingest
                      </Button>
                    )}
                    {v.id && (
                      <Button
                        size="sm"
                        variant="ghost"
                        onClick={() => deleteVideo(v.id!, v.name)}
                        className="gap-1 text-muted-foreground"
                        title="Remove this video's index row (keeps the JPG files)"
                      >
                        <Trash2 className="size-3.5" />
                      </Button>
                    )}
                  </div>
                </div>
              ))}
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
