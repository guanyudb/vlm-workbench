// Typed API client for the Surgical VLM Workbench backend.

async function asJson<T>(resp: Response): Promise<T> {
  if (!resp.ok) {
    let detail = resp.statusText;
    try {
      const body = await resp.json();
      if (body?.detail) detail = body.detail;
    } catch {
      /* ignore */
    }
    throw new Error(`${resp.status}: ${detail}`);
  }
  return resp.json();
}

export interface VideoEntry {
  name: string;
  path: string;
  size_bytes: number;
}

export interface FrameEntry {
  name: string;
  path: string;
  timestamp_s: number | null;
  video: string | null;
}

export interface ModelEntry {
  name: string;
  task: string;
  ready: boolean;
  vision: boolean | null;
}

export interface LocalModelEntry {
  name: string;
  display_name: string;
  hf_repo: string;
  accelerator: string;
  snapshot_dir: string | null;
  ready: boolean;
  notes: string | null;
}

export interface LocalRunSubmitResponse {
  run_id: string;
  databricks_run_id: string;
  yaml_path: string;
  output_dir: string;
  model_name: string;
  n_frames: number;
}

export interface IngestVideoRow {
  id: string | null;
  name: string;
  path: string;
  kind: "video" | "image_batch";
  size_bytes: number | null;
  duration_s: number | null;
  status: string;  // pending | queued | processing | ready | error
  status_message: string | null;
  n_frames_extracted: number | null;
  created_at: string | null;
  updated_at: string | null;
}

export interface DeployableModel {
  full_name: string;
  catalog: string;
  schema: string;
  name: string;
  versions: string[];
  base_model: string | null;
  n_train: string | null;
  train_loss: number | null;
  updated_at: string | null;
}

export interface ServingEndpointRow {
  name: string;
  state: string;
  config_state: string | null;
  model: string | null;
  version: string | null;
  workload_size: string | null;
  workload_type: string | null;
  creator: string | null;
  creation_timestamp: number | null;
  last_updated_timestamp: number | null;
  invocation_url: string | null;
  managed: boolean;
}

export interface FinetuneRunSummary {
  mlflow_run_id: string;
  run_name: string | null;
  status: string | null;
  start_time: number | null;
  end_time: number | null;
  base_model: string | null;
  n_train: string | null;
  lora_r: string | null;
  learning_rate: string | null;
  final_train_loss: number | null;
  train_elapsed_s: number | null;
}

export interface FrameLabel {
  frame_path: string;
  instrument: string;
  anatomy: string | null;
  tissue_condition: string | null;
  notes: string | null;
  source: string;
  labeled_by: string | null;
  created_at: string;
  updated_at: string;
}

export interface FrameLabelIn {
  frame_path: string;
  instrument: string;
  anatomy?: string | null;
  tissue_condition?: string | null;
  notes?: string | null;
  source?: string;
}

export interface LocalRunStatus {
  run_id: string;
  databricks_run_id: string;
  life_cycle_state: string | null;
  result_state: string | null;
  state_message: string | null;
  run_page_url: string | null;
  results: RunResultRow[] | null;
  model_name: string | null;
  successful: number | null;
  n_frames: number | null;
}

export interface RunResultRow {
  model: string;
  frame: string;
  ok: boolean;
  elapsed_s: number;
  raw?: string;
  parsed?: Record<string, unknown> | null;
  error?: string;
}

export interface PlaygroundRunRequest {
  frame_paths: string[];
  model_names: string[];
  prompt: string;
  max_tokens?: number;
  temperature?: number;
}

export interface PlaygroundRunResponse {
  total: number;
  successful: number;
  elapsed_s: number;
  models: string[];
  frames: string[];
  results: RunResultRow[];
}

export interface SnapshotSummary {
  id: string;
  name: string;
  n_frames: number;
  n_models: number;
  best_model: string | null;
  created_at: string;
  created_by: string | null;
}

export interface SnapshotDetail {
  id: string;
  name: string;
  notes?: string | null;
  frame_paths: string[];
  model_names: string[];
  prompt: string;
  best_model: string | null;
  results: RunResultRow[];
  elapsed_s: number | null;
  created_at: string;
  created_by: string | null;
}

export interface SaveSnapshotRequest {
  name: string;
  notes?: string;
  frame_paths: string[];
  model_names: string[];
  prompt: string;
  best_model?: string | null;
  results: RunResultRow[];
  elapsed_s?: number;
}

export interface StudioPerFrameRow {
  frame: string;
  timestamp_s: number | null;
  path: string | null;
  ok: boolean;
  raw?: string;
  parsed?: { instrument?: string; confidence?: number; anatomy?: string; tissue_condition?: string; evidence?: string };
  error?: string;
  elapsed_s?: number;
}

export interface StudioSection {
  title: string;
  start_s: number;
  end_s: number;
  primary_instrument: string;
  narrative: string;
  frame_indexes?: number[];
}

export interface StudioAnomaly {
  frame: string;
  issue: string;
  suggested_class?: string | null;
}

export interface StudioMlflowSummary {
  n_gold_overlap: number;
  accuracy_vs_gold: number;
  per_class: Record<string, {
    tp: number;
    fp: number;
    fn: number;
    precision: number;
    recall: number;
  }>;
  mlflow_url: string | null;
}

export interface StudioAnalysis {
  video_name: string;
  duration_s: number;
  n_frames: number;
  per_frame_model: string;
  validator_model: string;
  per_frame: StudioPerFrameRow[];
  validator: {
    ok: boolean;
    raw?: string;
    parsed?: { summary?: string; sections?: StudioSection[]; anomalies?: StudioAnomaly[] };
    error?: string;
    elapsed_s?: number;
  };
  total_elapsed_s: number;
  cached_at: number;
  from_cache?: boolean;
  cache_path?: string;
  mlflow?: StudioMlflowSummary | null;
}

export interface StudioAudioSegment {
  start: number;
  end: number;
  text: string;
}

export interface StudioAudioResponse {
  video_name: string;
  endpoint: string;
  shape?: string;
  status?: "ok" | "no_audio_stream" | string;
  message?: string;
  text?: string | null;
  segments: StudioAudioSegment[];
  raw_response?: unknown;
  from_cache?: boolean;
  cached_at?: number;
}

export const api = {
  health: () => fetch("/api/health").then(asJson<{ status: string }>),

  videos: () => fetch("/api/videos").then(asJson<VideoEntry[]>),

  frames: (opts: { source?: "extracted" | "eval" | "labeled"; video?: string } = {}) => {
    const params = new URLSearchParams();
    if (opts.source) params.set("source", opts.source);
    if (opts.video) params.set("video", opts.video);
    return fetch(`/api/frames?${params.toString()}`).then(asJson<FrameEntry[]>);
  },

  frameImageUrl: (path: string) =>
    `/api/frames/image?path=${encodeURIComponent(path)}`,

  models: (refresh = false) =>
    fetch(`/api/models${refresh ? "?refresh=true" : ""}`).then(asJson<ModelEntry[]>),

  localModels: () => fetch("/api/local-models").then(asJson<LocalModelEntry[]>),

  localRunSubmit: (req: { model_name: string; frame_paths: string[]; prompt: string; max_new_tokens?: number }) =>
    fetch("/api/playground/run/local", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(req),
    }).then(asJson<LocalRunSubmitResponse>),

  localRunStatus: (run_id: string, databricks_run_id: string) =>
    fetch(
      `/api/playground/run/local/${encodeURIComponent(run_id)}?databricks_run_id=${encodeURIComponent(databricks_run_id)}`
    ).then(asJson<LocalRunStatus>),

  // ── Labels (ground truth) ────────────────────────────────────────────
  listLabels: (frame_paths?: string[]) => {
    const q = frame_paths && frame_paths.length
      ? `?frame_paths=${encodeURIComponent(frame_paths.join(","))}`
      : "";
    return fetch(`/api/labels${q}`).then(asJson<FrameLabel[]>);
  },

  upsertLabel: (body: FrameLabelIn) =>
    fetch("/api/labels", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }).then(asJson<FrameLabel>),

  upsertLabelsBatch: (rows: FrameLabelIn[]) =>
    fetch("/api/labels/batch", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ rows }),
    }).then(asJson<{ inserted: number; rows: FrameLabel[] }>),

  deleteLabel: (frame_path: string) =>
    fetch(`/api/labels?frame_path=${encodeURIComponent(frame_path)}`, { method: "DELETE" })
      .then(asJson<{ deleted: number }>),

  labelsStats: () =>
    fetch("/api/labels/stats").then(asJson<{ total: number; by_instrument: { instrument: string; n: number }[] }>),

  labelsSyncToDelta: () =>
    fetch("/api/labels/sync-to-delta", { method: "POST" })
      .then(asJson<{ rows_synced: number; delta_table: string; genai_dataset?: string }>),

  // ── Fine-tuning ─────────────────────────────────────────────────────
  finetuneSubmit: (body: {
    base_model_name: string;
    uc_model_name?: string;
    train_prompt?: string;
    label_filter_instruments?: string[];
    snapshot_id?: string;
    num_epochs?: number;
    learning_rate?: number;
    lora_r?: number;
    lora_alpha?: number;
    accelerator?: string;
  }) =>
    fetch("/api/training/finetune", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }).then(asJson<{
      run_id: string;
      databricks_run_id: string;
      yaml_path: string;
      output_dir: string;
      uc_model_name: string;
      n_train: number;
    }>),

  finetuneStatus: (run_id: string, databricks_run_id: string) =>
    fetch(`/api/training/finetune/${encodeURIComponent(run_id)}?databricks_run_id=${encodeURIComponent(databricks_run_id)}`)
      .then(asJson<{
        run_id: string;
        databricks_run_id: string;
        life_cycle_state: string | null;
        result_state: string | null;
        state_message: string | null;
        run_page_url: string | null;
        result: {
          run_id: string;
          uc_model: string;
          mlflow_run_id: string;
          mlflow_url: string;
          manifest_name: string;
          merged_dir: string;
          n_train: number;
          n_val: number;
          train_loss: number;
          train_elapsed_s: number;
        } | null;
      }>),

  finetuneRuns: (limit = 25) =>
    fetch(`/api/training/runs?limit=${limit}`).then(asJson<FinetuneRunSummary[]>),

  // ── Auto-ingest ─────────────────────────────────────────────────────
  ingestVideos: () =>
    fetch("/api/ingest/videos").then(asJson<{ videos: IngestVideoRow[] }>),

  ingestSubmit: (body: { video_name?: string; candidate_fps?: number; max_frames?: number; force?: boolean }) =>
    fetch("/api/videos/ingest", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }).then(asJson<{ submitted: { name: string; video_id: string; databricks_run_id: string }[]; skipped: { name: string; reason: string }[] }>),

  // ── Model deployment ──────────────────────────────────────────────
  deployableModels: () =>
    fetch("/api/deploy/models").then(asJson<DeployableModel[]>),

  servingEndpoints: (only_managed = false) =>
    fetch(`/api/deploy/endpoints${only_managed ? "?only_managed=true" : ""}`)
      .then(asJson<ServingEndpointRow[]>),

  deploySubmit: (body: {
    model_full_name: string;
    model_version?: string;
    endpoint_name?: string;
    workload_type?: string;
    workload_size?: string;
    scale_to_zero?: boolean;
  }) =>
    fetch("/api/deploy/endpoints", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }).then(asJson<{
      name: string;
      model_full_name: string;
      version: string;
      workload_type: string;
      workload_size: string;
    }>),

  deployStatus: (name: string) =>
    fetch(`/api/deploy/endpoints/${encodeURIComponent(name)}`).then(asJson<any>),

  deployDelete: (name: string) =>
    fetch(`/api/deploy/endpoints/${encodeURIComponent(name)}`, { method: "DELETE" })
      .then(asJson<{ deleted: string }>),

  ingestImages: (body: { batch_name?: string; force?: boolean }) =>
    fetch("/api/images/ingest", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }).then(asJson<{ registered: { name: string; video_id: string; n_images: number }[] }>),

  deleteIngestVideo: (video_id: string) =>
    fetch(`/api/videos/${encodeURIComponent(video_id)}`, { method: "DELETE" })
      .then(asJson<{ deleted: number }>),

  saveSnapshot: (req: SaveSnapshotRequest) =>
    fetch("/api/playground/snapshots", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(req),
    }).then(asJson<{ id: string; created_at: string }>),

  listSnapshots: (limit = 30) =>
    fetch(`/api/playground/snapshots?limit=${limit}`).then(asJson<SnapshotSummary[]>),

  getSnapshot: (id: string) =>
    fetch(`/api/playground/snapshots/${encodeURIComponent(id)}`).then(asJson<SnapshotDetail>),

  deleteSnapshot: (id: string) =>
    fetch(`/api/playground/snapshots/${encodeURIComponent(id)}`, { method: "DELETE" }).then(
      asJson<{ deleted: number }>
    ),

  optimize: (req: {
    snapshot_id: string;
    optimizer: "gepa" | "dspy";
    teacher_model: string;
    student_model: string;
    n_rounds?: number;
    dspy_optimizer_type?: string;
    seed_prompt_override?: string;
  }) =>
    fetch("/api/playground/optimize", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(req),
    }).then(asJson<{
      run_id: string;
      databricks_run_id: string;
      yaml_path: string;
      output_dir: string;
      n_eval_examples: number;
      optimizer: string;
      student_model: string;
      teacher_model: string;
    }>),

  optimizeStatus: (run_id: string, databricks_run_id: string) =>
    fetch(
      `/api/playground/optimize/${encodeURIComponent(run_id)}?databricks_run_id=${encodeURIComponent(databricks_run_id)}`
    ).then(asJson<{
      run_id: string;
      databricks_run_id: string;
      life_cycle_state: string | null;
      result_state: string | null;
      state_message: string | null;
      run_page_url: string | null;
      best_prompt: string | null;
      score: number | null;
      history: number[] | null;
      best_prompt_path?: string;
      run_path?: string;
    }>),

  videoStreamUrl: (videoName: string) =>
    `/api/videos/${encodeURIComponent(videoName)}/stream`,

  studioAnalysis: (videoName: string) =>
    fetch(`/api/studio/analysis/${encodeURIComponent(videoName)}`).then(async (r) => {
      if (r.status === 404) return null;
      if (!r.ok) throw new Error(`${r.status} ${r.statusText}`);
      return (await r.json()) as StudioAnalysis;
    }),

  studioAnalyze: (
    body: {
      video_name: string;
      per_frame_model?: string;
      validator_model?: string;
      prompt?: string;
      snapshot_id?: string;
      max_frames?: number;
      force?: boolean;
    },
    handlers: { onResponse?: (r: StudioAnalysis) => void; onError?: (e: Error) => void },
  ): (() => void) => {
    const ctrl = new AbortController();
    (async () => {
      try {
        const resp = await fetch("/api/studio/analyze", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
          signal: ctrl.signal,
        });
        if (!resp.ok) {
          let detail = resp.statusText;
          try { const d = await resp.json(); if (d?.detail) detail = d.detail; } catch { /* */ }
          handlers.onError?.(new Error(`${resp.status}: ${detail}`));
          return;
        }
        handlers.onResponse?.(await resp.json());
      } catch (e) {
        if ((e as any)?.name !== "AbortError") handlers.onError?.(e as Error);
      }
    })();
    return () => ctrl.abort();
  },

  studioAudioGet: (videoName: string) =>
    fetch(`/api/studio/audio/${encodeURIComponent(videoName)}`).then(async (r) => {
      if (r.status === 404) return null;
      if (!r.ok) throw new Error(`${r.status} ${r.statusText}`);
      return (await r.json()) as StudioAudioResponse;
    }),

  studioAudioTranscribe: (videoName: string, force = false) =>
    fetch("/api/studio/audio/transcribe", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ video_name: videoName, force }),
    }).then(async (r) => {
      if (!r.ok) {
        let detail = r.statusText;
        try { const d = await r.json(); if (d?.detail) detail = d.detail; } catch { /* */ }
        throw new Error(`${r.status}: ${detail}`);
      }
      return (await r.json()) as StudioAudioResponse;
    }),

  /**
   * Submit a playground run and resolve when all (model × frame) calls return.
   * Returns an unsubscribe function that aborts the in-flight request.
   *
   * (We previously streamed via SSE but the Databricks Apps ingress buffers
   * stream responses regardless of x-accel-buffering. With 4-16 parallel
   * calls finishing in <10s, batched is plenty fast and reliable.)
   */
  playgroundRun: (
    req: PlaygroundRunRequest,
    handlers: {
      onResponse?: (r: PlaygroundRunResponse) => void;
      onError?: (err: Error) => void;
    },
  ): (() => void) => {
    const ctrl = new AbortController();
    (async () => {
      try {
        const resp = await fetch("/api/playground/run", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(req),
          signal: ctrl.signal,
        });
        if (!resp.ok) {
          let detail = resp.statusText;
          try {
            const body = await resp.json();
            if (body?.detail) detail = body.detail;
          } catch { /* ignore */ }
          handlers.onError?.(new Error(`${resp.status}: ${detail}`));
          return;
        }
        const data: PlaygroundRunResponse = await resp.json();
        handlers.onResponse?.(data);
      } catch (err) {
        if ((err as any)?.name !== "AbortError") {
          handlers.onError?.(err as Error);
        }
      }
    })();
    return () => ctrl.abort();
  },
};
