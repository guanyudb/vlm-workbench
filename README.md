# Surgical VLM Workbench

A Databricks App that takes a surgical video (or a folder of frames) from raw bytes
to a fine-tuned, deployed VLM, with a unified UI for every step in between.

The opinionated workflow:

```
Library  →  Playground  →  Label  →  Optimize  →  Train  →  Deploy  →  Studio
  ingest      explore     ground       prompt      LoRA      UC + Model    full-video
              models      truth                              Serving       review
              + prompt    in 28 keys                                       w/ scoring
```

Each step is its own tab. Each artifact (snapshot, label set, optimized prompt,
fine-tuned model, evaluation run) is tracked in Lakebase + MLflow + Unity
Catalog so the whole loop is reproducible and shareable.

## Stack

- **Backend**: FastAPI (`app.py`, ~7K LOC). Files API → Volume, OpenAI-compat
  → AI Gateway, Databricks Jobs API → serverless GPU, MLflow → UC.
- **Frontend**: React 18 + Vite + TypeScript + Tailwind v4 + shadcn/ui
  (apx-style design system).
- **State**: Lakebase (Postgres) for snapshots / labels / videos /
  extracted-frames index / studio analyses / audio transcripts. Delta tables
  for long-term backups. UC Volumes for binaries (videos, frames, model
  checkpoints).
- **Compute**: serverless GPU (`GPU_1xA10`, `GPU_8xH100`) with
  `databricks_ai_v4` base env for VLM inference + LoRA fine-tuning,
  AI Gateway for FMAPI VLMs.
- **Tracking**: MLflow GenAI primitives (`register_prompt`, `datasets`,
  `evaluate`) end-to-end.

## Tabs

| Tab | What it does |
|---|---|
| **Library** | Drop MP4s into `/Volumes/.../videos/inbox/` or JPGs into `/Volumes/.../images/<batch>/`. One click → smart-frame extraction job, frames registered in Lakebase, Playground sees them. |
| **Playground** | Pick frames + N VLMs (AI Gateway and/or local serverless GPU) + a prompt → side-by-side results with ✓/✗ vs gold and per-model accuracy. Save as a snapshot. |
| **Label** | Bootstrap labels from a snapshot's best model → verify each frame with hotkeys → save as ground truth → optionally Sync to Delta + UC GenAI dataset. |
| **Optimize** (dialog in Playground) | GEPA or DSPy prompt optimization against gold labels (or a teacher model). Optimized prompt is registered as a new MLflow `PromptVersion`. |
| **Train** | LoRA fine-tune Qwen3-VL / MedGemma on the labeled set. Auto-merges + registers in UC Model Registry + drops a manifest so the fine-tuned variant appears as a selectable local model in Playground. |
| **Deploy** | Spin up a Databricks Model Serving endpoint for any UC-registered model (base or fine-tuned). Endpoints appear in Playground's AI Gateway list for direct A/B comparison. |
| **Compare** | Side-by-side scoreboard for 2+ snapshots — shared frames, agreement %, per-cell predictions. |
| **Studio** | Watch a whole video with the per-frame VLM analysis aligned to the timeline + optional MLflow vs-gold scorecard. |

## Setup

### Requirements

- Databricks workspace with serverless GPU enabled and Model Serving access
- Unity Catalog with a writable catalog/schema (defaults: `hls_amer_catalog.guanyu_chen`)
- A Lakebase Postgres instance (defaults: `lakebasepoc`)
- A SQL warehouse the app's SP can `CAN_USE`
- Hugging Face token for gated repos (stored in a Databricks secret scope)

### Configure `app.yaml`

```yaml
resources:
  - name: sql-warehouse
    sql_warehouse:
      id: "<warehouse-id>"
      permission: CAN_USE
  - name: lakebase
    database:
      database_name: vlm_workbench
      instance_name: <lakebase-instance-name>
      permission: CAN_CONNECT_AND_CREATE
env:
  - name: DATABRICKS_HOST
    value: "https://<workspace>.cloud.databricks.com/"
  - name: MLFLOW_ENABLE_DB_SDK
    value: "1"
  - name: MLFLOW_TRACKING_URI
    value: "databricks"
  - name: MLFLOW_REGISTRY_URI
    value: "databricks-uc"
  - name: VOLUME_PATH
    value: "/Volumes/<catalog>/<schema>/<volume>"
```

### App SP permissions

The app's service principal needs:
- `CAN_USE` on the SQL warehouse
- `USE CATALOG`, `USE SCHEMA`, `MODIFY`, `SELECT`, `EXECUTE`, `CREATE TABLE`,
  `CREATE FUNCTION`, `CREATE MODEL`, `CREATE VOLUME` on the target catalog/schema
- `READ VOLUME`, `WRITE VOLUME` on the data Volume
- `READ` on the HF token secret scope
  (`databricks secrets put-acl <scope> <SP_ID> READ`)
- A Postgres role in the Lakebase instance keyed by the SP UUID

**Known limitation**: UC-managed MLflow Prompts have a permission check that
isn't grantable via SQL. If the app reports "Permission denied to update
prompt in schema", add the SP as a schema co-owner via the UC UI. The Delta
+ Lakebase paths are unaffected and the Optimize / Train / Studio MLflow run
logging still works.

### Build + deploy

```bash
./build.sh                                       # npm install + vite build → static/
WS_BASE=/Workspace/Users/<you>/vlm-workbench
databricks workspace import-dir . "$WS_BASE" --overwrite --profile <profile>
databricks apps deploy vlm-workbench --source-code-path "$WS_BASE" --profile <profile>
```

### Local dev

```bash
# Terminal 1 — backend (uses your CLI auth profile)
DATABRICKS_CONFIG_PROFILE=<profile> \
VOLUME_PATH=/Volumes/<catalog>/<schema>/<volume> \
DATABRICKS_WAREHOUSE_ID=<warehouse-id> \
uvicorn app:app --port 8000 --reload

# Terminal 2 — frontend (proxies /api → :8000)
cd frontend && npm install && npm run dev
```

Open http://localhost:5173.

## Repo layout

```
vlm-workbench/
├── app.py                          # FastAPI backend
├── app.yaml                        # Databricks App config
├── requirements.txt
├── build.sh                        # npm install + vite build
├── frontend/                       # React SPA
│   └── src/
│       ├── App.tsx                 # nav + routing
│       ├── Playground.tsx          # explore models + prompts
│       ├── Label.tsx               # ground-truth labeling
│       ├── Train.tsx               # LoRA fine-tune
│       ├── Deploy.tsx              # Model Serving endpoints
│       ├── Studio.tsx              # video + per-frame timeline
│       ├── Compare.tsx             # multi-snapshot scoreboard
│       ├── Videos.tsx              # Library tab
│       └── components/             # shadcn primitives + apx shell
├── local_inference_notebooks/      # bundled, parameterized notebooks
│   ├── run/                        # local VLM batch inference (Playground)
│   ├── finetune/                   # LoRA fine-tune (Train)
│   ├── repin/                      # re-register UC model with corrected reqs
│   └── setup_cache/                # snapshot HF weights to Volume
├── optimizer_notebooks/            # bundled prompt-optimizer notebooks
│   ├── gepa/
│   └── dspy/
├── ingest_notebooks/               # bundled smart-frame extractor (Library)
│   └── smart_frames/
└── experiments/                    # standalone verification jobs
    ├── surgical_vlm_eval/
    └── mlflow_genai_verify/
```

## Workflow walkthrough

Concrete example: arthroscopy procedure → instrument-ID model.

1. **Library** — drop `vid.mp4` into `/Volumes/.../videos/inbox/`. Click
   `Ingest`. Smart-frame extractor runs (~1 min for a 4-min video), 30
   best frames register in Lakebase.
2. **Playground** — pick all 30 frames. Select Sonnet 4.5, GPT-5-2, Qwen3-VL
   (local), MedGemma (local). Run. See ✓/✗ per cell (vs any pre-existing
   labels) + per-model accuracy.
3. Save snapshot. Send to **Label**.
4. **Label** — Bootstrap from snapshot (uses best_model's predictions to
   pre-fill). Verify each frame with `1`–`9` hotkeys + `Enter` to save.
5. **Sync to Delta** + UC GenAI dataset.
6. Back to **Playground** → Optimize Prompt. Teacher = `Gold labels`,
   student = Qwen3-VL. Run GEPA for 5 rounds. Apply the resulting prompt.
   The MLflow run has the trajectory, the optimized prompt is registered
   as a new version.
7. **Train** — base = `qwen3-vl-8b`, label scope = all 28 labels.
   Start fine-tune. ~10 min on 1×A10. New UC model auto-appears in
   Playground's Local section.
8. **Playground** — A/B base Qwen vs the fine-tune on the same 28 frames.
   Accuracy chips show the lift.
9. **Deploy** — pick the fine-tuned model → workload `GPU_LARGE` → Deploy.
   ~10 min cold-start. Endpoint appears in Playground's AI Gateway list.
10. **Studio** — analyze the full video with the deployed endpoint.
    MLflow vs-gold scorecard logs everything to the workbench experiment.

## License

Internal. Code is shared as a reference Databricks App for VLM
experimentation workflows; not a supported product.
