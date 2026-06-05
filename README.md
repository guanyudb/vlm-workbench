# Surgical VLM Workbench

A Databricks App that takes a surgical video (or a folder of frames) from raw
bytes to a fine-tuned, deployed VLM, with a unified UI for every step in
between.

The opinionated workflow:

```
Library  →  Playground  →  Label  →  Optimize  →  Train  →  Deploy  →  Studio
  ingest      explore     ground       prompt      LoRA      UC + Model    full-video
              models      truth                              Serving       review
              + prompt    in 28 keys                                       w/ scoring
```

Each step is its own tab. Each artifact (snapshot, label set, optimized
prompt, fine-tuned model, evaluation run) is tracked in Lakebase + MLflow +
Unity Catalog so the whole loop is reproducible and shareable.

## Stack

- **Backend** — FastAPI (`app.py`). Files API → Volume, OpenAI-compat → AI
  Gateway, Databricks Jobs API → serverless GPU, MLflow → UC.
- **Frontend** — React 18 + Vite + TypeScript + Tailwind v4 + shadcn/ui.
- **State** — Lakebase (Postgres) for snapshots / labels / videos /
  extracted-frames index / Studio analyses / audio transcripts. Delta tables
  for long-term backups. UC Volumes for binaries (videos, frames, model
  checkpoints).
- **Compute** — serverless GPU (`GPU_1xA10`, `GPU_8xH100`) with
  `databricks_ai_v4` base env for VLM inference + LoRA fine-tuning, AI
  Gateway for FMAPI VLMs.
- **Tracking** — MLflow GenAI primitives (`register_prompt`, `datasets`,
  `evaluate`) end-to-end.

## Tabs

| Tab | What it does |
| --- | --- |
| **Library** | Upload (drag-and-drop) or drop MP4s into `videos/inbox/` and JPGs into `images/<batch>/`. One click → smart-frame extraction job, frames registered in Lakebase, Playground sees them. |
| **Playground** | Pick frames + N VLMs (AI Gateway and/or local serverless GPU) + a prompt → side-by-side results with ✓/✗ vs gold and per-model accuracy. Save as a snapshot. |
| **Label** | Bootstrap labels from a snapshot's best model → verify each frame with hotkeys → save as ground truth → optionally Sync to Delta + UC GenAI dataset. |
| **Optimize** (dialog in Playground) | GEPA or DSPy prompt optimization against gold labels (or a teacher model). Optimized prompt is registered as a new MLflow `PromptVersion`. |
| **Train** | LoRA fine-tune Qwen3-VL / MedGemma on the labeled set. Auto-merges + registers in UC Model Registry + drops a manifest so the fine-tuned variant appears as a selectable local model in Playground. |
| **Deploy** | Spin up a Databricks Model Serving endpoint for any UC-registered model (base or fine-tuned). Endpoints appear in Playground's AI Gateway list for direct A/B comparison. |
| **Compare** | Side-by-side scoreboard for 2+ snapshots — shared frames, agreement %, per-cell predictions. |
| **Studio** | Watch a whole video with the per-frame VLM analysis aligned to the timeline + optional MLflow vs-gold scorecard. |
| **Setup** | Health checks for every dependency (UC, Volume, Lakebase, SQL warehouse, HF token, MLflow experiment, GenAI prompts). One-click HF token paste + model cache. Editable task vocabulary + response schema + prompt template. |

## Deployment

The workbench deploys via **Databricks Asset Bundles** (DAB). One command
creates everything: the App, UC schema/volume, secret scopes, jobs, and the
post-deploy job that grants permissions and seeds workspace metadata.

### Prerequisites

- A Databricks workspace with serverless GPU enabled and Model Serving access
- Unity Catalog with a writable catalog where you have `CREATE SCHEMA`
- A Lakebase Postgres **Project** (new Autoscaling model) — create one in
  the workspace UI under Compute → Lakebase. Note the project name + the
  database name (default `databricks_postgres`).
- A SQL warehouse the App SP can `CAN_USE`
- A Hugging Face token (for gated repos like MedGemma). The App's Setup tab
  can also accept the token directly — you don't need to put it in a
  workspace secret beforehand.
- `databricks` CLI authenticated to the target workspace
  (`databricks auth login`)

### Configure your workspace values

Copy the sample and fill in your workspace's values:

```bash
cp .databricks/bundle/dev/variable-overrides.json.sample \
   .databricks/bundle/dev/variable-overrides.json
```

Edit `variable-overrides.json`:

```jsonc
{
  "uc_catalog": "<your_catalog>",
  "uc_schema": "<your_schema>",
  "sql_warehouse_id": "<your_warehouse_id>",
  "lakebase_instance": "<your_lakebase_project_name>",
  "lakebase_branch": "production",
  "lakebase_database": "databricks_postgres",
  "lakebase_database_path": "databricks-postgres",
  "hf_secret_scope": "vlmwb_hf",
  "hf_secret_key": "HF_TOKEN"
}
```

Notes:
- `lakebase_database_path` is the DAB resource slug for the Postgres
  database (typically the database name with `_` → `-`). For the default
  `databricks_postgres` database this is `databricks-postgres`.
- `hf_secret_scope` is a per-deploy scope (default `vlmwb_hf`). The post-
  deploy step creates it if it doesn't exist; you paste your HF token from
  the Setup tab afterward.

### Deploy

```bash
./deploy.sh dev --profile <your-profile>
```

This wraps the canonical sequence and runs it as one command:

1. `./build.sh` — builds the React SPA into `static/`.
2. Pre-seed workbench secrets (idempotent).
3. `databricks bundle deploy -t dev` — creates the App, UC schema, volume,
   secret scope, jobs, and binds them all together.
4. `databricks bundle run -t dev vlmwb_postdeploy_setup` — grants the App
   SP the UC + Postgres + secret-scope permissions it needs.
5. `databricks bundle run -t dev vlm_workbench_app` — pushes the source to
   the App and starts it.

The CLI prints the App URL at the end. Open it in your browser.

### First-time setup in the App

1. Open the **Setup** tab. Every check should be green except possibly the
   HuggingFace token (placeholder seeded on first deploy) and the Local
   model cache (empty until you cache models).
2. On the HuggingFace row, click **Set token / cache** → paste your HF
   token → keep both model checkboxes ticked → **Save + cache 2**. The
   workbench submits a Databricks job that snapshots Qwen3-VL-8B and
   MedGemma-4b-it to your Volume (~5–30 min depending on network).
3. Open the **Library** tab → drop an MP4 (or use the drag-and-drop zone)
   → click **Ingest**. The smart-frame extractor runs and frames appear in
   Playground.

### Multiple workspaces

To deploy to a different workspace (e.g., for a customer demo), create a
new target in `databricks.yml` and a matching `variable-overrides.json`
under `.databricks/bundle/<target>/`. Targets are workspace-agnostic; all
workspace-specific values live in the overrides file.

## Local development

```bash
# Terminal 1 — backend (uses your CLI auth profile)
DATABRICKS_CONFIG_PROFILE=<profile> \
UC_CATALOG=<catalog> UC_SCHEMA=<schema> VOLUME_NAME=<volume> \
DATABRICKS_WAREHOUSE_ID=<warehouse-id> \
uvicorn app:app --port 8000 --reload

# Terminal 2 — frontend (proxies /api → :8000)
cd frontend && npm install && npm run dev
```

Open http://localhost:5173.

## Repo layout

```
vlm-workbench/
├── README.md
├── databricks.yml                  # DAB bundle root
├── variables.yml                   # Per-workspace knobs (defaults + descriptions)
├── deploy.sh                       # One-command wrapper (build → deploy → postdeploy → app)
├── build.sh                        # npm install + vite build → static/
├── app.py                          # FastAPI backend
├── app.yaml                        # Databricks Apps runtime config (env, command)
├── requirements.txt
├── resources/                      # DAB resource definitions
│   ├── app.yml                     # The Databricks App + its resource bindings
│   ├── schema.yml                  # UC schema
│   ├── volumes.yml                 # UC volume
│   ├── secrets.yml                 # Workbench secret scope
│   └── jobs/
│       └── postdeploy_setup.yml    # Post-deploy GRANT + bootstrap job
├── setup/
│   ├── postdeploy.py               # Notebook the postdeploy job runs
│   └── lakebase_probe.py           # Debug notebook (probes Lakebase from user identity)
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
│       ├── Setup.tsx               # Health checks + task config + HF dialog
│       └── components/             # shadcn primitives + apx shell + runs pill
├── local_inference_notebooks/      # Bundled, parameterized notebooks
│   ├── run/                        # Local VLM batch inference (Playground)
│   ├── finetune/                   # LoRA fine-tune (Train)
│   ├── repin/                      # Re-register UC model with corrected reqs
│   └── setup_cache/                # Snapshot HF weights to Volume
├── optimizer_notebooks/            # Bundled prompt-optimizer notebooks
│   ├── gepa/
│   └── dspy/
├── ingest_notebooks/               # Bundled smart-frame extractor (Library)
│   └── smart_frames/
└── experiments/                    # Standalone verification jobs
```

## End-to-end workflow

Concrete example: arthroscopy procedure → instrument-ID model.

1. **Library** — drop `vid.mp4` into the upload zone (or copy it into
   `/Volumes/.../videos/inbox/`). Click **Ingest**. Smart-frame extractor
   runs (~1 min for a 4-min video), the best frames register in Lakebase.
2. **Playground** — pick all frames. Select Sonnet 4.5, GPT-5, Qwen3-VL
   (local), MedGemma (local). Run. See ✓/✗ per cell (vs any pre-existing
   labels) + per-model accuracy.
3. Save snapshot. Send to **Label**.
4. **Label** — Bootstrap from snapshot (uses the best model's predictions
   to pre-fill). Verify each frame with `1`–`9` hotkeys + `Enter` to save.
5. **Sync to Delta** + UC GenAI dataset.
6. Back to **Playground** → **Optimize prompt**. Teacher = `Gold labels`,
   student = Qwen3-VL. Run GEPA for 5 rounds. Apply the resulting prompt.
   The MLflow run has the trajectory; the optimized prompt is registered
   as a new version.
7. **Train** — base = `qwen3-vl-8b`, label scope = all 28 labels. Start
   fine-tune. ~10 min on 1×A10. New UC model auto-appears in Playground's
   Local section.
8. **Playground** — A/B base Qwen vs the fine-tune on the same frames.
   Accuracy chips show the lift.
9. **Deploy** — pick the fine-tuned model → workload size → **Deploy**.
   ~10 min cold-start. Endpoint appears in Playground's AI Gateway list.
10. **Studio** — analyze the full video with the deployed endpoint.
    MLflow vs-gold scorecard logs everything to the workbench experiment.

## License

Code is shared as a reference Databricks App for VLM experimentation
workflows; not a supported product.
