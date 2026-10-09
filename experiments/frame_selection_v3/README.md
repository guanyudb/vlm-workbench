# Frame selection v3.1 — SigLIP2 zero-shot gate + MMR selector

Shipped in `ingest_notebooks/smart_frames/notebook.py` (selector_version `v3.1`).
Falls back loudly to v1.5 when the SigLIP2 files or onnxruntime are missing.

## What changed vs v1.5

1. **One CPU vision pass per 1-fps candidate** with SigLIP2-base (Apache-2.0),
   exported to ONNX (`export_onnx.py`). Text prompts are embedded once, offline,
   into `siglip2_text_concepts.npz`, so ingest only runs the vision encoder.
2. **Zero-shot privacy gate.** `oob_zs = L(out-of-body) − L(inside the body)`,
   where each L is the logsumexp of SigLIP logits over that concept's prompts.
   The prompt sets are in `extract.py`: operating room, faces, skin, screens,
   title slides, X-ray/CT/MRI, diagrams, gloved hands, video-player UI.
   Frames with `oob_zs > −1.0` are never persisted. This works alongside the
   existing scope-mask border check and colour heuristic.
3. **Utility** = v1.5 image-quality value + 0.35·rank(instrument visible)
   − 0.5·rank(obscured/blurry/glare/text overlay). Candidates in the bottom
   35% by image quality get −1, so semantics can't promote a glare-washed close-up.
4. **Selector:**
   - v1.5's coverage anchors (≥1 per Y-second window) and bookends, using utility
   - then an MMR fill: utility − 0.6·(max cosine to the frames already picked), under v1.5's relaxing time spacing
   - near-duplicates (cosine > 0.95) are skipped; that cutoff relaxes last, so static videos still reach K

## Results (measured 2026-10-09)

**Out-of-body gate.** Measured on 292 real in-body arthroscopy frames (private
data, scored locally only) and 35 operating-room/equipment images (Wikimedia
CC/PD, plus synthetic skin-in-scope frames):

| Detector | In-body false rejects | Out-of-body caught |
|---|---|---|
| Colour heuristic (v1.5) | 0.7% | 26% |
| SigLIP2 zero-shot, thr −1.0 | 1.7% | 100% |

On the public videos, every frame with `oob_zs` between −1 and 3.5 was
genuinely non-surgical: title cards (one with patient age/sex), X-ray and CT
slides, open-wound skin, a gloved hand holding the implant. That's why the
threshold is −1 rather than a stricter +2.5.

**Selector.** Six public CC-BY videos: four knee/shoulder arthroscopy clips
from BMC/Hindawi supplements on Wikimedia Commons, and two laparoscopy clips.
Same quality gate for both pipelines.

| Video | K | Pipeline | dup pairs >0.95 | max gap (s) | semantic clusters | instrument pct | obscured pct | non-surgical picks |
|---|---|---|---|---|---|---|---|---|
| knee_cartilage | 12 | v1.5 | 4 | 13 | 7/12 | 0.38 | 0.31 | 2 |
| knee_cartilage | 12 | v3.1 | 0 | 17 | 9/12 | 0.51 | 0.42 | 0 |
| shoulder_chl | 12 | v1.5 | 0 | 41 | 9/12 | 0.44 | 0.37 | 1 |
| shoulder_chl | 12 | v3.1 | 0 | 41 | 9/12 | 0.68 | 0.34 | 0 |
| knee_tibial_insert | 12 | v1.5 | 1 | 12 | 2/4 | 0.32 | 0.19 | 9 |
| knee_tibial_insert | 12 | v3.1 | 1 | 21 | 4/4 | 0.56 | 0.56 | 0 |
| knee_femoral | 8 | v1.5 | 9 | 7 | 5/8 | 0.59 | 0.54 | 0 |
| knee_femoral | 8 | v3.1 | 7 | 8 | 5/8 | 0.61 | 0.52 | 0 |
| lap_biliary | 12 | v1.5 | 7 | 29 | 7/12 | 0.45 | 0.36 | 3 |
| lap_biliary | 12 | v3.1 | 0 | 36 | 7/12 | 0.60 | 0.39 | 0 |
| lap_hernia | 6 | v1.5 | 12 | 5 | 5/6 | 0.61 | 0.67 | 0 |
| lap_hernia | 6 | v3.1 | 11 | 5 | 4/6 | 0.55 | 0.44 | 0 |

The "instrument" and "obscured" percentiles use the same SigLIP2 scores the
selector optimizes, so treat them as sanity checks. The independent signal is
the judge below.

**Blind pairwise VLM judges** (`judge.py`, public videos only; A/B order
randomized and both orders asked). Claude Opus 5.5 preferred v3.1 8 of 12 times
and Gemini 3.8 Flash 9 of 12 — **17/24 overall**. v3.1 won every call on the
four longer videos. It lost on the two shortest (26 s and 19 s), where a
playback overlay covers most frames.

**Cost.**
- On serverless env v5 (4 vCPU): 371 ms/frame at batch 16, and 2.5 GB peak
  memory for a 3-minute video. A 1-hour video is ~22 min of model time.
- On a Mac: ~110 ms/frame.
- ONNX fp32 is 355 MB; it matches PyTorch at cosine 0.9999997.

## Reproduce

All scripts write to `/tmp/fs_v3_eval`. They need torch, transformers,
onnxruntime, av and opencv in a venv.

- `extract.py` decodes videos (MP4) at 1 fps and computes quality signals plus
  SigLIP2 features. Set `FS_V3_INBODY_GLOBS` to globs of known in-body frames
  for calibration.
- `compare.py` runs v1.5 vs v3.1 on identical candidates: metrics plus contact sheets.
- `judge.py` runs the blind pairwise judge via Databricks FMAPI (CLI profile
  `vlm`). Use it on public videos only.
- `export_onnx.py` exports the vision ONNX and text-concept npz and benchmarks them.
- `run_notebook_local.py <video.mp4> <out_dir>` executes the real ingest
  notebook locally with stubbed Databricks/Lakebase/decord.

Deploy the model by copying `siglip2_base_vision.onnx` and
`siglip2_text_concepts.npz` to `<volume>/local_models/siglip2_vlmwb_v1/`; the
app passes that directory as `siglip_model_dir`.

## Known gaps

- **Open-surgery frames** pass as "in body". They're surgical but off-domain
  for arthroscopy training. A domain prompt pair (arthroscopic vs open surgery)
  is the next step.
- **Burned-in captions** on arthroscopic frames are only softly penalized.
- **No real scope-withdrawal footage** (camera leaving the patient) has been
  tested yet. The face-through-the-scope case is covered by prompts and
  synthetic frames only.
- **Speed:** int8 quantization or a smaller tower would roughly halve the
  serverless cost.
- **Re-select:** `/reselect` in app.py still runs v1.5 logic. It reads the
  persisted utility as `value`, but has no MMR.
