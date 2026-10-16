# End-to-end demo: from surgical video to a fine-tuned model that beats its base

**Story.** Smart frames from public surgery videos → a frontier model drafts labels →
a person verifies them → the same small open model (Qwen3-VL-8B) gets better
three ways: default prompt → optimized prompt (GEPA) → LoRA fine-tune. It is scored
on frames it never saw, and you own and serve the result.

## Rehearsal result (2026-10-16)

33 held-out frames: the last 30% of each of the 5 CC-BY demo videos, with a 5 s gap
before the cut. They were never used for training or prompt optimization. All
three variants start from the same training prompt (`DEFAULT_FT_PROMPT`).

| Variant | Accuracy | Instrument frames (8) | False "instrument" on 25 empty views |
|---|---|---|---|
| Always answer "no instrument" | 76% (25/33) | 0/8 | 0 |
| Qwen3-VL-8B, base | 61% (20/33) | 2/8 | 7 |
| + GEPA-optimized prompt (5 rounds) | 79% (26/33) | 1/8 | 0 |
| **LoRA fine-tune (80 labels, 3 epochs, 1×A10, ~8 min)** | **88% (29/33)** | **4/8** | **0** |

- **Base model:** it hedges. 11 of its 13 errors are the catch-all `other_metal_tool`,
  given for empty views and for real graspers or cannulas.
- **GEPA:** the optimized prompt removes those false alarms but over-corrects, calling
  some graspers and cannulas "no instrument".
- **Fine-tune:** gets every empty view and every grasper right. Its 4 misses are
  rare classes (shaver, 2× cannula, probe).

**Caveats.** Read these before presenting the numbers.
- **The labels are provisional.** They are the 115 frames where Claude Opus 5.5 and
  Gemini 3.8 agree, tagged `source = auto-consensus:opus-5-5+gemini-3-8-flash`.
  Verify them in Label before the real demo, then re-run the fine-tune and scoring.
  A frontier-model "reference" accuracy isn't meaningful while it also defines the labels.
- **The sample is small.** With 33 frames, one frame is 3 points. The held-out set is
  the later part of the same procedures, not new videos.
- **Agreement filtering made the labeled frames easier.** The 43 frames where the two
  models disagree are unlabeled, and they are the hard ones.

## Demo script (~15 min)

The runs are already done and saved as snapshots, so nothing waits on a GPU live.
Snapshot names all start with `demo ·`.

| # | Tab | Do | Say / show |
|---|---|---|---|
| 1 | Library | Open `cc-by-shoulder-coracohumeral-release.mp4` | v3.1 smart frames and the status line: candidates, out-of-body rejected, picks |
| 2 | Studio | Play the shoulder video, click picks | Picks land on real moments (instrument entry, burr/shaver work) |
| 3 | Playground | Source "Smart-extracted", load `demo · drafts (Opus 5.5 + Gemini 3.8)` | Two frontier models pre-label every frame |
| 4 | Label | Stage from the drafts snapshot, verify a few, show "Save all drafts" | Humans confirm instead of typing labels from scratch |
| 5 | Playground → Optimize | Show the GEPA run on `demo · train frames` with gold teacher, student `qwen3-vl-8b` | Prompt optimization: 61% → 76% on training frames, no training job needed |
| 6 | Refine | Filter by snapshot `demo · train frames`, show the preview and the finished run | 80 labels, 3 epochs on 1×A10, MLflow run, UC model registered |
| 7 | **Compare** | Select the three `demo · held-out · …` snapshots | **Accuracy vs gold: 61% → 79% → 88%**; toggle "only frames where snapshots disagree" |
| 8 | Deploy (optional) | Serve the fine-tuned model | Your own endpoint, under Unity Catalog governance |

Do **not** use Playground's "Eval set (curated 31)" source, the older
`ft_1779469931_qwen3vl8b` model, or labels from before this rehearsal. Those come
from private footage.

## Before the real demo

1. In Label, stage `demo · drafts`, then verify or correct the 115 provisional labels
   and resolve the 43 disagreements.
2. Re-run `launch` → `collect_*` → `score` (below) and update the table above.
3. Smoke-test the tabs in step order the day before (GPU availability varies; the
   1×H100 pool queued 20+ min during rehearsal).

## Reproduce

`rehearse.py` drives the app's API with your Databricks CLI login
(`DATABRICKS_PROFILE`, default `vlm`). Steps, in order:

```
python3 -I rehearse.py reselect     # 158 frames across the 5 demo videos
python3 -I rehearse.py drafts       # Opus 5.5 + Gemini 3.8 → drafts snapshot
python3 -I rehearse.py consensus    # agreeing frames → provisional labels
python3 -I rehearse.py launch       # temporal split + base run, GEPA, fine-tune (GPU)
python3 -I rehearse.py collect_base
python3 -I rehearse.py eval_gepa && python3 -I rehearse.py collect_gepa
python3 -I rehearse.py eval_ft   && python3 -I rehearse.py collect_ft
python3 -I rehearse.py score
```

Wait for each GPU job to finish before its `collect_*` / `eval_*` step. Run IDs are
in the state file.

## Issues the rehearsal found (fixed on `mvp-gaps`)

- Snapshot saves failed on a `snapshots` table owned by another role (`CREATE INDEX IF NOT EXISTS` needs ownership).
- Claude Opus 5.5 rejects `response_format` with a "structured output" error that the Playground retry didn't recognize.
- An unbounded `--upgrade transformers` pulled 5.x, which removed `warmup_ratio`. Every fine-tune failed until the versions were capped.
- The fine-tune default GPU is now 1×A10. It started immediately, while 1×H100 queued.
