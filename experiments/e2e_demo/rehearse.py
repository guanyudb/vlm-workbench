"""Rehearsal pipeline for the vlm-workbench end-to-end demo (see README.md),
driven through the app's own API, so every artifact (snapshots, labels, runs)
is visible in the UI. Run the steps in order; each saves progress to STATE:

  reselect → drafts → retry (only if some draft calls failed) → consensus → split
  → launch (base run + GEPA + fine-tune on GPU) → collect_base
  → eval_gepa → collect_gepa → eval_ft → collect_ft → score

Env: DATABRICKS_PROFILE (default "vlm"), VLMWB_APP (default "vlm-workbench").
"""
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request

PROFILE = os.environ.get("DATABRICKS_PROFILE", "vlm")
APP = json.loads(subprocess.run(
    ["databricks", "apps", "get", os.environ.get("VLMWB_APP", "vlm-workbench"), "-p", PROFILE, "-o", "json"],
    capture_output=True, text=True, check=True).stdout)["url"].rstrip("/")
STATE = os.environ.get("VLMWB_DEMO_STATE", "/tmp/vlmwb_demo/state.json")
DRAFTERS = ["databricks-claude-opus-5-5", "databricks-gemini-3-8-flash"]
LABEL_SOURCE = "auto-consensus:opus-5-5+gemini-3-8-flash"
# Frames per video after re-select (capped by each video's gated candidates).
TARGET_K = {
    "cc-by-shoulder-coracohumeral-release.mp4": 60,
    "cc-by-lap-biliary-peritonitis.mp4": 40,
    "cc-by-knee-cartilage-regeneration.mp4": 30,
    "cc-by-knee-femoral-loosening.mp4": 20,
    "cc-by-knee-tibial-insert.mp4": 8,
}

_tok = {"v": None, "t": 0.0}


def token():
    if not _tok["v"] or time.time() - _tok["t"] > 1500:
        out = subprocess.run(["databricks", "auth", "token", "-p", PROFILE], capture_output=True, text=True, check=True)
        _tok["v"], _tok["t"] = json.loads(out.stdout)["access_token"], time.time()
    return _tok["v"]


def api(method, path, body=None, timeout=900):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(APP + path, data=data, method=method,
                                 headers={"Authorization": f"Bearer {token()}", "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read() or b"null")
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"{method} {path} → {e.code}: {e.read()[:400]!r}")


def load_state():
    return json.load(open(STATE)) if os.path.exists(STATE) else {}


def save_state(s):
    json.dump(s, open(STATE, "w"), indent=1)


def primary_class(parsed):
    """Same rule as the app's _instrument_class_from_result / lib/instruments."""
    if not isinstance(parsed, dict):
        return None
    if isinstance(parsed.get("instrument"), str) and parsed["instrument"]:
        return parsed["instrument"].strip().lower()
    inst = parsed.get("instruments")
    if isinstance(inst, list) and inst and isinstance(inst[0], dict) and isinstance(inst[0].get("class"), str):
        return inst[0]["class"].strip().lower()
    if isinstance(inst, list) and not inst:
        return "no_instrument_visible"
    return None


def demo_frames():
    out = {}
    for name in TARGET_K:
        rows = api("GET", f"/api/frames?source=extracted&video={urllib.request.quote(name)}")
        out[name] = sorted(r["path"] for r in rows)
    return out


def cmd_reselect():
    vids = {v["name"]: v for v in api("GET", "/api/ingest/videos")["videos"]}
    for name, k in TARGET_K.items():
        r = api("POST", f"/api/videos/{vids[name]['id']}/reselect", {"max_frames": k})
        print(f"{name}: selected {r.get('n_selected')} (asked {k})")
    frames = demo_frames()
    s = load_state()
    s["frames"] = frames
    save_state(s)
    print("total frames:", sum(len(v) for v in frames.values()))


def cmd_drafts():
    s = load_state()
    frames = [p for v in s["frames"].values() for p in v]
    results, t0 = [], time.time()
    for i in range(0, len(frames), 12):
        batch = frames[i:i + 12]
        r = api("POST", "/api/playground/run", {"frame_paths": batch, "model_names": DRAFTERS})
        results += r["results"]
        s["draft_results"] = results
        save_state(s)  # persist every batch — a later failure must not lose paid-for calls
        print(f"  {i + len(batch)}/{len(frames)} frames · ok {r['successful']}/{r['total']} · {r['elapsed_s']}s")
    snap = api("POST", "/api/playground/snapshots", {
        "name": "demo · drafts (Opus 5.5 + Gemini 3.8)",
        "notes": "Rehearsal drafts on the 5 CC-BY demo videos; stage in Label to verify.",
        "frame_paths": frames, "model_names": DRAFTERS,
        "prompt": api("GET", "/api/task-config")["rendered_prompt"],
        "best_model": DRAFTERS[0], "results": results, "elapsed_s": round(time.time() - t0, 1),
    })
    s["drafts_snapshot"] = snap.get("id")
    s["draft_results"] = results
    save_state(s)
    print("snapshot", snap.get("id"), "·", len(results), "results")


def cmd_retry():
    """Re-run only failed (model, frame) calls, then replace the drafts snapshot."""
    s = load_state()
    path_of = {os.path.basename(p): p for v in s["frames"].values() for p in v}
    ok = {(r["model"], r["frame"]): r for r in s["draft_results"] if r.get("ok")}
    todo = {}
    for r in s["draft_results"]:
        if not r.get("ok"):
            todo.setdefault(r["model"], []).append(path_of[r["frame"]])
    for model, paths in todo.items():
        for i in range(0, len(paths), 12):
            r = api("POST", "/api/playground/run", {"frame_paths": paths[i:i + 12], "model_names": [model]})
            for row in r["results"]:
                if row.get("ok"):
                    ok[(row["model"], row["frame"])] = row
            s["draft_results"] = list(ok.values()) + [x for x in s["draft_results"]
                                                      if not x.get("ok") and (x["model"], x["frame"]) not in ok]
            save_state(s)
            print(f"  {model}: {min(i + 12, len(paths))}/{len(paths)} · ok {r['successful']}/{r['total']}")
    old = s.get("drafts_snapshot")
    frames = [p for v in s["frames"].values() for p in v]
    snap = api("POST", "/api/playground/snapshots", {
        "name": "demo · drafts (Opus 5.5 + Gemini 3.8)",
        "notes": "Rehearsal drafts on the 5 CC-BY demo videos; stage in Label to verify.",
        "frame_paths": frames, "model_names": DRAFTERS,
        "prompt": api("GET", "/api/task-config")["rendered_prompt"],
        "best_model": DRAFTERS[0], "results": s["draft_results"],
    })
    if old:
        api("DELETE", f"/api/playground/snapshots/{old}")
    s["drafts_snapshot"] = snap["id"]
    save_state(s)
    from collections import Counter
    print("snapshot", snap["id"], Counter((r["model"], bool(r.get("ok"))) for r in s["draft_results"]))


def cmd_consensus():
    s = load_state()
    by = {}
    for r in s["draft_results"]:
        if r.get("ok"):
            by.setdefault(r["frame"], {})[r["model"]] = r.get("parsed")
    path_of = {os.path.basename(p): p for v in s["frames"].values() for p in v}
    rows, disagree, missing = [], [], 0
    for fname, preds in by.items():
        a, b = (primary_class(preds.get(m)) for m in DRAFTERS)
        if not a or not b:
            missing += 1
            continue
        if a != b:
            disagree.append((fname, a, b))
            continue
        lead = preds[DRAFTERS[0]] or {}
        rows.append({"frame_path": path_of[fname], "instrument": a,
                     "anatomy": (lead.get("anatomy") or None), "tissue_condition": (lead.get("tissue_condition") or None),
                     "notes": "provisional rehearsal label (two frontier models agree) — verify before the demo",
                     "source": LABEL_SOURCE})
    for i in range(0, len(rows), 50):
        api("POST", "/api/labels/batch", {"rows": rows[i:i + 50]})
    s["consensus_paths"] = [r["frame_path"] for r in rows]
    s["disagreements"] = disagree
    save_state(s)
    print(f"labeled {len(rows)} · disagreed {len(disagree)} · unparsed {missing}")


# Same text as app.py DEFAULT_FT_PROMPT — the fine-tune's training prompt.
FT_PROMPT = (
    'Identify the surgical instrument by its visible visual features. '
    'Respond with strict JSON: {"instrument": "<class>", "anatomy": "<short>", '
    '"tissue_condition": "<short>"} where <class> is one of: probe, shaver, burr, '
    'grasper, biter, suture_passer, anchor_driver, electrocautery, cannula, '
    'scissors, drill_guide, trocar, knot_pusher, rasp, other_metal_tool, '
    'no_instrument_visible.'
)
HOLDOUT_FRAC, GAP_S = 0.30, 5.0


def cmd_launch():
    """Temporal split (last 30% of each video held out, 5 s gap), split
    snapshots, then the three GPU jobs: base run, GEPA, fine-tune."""
    s = load_state()
    labeled = set(s["consensus_paths"])
    dur = {v["name"]: float(v.get("duration_s") or 0) for v in api("GET", "/api/ingest/videos")["videos"]}
    train, held = [], []
    for name in TARGET_K:
        cut = (1 - HOLDOUT_FRAC) * dur[name]
        for r in api("GET", f"/api/frames?source=extracted&video={urllib.request.quote(name)}"):
            if r["path"] not in labeled:
                continue
            t = float(r.get("timestamp_s") or 0)
            if t >= cut:
                held.append(r["path"])
            elif t < cut - GAP_S:
                train.append(r["path"])
    prompt = api("GET", "/api/task-config")["rendered_prompt"]

    def snap(name, paths, notes):
        base = set(os.path.basename(p) for p in paths)
        return api("POST", "/api/playground/snapshots", {
            "name": name, "notes": notes, "frame_paths": sorted(paths), "model_names": DRAFTERS,
            "prompt": prompt, "best_model": DRAFTERS[0],
            "results": [r for r in s["draft_results"] if r["frame"] in base]})["id"]
    s["train_paths"], s["held_paths"] = sorted(train), sorted(held)
    s["train_snapshot"] = snap("demo · train frames (first 70% of each video)", train,
                               "Rehearsal training split; labels are provisional (two frontier models agree).")
    s["held_snapshot"] = snap("demo · held-out frames (last 30% of each video)", held,
                              "Rehearsal evaluation split; never used for training or prompt optimization.")
    save_state(s)
    print(f"train {len(train)} · held-out {len(held)} · snapshots {s['train_snapshot']} / {s['held_snapshot']}")

    jobs = {}
    jobs["base"] = api("POST", "/api/playground/run/local",
                       {"model_name": "qwen3-vl-8b", "frame_paths": s["held_paths"], "prompt": FT_PROMPT})
    jobs["gepa"] = api("POST", "/api/playground/optimize",
                       {"snapshot_id": s["train_snapshot"], "optimizer": "gepa", "teacher_model": "__gold__",
                        "student_model": "qwen3-vl-8b", "n_rounds": 5, "seed_prompt_override": FT_PROMPT})
    jobs["finetune"] = api("POST", "/api/training/finetune",
                           {"base_model_name": "qwen3-vl-8b", "snapshot_ids": [s["train_snapshot"]],
                            "train_prompt": FT_PROMPT, "num_epochs": 3, "accelerator": "GPU_1xH100"})
    s["jobs"] = {k: {kk: v.get(kk) for kk in ("run_id", "databricks_run_id", "uc_model_name", "n_train",
                                              "n_eval_examples", "n_frames")} for k, v in jobs.items()}
    save_state(s)
    print(json.dumps(s["jobs"], indent=1))


def _run_status(job):
    return api("GET", f"/api/playground/run/local/{job['run_id']}?databricks_run_id={job['databricks_run_id']}")


def cmd_collect(kind):
    """Local-run results → a Compare-ready snapshot (rows keyed by frame basename)."""
    s = load_state()
    job = s["jobs"][kind]
    st = _run_status(job)
    model = st.get("model_name") or job.get("model_name") or kind
    label = {"base": "qwen3-vl-8b (base)", "gepa_eval": "qwen3-vl-8b + GEPA prompt",
             "ft_eval": "qwen3-vl-8b fine-tuned"}[kind]
    rows = []
    for r in st.get("results") or []:
        base = os.path.basename(r.get("frame") or "")
        rows.append({**r, "frame": base, "model": label})
    if not rows:
        raise SystemExit(f"{kind}: no results yet ({st.get('state_message') or st})")
    sid = api("POST", "/api/playground/snapshots", {
        "name": f"demo · held-out · {label}", "notes": "Rehearsal evaluation on the held-out split.",
        "frame_paths": s["held_paths"], "model_names": [label], "best_model": label,
        "prompt": job.get("prompt") or FT_PROMPT, "results": rows})["id"]
    s.setdefault("eval_snapshots", {})[kind] = sid
    s.setdefault("eval_rows", {})[kind] = rows
    save_state(s)
    print(f"{kind}: {sum(1 for r in rows if r.get('ok'))}/{len(rows)} ok → snapshot {sid}")


def cmd_eval_gepa():
    s = load_state()
    g = s["jobs"]["gepa"]
    st = api("GET", f"/api/playground/optimize/{g['run_id']}?databricks_run_id={g['databricks_run_id']}")
    best = (st.get("best_prompt") or "").strip()
    if not best:
        raise SystemExit(f"no best_prompt yet: {json.dumps({k: st.get(k) for k in ('life_cycle_state', 'result_state', 'state_message', 'score', 'error_detail')})[:400]}")
    s["gepa_prompt"], s["gepa_score"], s["gepa_rounds"] = best, st.get("score"), st.get("round_scores")
    r = api("POST", "/api/playground/run/local", {"model_name": "qwen3-vl-8b", "frame_paths": s["held_paths"], "prompt": best})
    s["jobs"]["gepa_eval"] = {"run_id": r["run_id"], "databricks_run_id": r["databricks_run_id"], "prompt": best}
    save_state(s)
    print("GEPA train score", st.get("score"), "rounds", st.get("round_scores"), "· eval run", r["databricks_run_id"])
    print("--- optimized prompt ---\n" + best[:1500])


def cmd_eval_ft():
    s = load_state()
    name = s["jobs"]["finetune"]["uc_model_name"]
    names = [m["name"] for m in api("GET", "/api/local-models")]
    if name not in names:
        raise SystemExit(f"{name} not in local models yet: {names}")
    r = api("POST", "/api/playground/run/local", {"model_name": name, "frame_paths": s["held_paths"], "prompt": FT_PROMPT})
    s["jobs"]["ft_eval"] = {"run_id": r["run_id"], "databricks_run_id": r["databricks_run_id"], "prompt": FT_PROMPT}
    save_state(s)
    print("fine-tuned eval run", r["databricks_run_id"])


def cmd_score():
    from collections import Counter
    s = load_state()
    gold = {os.path.basename(r["frame_path"]): r["instrument"]
            for r in api("POST", "/api/labels/query", {"frame_paths": s["held_paths"]})}
    vocab = set(api("GET", "/api/task-config")["vocabulary"])
    n = len(gold)
    maj, maj_n = Counter(gold.values()).most_common(1)[0]
    print(f"held-out frames with labels: {n} · class mix {dict(Counter(gold.values()).most_common())}")
    print(f"majority-class baseline ('{maj}'): {maj_n}/{n} = {maj_n / n:.0%}")
    inst = [f for f, g in gold.items() if g != "no_instrument_visible"]
    table = []
    for kind in ("base", "gepa_eval", "ft_eval"):
        rows = (s.get("eval_rows") or {}).get(kind)
        if not rows:
            continue
        pred = {r["frame"]: primary_class(r.get("parsed")) if r.get("ok") else None for r in rows}
        acc = sum(pred.get(f) == g for f, g in gold.items())
        acc_i = sum(pred.get(f) == gold[f] for f in inst)
        in_vocab = sum(1 for f in gold if pred.get(f) in vocab)
        table.append((rows[0]["model"], acc, acc_i, in_vocab))
    print(f"{'variant':30s} {'accuracy':>12s} {'instrument frames':>18s} {'valid class':>12s}")
    for name, acc, acc_i, iv in table:
        print(f"{name:30s} {acc:3d}/{n} {acc / n:4.0%}   {acc_i:3d}/{len(inst)} {acc_i / max(len(inst), 1):4.0%}   {iv:3d}/{n}")
    s["score_table"] = table
    save_state(s)


def cmd_split():
    s = load_state()
    labels = {r["frame_path"]: r["instrument"] for r in api("POST", "/api/labels/query", {"frame_paths": s["consensus_paths"]})}
    from collections import Counter
    for name, paths in s["frames"].items():
        c = Counter(labels[p] for p in paths if p in labels)
        print(f"{name:45s} labeled {sum(c.values()):3d}/{len(paths):3d}  {dict(c.most_common())}")
    print("ALL", dict(Counter(labels.values()).most_common()))


if __name__ == "__main__":
    {"reselect": cmd_reselect, "drafts": cmd_drafts, "retry": cmd_retry, "launch": cmd_launch, "eval_gepa": cmd_eval_gepa, "eval_ft": cmd_eval_ft,
     "score": cmd_score, "collect_base": lambda: cmd_collect("base"), "collect_gepa": lambda: cmd_collect("gepa_eval"),
     "collect_ft": lambda: cmd_collect("ft_eval"), "consensus": cmd_consensus, "split": cmd_split}[sys.argv[1]]()
