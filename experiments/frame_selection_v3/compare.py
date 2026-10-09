"""Current pipeline vs v3 on the public CC-BY videos (features from extract.py)."""
import json
import math
import os

import cv2
import numpy as np

FE = "/tmp/fs_v3_eval/feats"
OUT = "/tmp/fs_v3_eval/out"
os.makedirs(OUT, exist_ok=True)
VIDS = ["knee_cartilage", "shoulder_chl", "knee_tibial_insert", "knee_femoral", "lap_biliary", "lap_hernia"]
OOB_THR = -1.0


def rank01(x):
    x = np.asarray(x, float)
    if len(x) < 2:
        return np.zeros_like(x)
    return np.argsort(np.argsort(x)) / (len(x) - 1)


def gate(d, use_zeroshot):
    """Same quality gate as the notebook (blackout ~ 1 - lit fraction)."""
    grad_med = np.median(d["grad"])
    # Notebook measures blackout inside the scope mask; approximate with each
    # video's typical lit area (a frame lit < 40% of usual = scope-out).
    ok = ((d["whiteout"] <= 0.35) & (d["lit_frac"] >= 0.4 * np.median(d["lit_frac"]))
          & (d["mean_lum"] >= 18) & (d["mean_lum"] <= 246)
          & (d["grad"] >= 0.55 * grad_med) & ~d["heur_oob"])
    if use_zeroshot:
        ok &= (d["L_oob"] - d["L_inbody"]) <= OOB_THR
    return np.flatnonzero(ok)


def select_v15(ts, value, k, y, dur):
    """The app's _hybrid_coverage_select, verbatim logic, over index arrays."""
    pool = sorted(range(len(ts)), key=lambda i: ts[i])
    n_min = max(1, math.ceil(dur / y))
    picked = []
    for w in range(n_min):
        w0, w1 = w * y, min((w + 1) * y, dur + 1e-6)
        inwin = [i for i in pool if w0 <= ts[i] < w1]
        if inwin:
            best = max(inwin, key=lambda i: value[i])
            if best not in picked:
                picked.append(best)
    for bk in (pool[0], pool[-1]):
        if bk not in picked and len(picked) < max(k, n_min):
            picked.append(bk)
    remaining = sorted([i for i in pool if i not in picked], key=lambda i: -value[i])
    keff = min(max(k, n_min), len(pool))
    for min_gap in (0.5 * y, 0.33 * y, 0.2 * y, 0.0):
        for i in remaining:
            if len(picked) >= keff:
                break
            if i in picked:
                continue
            if all(abs(ts[i] - ts[p]) >= min_gap for p in picked):
                picked.append(i)
        if len(picked) >= keff:
            break
    return sorted(picked, key=lambda i: ts[i])


def select_v3(ts, emb, utility, k, y, dur, lam=0.6, dup=0.95):
    """Coverage anchors (>=1 per y-second window, best utility, preferring a
    non-duplicate) + bookends, then MMR fill: score = utility - lam * max cosine
    to picks. The near-duplicate threshold relaxes (0.95 -> 1.0) only if needed
    to reach K, so static videos still get K frames."""
    pool = sorted(range(len(ts)), key=lambda i: ts[i])
    n_min = max(1, math.ceil(dur / y))
    keff = min(max(k, n_min), len(pool))
    picked = []

    def max_sim(i):
        return max((float(emb[i] @ emb[p]) for p in picked), default=0.0)

    for w in range(n_min):
        w0, w1 = w * y, min((w + 1) * y, dur + 1e-6)
        inwin = [i for i in pool if w0 <= ts[i] < w1 and i not in picked]
        if inwin:
            fresh = [i for i in inwin if max_sim(i) <= dup]
            picked.append(max(fresh or inwin, key=lambda i: utility[i]))
    for bk in (pool[0], pool[-1]):
        if bk not in picked and len(picked) < keff and max_sim(bk) <= dup:
            picked.append(bk)
    # Same progressive temporal spacing as v1.5 (keeps time coverage), with the
    # near-duplicate threshold relaxed last so diversity is enforced first.
    for min_gap, thr in [(g, dup) for g in (0.5 * y, 0.33 * y, 0.2 * y, 0.0)] + [(0.0, 0.97), (0.0, 0.99), (0.0, 1.01)]:
        while len(picked) < keff:
            cand = [(utility[i] - lam * max_sim(i), i) for i in pool
                    if i not in picked and max_sim(i) <= thr
                    and all(abs(ts[i] - ts[p]) >= min_gap for p in picked)]
            if not cand:
                break
            picked.append(max(cand)[1])
    return sorted(picked, key=lambda i: ts[i])


def kmeans_labels(x, k, iters=30, seed=0):
    rng = np.random.default_rng(seed)
    c = x[rng.choice(len(x), size=k, replace=False)]
    for _ in range(iters):
        lab = np.argmax(x @ c.T, 1)
        for j in range(k):
            m = lab == j
            if m.any():
                v = x[m].mean(0)
                c[j] = v / (np.linalg.norm(v) + 1e-9)
    return np.argmax(x @ c.T, 1)


def metrics(picks, d, gated, k_sem):
    e = d["emb"][picks]
    sims = e @ e.T
    iu = np.triu_indices(len(picks), 1)
    t = np.sort(d["ts"][picks])
    dur = float(d["ts"].max())
    gaps = np.diff(np.concatenate([[0.0], t, [dur]]))
    lab = kmeans_labels(d["emb"][gated], k_sem)
    hit = len(set(lab[np.searchsorted(gated, p)] for p in picks if p in set(gated)))
    inst = d["L_inst"] - d["L_noinst"]
    bad = d["L_bad"] - d["L_clear"]
    pct = lambda v: float(np.mean([np.mean(v[gated] <= v[p]) for p in picks]))
    oob = (d["L_oob"] - d["L_inbody"])[picks] > OOB_THR
    return {
        "k": len(picks),
        "mean_cos": float(sims[iu].mean()) if len(picks) > 1 else 0.0,
        "dup_pairs_0.95": int((sims[iu] > 0.95).sum()),
        "max_gap_s": float(gaps.max()),
        "semantic_clusters_hit": f"{hit}/{k_sem}",
        "inst_pctile": pct(inst),
        "obscured_pctile": pct(bad),
        "quality_pctile": pct(rank01(0.6 * rank01(d["grad"]) + 0.4 * rank01(d["contrast"]))),
        "oob_leaks": int(oob.sum()),
    }


def sheet(d, picks, path, title):
    tiles = [d["thumbs"][i] for i in picks]
    cols = 6
    while len(tiles) % cols:
        tiles.append(np.zeros_like(d["thumbs"][0]))
    rows = [np.concatenate(tiles[r:r + cols], 1) for r in range(0, len(tiles), cols)]
    img = np.concatenate(rows, 0)
    cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


results = {}
for n in VIDS:
    d = dict(np.load(f"{FE}/{n}.npz"))
    dur = float(d["ts"].max())
    k = int(min(12, max(4, len(d["ts"]) // 3)))
    y = float(np.clip(2.0 * dur / k, 10.0, 60.0))
    g_cur, g_v3 = gate(d, False), gate(d, True)
    k_sem = max(2, min(k, len(g_v3) // 2))

    def value_of(g):
        return 0.6 * rank01(d["grad"][g]) + 0.4 * rank01(d["contrast"][g]) - 0.5 * d["whiteout"][g]

    cur = g_cur[select_v15(d["ts"][g_cur], value_of(g_cur), k, y, dur)]
    abl = g_v3[select_v15(d["ts"][g_v3], value_of(g_v3), k, y, dur)]
    inst = rank01((d["L_inst"] - d["L_noinst"])[g_v3])
    bad = rank01((d["L_bad"] - d["L_clear"])[g_v3])
    val = value_of(g_v3)
    util = val + 0.35 * inst - 0.5 * bad
    # Quality floor: candidates below the 35th percentile of image quality are
    # never preferred by the semantic terms (glare-washed close-ups etc.).
    util = np.where(rank01(val) < 0.35, util - 1.0, util)
    v3 = g_v3[select_v3(d["ts"][g_v3], d["emb"][g_v3], util, k, y, dur)]
    results[n] = {
        "dur_s": dur, "K": k, "Y": y, "gated_current": int(len(g_cur)), "gated_v3": int(len(g_v3)),
        "current": metrics(cur, d, g_v3, k_sem),
        "v15_on_v3_gate": metrics(abl, d, g_v3, k_sem),
        "v3": metrics(v3, d, g_v3, k_sem),
        "picks": {"current": d["ts"][cur].tolist(), "v3": d["ts"][v3].tolist()},
    }
    sheet(d, cur, f"{OUT}/{n}__current.jpg", "current")
    sheet(d, v3, f"{OUT}/{n}__v3.jpg", "v3")

json.dump(results, open(f"{OUT}/results.json", "w"), indent=1)
cols = ["mean_cos", "dup_pairs_0.95", "max_gap_s", "semantic_clusters_hit", "inst_pctile", "obscured_pctile", "quality_pctile", "oob_leaks"]
print(f"{'video':20s} {'pipeline':16s} " + " ".join(f"{c[:14]:>14s}" for c in cols))
for n, r in results.items():
    for p in ("current", "v15_on_v3_gate", "v3"):
        m = r[p]
        print(f"{n:20s} {p:16s} " + " ".join(f"{(f'{m[c]:.3f}' if isinstance(m[c], float) else str(m[c])):>14s}" for c in cols))
