"""Decode videos at 1 fps and compute quality + SigLIP2 features (local only).

Outputs /tmp/fs_v3_eval/feats/<name>.npz with per-candidate arrays, and
/tmp/fs_v3_eval/feats/pools.npz for still-image pools (in-body / OOB sets).
"""
import glob
import json
import os
import re
import sys

import av
import cv2
import numpy as np
import torch
from transformers import AutoModel, AutoProcessor

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT = "/tmp/fs_v3_eval/feats"
os.makedirs(OUT, exist_ok=True)

VIDEOS = {n: f"/tmp/fs_v3_eval/videos/{n}.mp4" for n in (
    "knee_cartilage", "shoulder_chl", "knee_tibial_insert", "knee_femoral", "lap_biliary", "lap_hernia")}

PROMPTS = {
    "inbody": ["an arthroscopic view inside a knee joint",
               "an endoscopic surgical view of tissue inside the body",
               "a laparoscopic view inside the abdomen",
               "an arthroscopic image of cartilage and synovium"],
    "oob": ["a photo of an operating room", "a photo of a person's face",
            "a photo of human skin on the outside of the body", "a computer screen with text",
            "a title slide with text", "surgical drapes and medical equipment",
            "a hospital room with a patient monitor",
            "an X-ray radiograph image", "a CT scan slice", "an MRI scan image",
            "a medical illustration or diagram", "an open surgical incision with skin around it",
            "a gloved hand holding a medical implant", "a person's hands outside the body",
            "a video player interface with playback controls"],
    "inst": ["a surgical instrument visible in the arthroscopic view",
             "a metal surgical tool tip touching tissue", "an arthroscopic shaver inside the joint",
             "a probe hook inside the knee joint", "surgical forceps grasping tissue"],
    "noinst": ["tissue only with no surgical instrument visible",
               "an empty view of tissue without any tools"],
    "bad": ["a blurry out of focus surgical view", "a surgical view obscured by bubbles",
            "a surgical view obscured by blood", "a hazy cloudy endoscopic view",
            "a frame with heavy motion blur", "a frame washed out by bright glare",
            "a frame with a text overlay covering the image"],
    "clear": ["a sharp clear well lit surgical view", "a crisp detailed view of tissue"],
}

dev = "cpu"


def _t(o):
    """transformers 5.x returns ModelOutput objects from get_*_features."""
    if torch.is_tensor(o):
        return o
    return o.pooler_output if getattr(o, "pooler_output", None) is not None else o[0]
model = AutoModel.from_pretrained("google/siglip2-base-patch16-224").eval()
proc = AutoProcessor.from_pretrained("google/siglip2-base-patch16-224")
with torch.no_grad():
    names, texts = [], []
    for k, ps in PROMPTS.items():
        for p in ps:
            names.append(k)
            texts.append(p)
    t_in = proc(text=texts, padding="max_length", max_length=64, return_tensors="pt")
    temb = _t(model.get_text_features(**t_in))
    temb = temb / temb.norm(dim=-1, keepdim=True)
scale = float(model.logit_scale.exp())
bias = float(model.logit_bias)
concept_of = np.array(names)
np.savez(f"{OUT}/text.npz", temb=temb.numpy(), concepts=concept_of, texts=np.array(texts), scale=scale, bias=bias)


def image_feats(rgbs):
    embs = []
    with torch.no_grad():
        for i in range(0, len(rgbs), 32):
            x = proc(images=rgbs[i:i + 32], return_tensors="pt")
            e = _t(model.get_image_features(**x))
            embs.append((e / e.norm(dim=-1, keepdim=True)).numpy())
    return np.concatenate(embs) if embs else np.zeros((0, temb.shape[1]))


def concept_logits(emb):
    """logsumexp of SigLIP logits over each concept's prompts → (N, n_concepts)."""
    lg = scale * emb @ temb.numpy().T + bias
    out = {}
    for c in PROMPTS:
        m = concept_of == c
        sub = lg[:, m]
        mx = sub.max(1, keepdims=True)
        out[c] = (mx + np.log(np.exp(sub - mx).sum(1, keepdims=True))).ravel()
    return out


def quality(rgb):
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    lit = g > 25
    gb = cv2.GaussianBlur(g, (0, 0), 2.0).astype(np.float32)
    grad = np.sqrt(cv2.Sobel(gb, cv2.CV_32F, 1, 0) ** 2 + cv2.Sobel(gb, cv2.CV_32F, 0, 1) ** 2)
    v = g[lit] if lit.any() else g.ravel()
    return dict(grad=float(grad[lit].mean()) if lit.any() else 0.0, contrast=float(v.std()),
                whiteout=float((v > 240).mean()), mean_lum=float(v.mean()), lit_frac=float(lit.mean()))


# Current notebook OOB heuristic, loaded from the shipped notebook source.
sys.path.insert(0, f"{REPO}/experiments")
from test_oob_guard import _is_oob_frame  # noqa: E402


def decode_1fps(path):
    frames, ts = [], []
    with av.open(path) as c:
        s = c.streams.video[0]
        nxt = 0.0
        for f in c.decode(s):
            t = float(f.time or 0.0)
            if t + 1e-6 >= nxt:
                rgb = f.to_ndarray(format="rgb24")
                if max(rgb.shape[:2]) > 640:
                    sc = 640 / max(rgb.shape[:2])
                    rgb = cv2.resize(rgb, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA)
                frames.append(rgb)
                ts.append(t)
                nxt += 1.0
    return frames, np.array(ts)


for name, path in VIDEOS.items():
    if not path or not os.path.exists(path):
        print("skip", name, path)
        continue
    frames, ts = decode_1fps(path)
    q = [quality(f) for f in frames]
    emb = image_feats(frames)
    cl = concept_logits(emb)
    heur = np.array([_is_oob_frame(f)[0] for f in frames])
    small = np.stack([cv2.resize(f, (160, 120), interpolation=cv2.INTER_AREA) for f in frames])
    np.savez_compressed(f"{OUT}/{name}.npz", ts=ts, emb=emb, thumbs=small, heur_oob=heur,
                        **{k: np.array([d[k] for d in q]) for k in q[0]},
                        **{f"L_{k}": v for k, v in cl.items()})
    print(f"{name}: {len(frames)} candidates, dur {ts[-1]:.0f}s")


def load_imgs(paths):
    out = []
    for p in paths:
        bgr = cv2.imread(p)
        if bgr is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if max(rgb.shape[:2]) > 640:
            sc = 640 / max(rgb.shape[:2])
            rgb = cv2.resize(rgb, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA)
        out.append(rgb)
    return out


# Still-image pools. Private frames: local model only, never displayed or sent anywhere.
# In-body calibration frames come from FS_V3_INBODY_GLOBS (os.pathsep-separated
# globs) so no local/private paths live in the repo. Those frames are scored by
# the local model only — never displayed or sent anywhere.
pools = {
    "cust_inbody": sorted(p for g in os.environ.get("FS_V3_INBODY_GLOBS", "").split(os.pathsep) if g
                          for p in glob.glob(g)),
    "oob_pos": sorted(glob.glob("/tmp/fs_v3_eval/oob_pos/*")) + sorted(glob.glob("/tmp/fs_v3_data/oob_images/*.png")),
}
res = {}
for k, ps in pools.items():
    imgs = load_imgs(ps)
    if not imgs:
        continue
    emb = image_feats(imgs)
    cl = concept_logits(emb)
    heur = np.array([_is_oob_frame(f)[0] for f in imgs])
    res.update({f"{k}__{c}": v for c, v in cl.items()})
    res[f"{k}__heur"] = heur
    res[f"{k}__n"] = np.array(len(imgs))
    print(f"pool {k}: {len(imgs)} images")
np.savez(f"{OUT}/pools.npz", **res)
json.dump({"scale": scale, "bias": bias}, open(f"{OUT}/meta.json", "w"))
print("done")
