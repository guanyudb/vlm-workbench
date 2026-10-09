"""Blind pairwise VLM judge: current pipeline vs v3 picks (public CC-BY videos only)."""
import base64
import json
import os
import random
import subprocess
import urllib.request

import av
import cv2
import numpy as np

R = json.load(open("/tmp/fs_v3_eval/out/results.json"))
PROFILE = os.environ.get("DATABRICKS_PROFILE", "vlm")
HOST = json.loads(subprocess.run(["databricks", "auth", "describe", "-p", PROFILE, "-o", "json"],
                                 capture_output=True, text=True).stdout)["details"]["host"].rstrip("/")
JUDGES = ["databricks-claude-opus-5-5", "databricks-gemini-3-8-flash"]
TOK = json.loads(subprocess.run(["databricks", "auth", "token", "-p", PROFILE], capture_output=True, text=True).stdout)["access_token"]

PROMPT = """You are curating training data for a vision-language model that must identify surgical instruments and anatomy in endoscopic/arthroscopic video frames.
Two candidate frame sets were selected from the SAME surgical video (frames in time order, numbered). Judge which set is better training data.
Criteria, in priority order:
1. Every frame is a real in-body surgical view — penalize title cards, text slides, black frames, CT/diagram images, and any out-of-body shots (hands, skin, room).
2. Frames are clear (sharp, not obscured by blur, bubbles, blood or haze).
3. Instruments are visible where the procedure uses them, across different instruments/actions.
4. The set covers distinct moments of the procedure, without near-duplicate frames.
Respond with strict JSON only: {"winner": "A" | "B" | "tie", "reason": "<one or two sentences>"}"""


def frames_at(path, times):
    want = sorted(times)
    out, j = {}, 0
    with av.open(path) as c:
        for f in c.decode(c.streams.video[0]):
            t = float(f.time or 0)
            while j < len(want) and t + 1e-3 >= want[j]:
                out[want[j]] = f.to_ndarray(format="rgb24")
                j += 1
            if j >= len(want):
                break
    return [out[t] for t in times if t in out]


def sheet_b64(imgs):
    tiles = []
    for n, im in enumerate(imgs, 1):
        t = cv2.resize(im, (320, 240), interpolation=cv2.INTER_AREA)
        cv2.putText(t, str(n), (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        tiles.append(t)
    while len(tiles) % 4:
        tiles.append(np.zeros((240, 320, 3), np.uint8))
    img = np.concatenate([np.concatenate(tiles[r:r + 4], 1) for r in range(0, len(tiles), 4)], 0)
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buf.tobytes()).decode()


def ask(model, a_b64, b_b64):
    content = [{"type": "text", "text": PROMPT},
               {"type": "text", "text": "Set A:"}, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{a_b64}"}},
               {"type": "text", "text": "Set B:"}, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b_b64}"}}]
    body = json.dumps({"messages": [{"role": "user", "content": content}], "max_tokens": 4000}).encode()
    req = urllib.request.Request(f"{HOST}/serving-endpoints/{model}/invocations", data=body,
                                 headers={"Authorization": f"Bearer {TOK}", "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        msg = json.loads(r.read())["choices"][0]["message"]["content"]
    if isinstance(msg, list):
        msg = "".join(p.get("text", "") for p in msg if isinstance(p, dict))
    s, e = msg.find("{"), msg.rfind("}")
    return json.loads(msg[s:e + 1])


random.seed(7)
tally = {m: {"v3": 0, "current": 0, "tie": 0} for m in JUDGES}
rows = []
for name, r in R.items():
    path = f"/tmp/fs_v3_eval/videos/{name}.mp4"
    sheets = {p: sheet_b64(frames_at(path, r["picks"][p])) for p in ("current", "v3")}
    for model in JUDGES:
        for order in (("current", "v3"), ("v3", "current")):
            try:
                v = ask(model, sheets[order[0]], sheets[order[1]])
                w = {"A": order[0], "B": order[1]}.get(v.get("winner"), "tie")
            except Exception as ex:
                v, w = {"reason": f"error: {ex}"[:200]}, "error"
            if w in tally[model]:
                tally[model][w] += 1
            rows.append({"video": name, "judge": model, "A": order[0], "winner": w, "reason": v.get("reason", "")})
            print(f"{name:20s} {model.replace('databricks-', ''):18s} A={order[0]:8s} -> {w:8s} {v.get('reason', '')[:110]}")
json.dump({"tally": tally, "rows": rows}, open("/tmp/fs_v3_eval/out/judge.json", "w"), indent=1)
print(json.dumps(tally, indent=1))
