"""Execute the real smart_frames ingest notebook locally with stubs (no Databricks).

Usage: python run_notebook_local.py <video.mp4> <out_dir> [siglip_dir|""]
"""
import json
import os
import sys
import types

import cv2
import numpy as np
import yaml

video, out_dir = sys.argv[1], sys.argv[2]
siglip_dir = sys.argv[3] if len(sys.argv) > 3 else "/tmp/fs_v3_eval/onnx"
os.makedirs(out_dir, exist_ok=True)
NB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "ingest_notebooks", "smart_frames", "notebook.py")

# torch absent, as on the serverless CPU env
sys.modules["torch"] = None


class VideoReader:
    """decord.VideoReader shim over cv2 (index → RGB ndarray, optional resize)."""

    def __init__(self, path, ctx=None, width=-1, height=-1):
        self.path = path
        self.cap = cv2.VideoCapture(path)
        self.n = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS)) or 25.0
        self.size = (width, height) if width and width > 0 and height and height > 0 else None

    def __len__(self):
        return self.n

    def get_avg_fps(self):
        return self.fps

    def __getitem__(self, i):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, bgr = self.cap.read()
        if not ok:
            raise IndexError(i)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if self.size:
            rgb = cv2.resize(rgb, self.size, interpolation=cv2.INTER_AREA)
        return types.SimpleNamespace(asnumpy=lambda: rgb, shape=rgb.shape)

    def get_frame_timestamp(self, idx):
        # Real presentation times (VFR-aware), like decord, read once via PyAV.
        if not hasattr(self, "_pts"):
            import av
            with av.open(self.path) as c:
                st = c.streams.video[0]
                pts = sorted(float(p.pts * p.time_base) for p in c.demux(st) if p.pts is not None)
            self._pts = np.array(pts[: self.n] + [pts[-1]] * max(0, self.n - len(pts)))
        i = np.atleast_1d(idx)
        nxt = np.append(self._pts[1:], self._pts[-1] + 1 / self.fps)
        return np.stack([self._pts[i], nxt[i]], 1)


decord = types.ModuleType("decord")
decord.VideoReader = VideoReader
decord.cpu = lambda i=0: None
sys.modules["decord"] = decord

status_log = []


class _Cur:
    def __enter__(self): return self
    def __exit__(self, *a): return False
    def execute(self, sql, params=None):
        if "UPDATE videos SET status" in sql:
            status_log.append((params[0], params[1]))
    def executemany(self, *a, **k): pass
    def fetchone(self): return None
    def fetchall(self): return []


class _Conn:
    def __enter__(self): return self
    def __exit__(self, *a): return False
    def cursor(self): return _Cur()
    def transaction(self): return self


psycopg = types.ModuleType("psycopg")
psycopg.connect = lambda **k: _Conn()
sys.modules["psycopg"] = psycopg

sdk = types.ModuleType("databricks.sdk")


class _WC:
    def __init__(self, *a, **k):
        self.api_client = types.SimpleNamespace(do=lambda *a, **k: {"token": "local-test"})


sdk.WorkspaceClient = _WC
sys.modules["databricks"] = types.ModuleType("databricks")
sys.modules["databricks.sdk"] = sdk

cfg = {"video_path": video, "video_name": os.path.basename(video), "video_id": "local-test",
       "output_dir": out_dir, "pg_host": "x", "pg_port": 5432, "pg_dbname": "x", "pg_user": "x",
       "pg_instance": "local", "max_frames": 12, "siglip_model_dir": siglip_dir}
cfg_path = os.path.join(out_dir, "cfg.yaml")
yaml.safe_dump(cfg, open(cfg_path, "w"))


class _Exit(Exception):
    pass


def _exit(payload):
    raise _Exit(payload)


dbutils = types.SimpleNamespace(
    widgets=types.SimpleNamespace(text=lambda *a, **k: None, get=lambda name: cfg_path),
    notebook=types.SimpleNamespace(exit=_exit))

src = open(NB).read()
g = {"__name__": "__main__", "dbutils": dbutils}
try:
    exec(compile(src, NB, "exec"), g)
    print("NOTEBOOK FINISHED WITHOUT EXIT")
except _Exit as e:
    res = json.loads(e.args[0])
    print("EXIT PAYLOAD:", json.dumps(res, indent=1))
print("STATUS LOG:", status_log[-2:])
# Debug aid: every candidate with its gate reason and OOB signals.
keep = ("timestamp_s", "frame_index", "gate_reason", "oob_zs", "border_lum", "oob_detector_flag")
json.dump([{k: c.get(k) for k in keep} for c in g.get("candidates", [])],
          open(os.path.join(out_dir, "candidates_all.json"), "w"))
print("border_base:", g.get("border_base"), "| full_frame:", g.get("full_frame"),
      "| mask coverage:", round(float(g["M"].mean()), 3) if "M" in g else None)
side = json.load(open(os.path.join(out_dir, "candidates.json")))
sel = [c for c in side["candidates"] if c["selected"]]
print(f"selected {len(sel)}: " + ", ".join(f"{c['timestamp_s']:.0f}s" for c in sel))
