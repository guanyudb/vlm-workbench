"""Export SigLIP2 vision tower (image -> normalized embedding) to ONNX and benchmark."""
import json
import os
import resource
import time

import numpy as np
import torch
from transformers import AutoModel

OUT = "/tmp/fs_v3_eval/onnx"
os.makedirs(OUT, exist_ok=True)
model = AutoModel.from_pretrained("google/siglip2-base-patch16-224").eval()


class Vision(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, pixel_values):
        o = self.m.get_image_features(pixel_values=pixel_values)
        e = o if torch.is_tensor(o) else (o.pooler_output if getattr(o, "pooler_output", None) is not None else o[0])
        return e / e.norm(dim=-1, keepdim=True)


v = Vision(model).eval()
x = torch.randn(2, 3, 224, 224)
path = f"{OUT}/siglip2_base_vision.onnx"
torch.onnx.export(v, (x,), path, input_names=["pixel_values"], output_names=["embedding"],
                  dynamic_axes={"pixel_values": {0: "n"}, "embedding": {0: "n"}}, opset_version=17, dynamo=False)

import onnxruntime as ort  # noqa: E402

so = ort.SessionOptions()
so.intra_op_num_threads = 4
sess = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
xb = torch.rand(16, 3, 224, 224) * 2 - 1
with torch.no_grad():
    ref = v(xb).numpy()
got = sess.run(None, {"pixel_values": xb.numpy()})[0]
cos = (ref * got).sum(1)
res = {"onnx_mb": round(os.path.getsize(path) / 2**20, 1), "min_cos_vs_torch": float(cos.min())}
for b in (1, 16):
    xi = np.random.rand(b, 3, 224, 224).astype(np.float32) * 2 - 1
    sess.run(None, {"pixel_values": xi})
    t0 = time.time()
    for _ in range(3):
        sess.run(None, {"pixel_values": xi})
    res[f"ms_per_frame_b{b}_4threads"] = round((time.time() - t0) / 3 / b * 1000, 1)
res["peak_rss_mb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 0)  # macOS: bytes

# Text side is precomputed once: embeddings + concept labels + SigLIP scale/bias.
t = np.load("/tmp/fs_v3_eval/feats/text.npz")
np.savez(f"{OUT}/siglip2_text_concepts.npz", temb=t["temb"].astype(np.float32), concepts=t["concepts"],
         texts=t["texts"], scale=t["scale"], bias=t["bias"])
res["text_npz_kb"] = round(os.path.getsize(f"{OUT}/siglip2_text_concepts.npz") / 1024, 1)
print(json.dumps(res, indent=1))
json.dump(res, open(f"{OUT}/bench.json", "w"), indent=1)
