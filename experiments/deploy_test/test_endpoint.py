import base64, json, os, sys
from databricks.sdk import WorkspaceClient

w = WorkspaceClient(profile="e2FE")
name = "vlmwb-test-ft-qwen3vl"
frame = "/Volumes/hls_lifesciences/vlmwb_guanyu_chen/medical_video/extracted_frames/VinayVideoThing1/VinayVideoThing1_frame_0012.31s.jpg"
prompt = "Identify the surgical instrument in this image. Respond with JSON like {\"instrument\": \"<name>\"}."

# Download via SDK files API to local
local = "/tmp/test_frame.jpg"
try:
    fr = w.files.download(frame)
    with open(local, "wb") as f:
        f.write(fr.contents.read())
    print(f"downloaded test frame to {local} size={os.path.getsize(local)}")
except Exception as e:
    print(f"download failed: {e}")
    sys.exit(1)

img_b64 = base64.b64encode(open(local, "rb").read()).decode()
resp = w.serving_endpoints.query(
    name=name,
    dataframe_records=[{"prompt": prompt, "image": img_b64}],
)
print("predictions:", resp.predictions)
