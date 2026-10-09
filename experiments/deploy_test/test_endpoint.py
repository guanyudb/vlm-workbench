import base64, json, os, sys
from databricks.sdk import WorkspaceClient

# Usage: python test_endpoint.py <endpoint-name> <volume-frame-path> [profile]
#   e.g. python test_endpoint.py vlmwb-test-ft-qwen3vl \
#          /Volumes/<catalog>/<schema>/<volume>/extracted_frames/<video>/candidates/<frame>.jpg
if len(sys.argv) < 3:
    sys.exit("usage: test_endpoint.py <endpoint-name> <volume-frame-path> [profile]")
name, frame = sys.argv[1], sys.argv[2]
w = WorkspaceClient(profile=sys.argv[3]) if len(sys.argv) > 3 else WorkspaceClient()
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
