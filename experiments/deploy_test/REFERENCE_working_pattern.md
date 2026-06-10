# Working register pattern (from `~/Documents/Projects/Cursor/vlm_finetune/02_register_base_model.ipynb`)

The user has a known-good Qwen3-VL register notebook in another project. The
`AttributeError: 'list' object has no attribute 'keys'` you've been hitting
is from an old transformers version; the working code below pins
`transformers>=4.57.0`. Use this pattern instead of iterating from scratch.

## Pip install block (cell 1)
```python
%pip install --quiet --upgrade pip
%pip install --quiet transformers>=4.57.0 accelerate>=0.26.0 torch>=2.1.0 torchvision>=0.16.0
%pip install --quiet pillow>=10.0.0 pyyaml>=6.0 qwen-vl-utils
%pip install --quiet "mlflow[databricks]>=3.1"
dbutils.library.restartPython()
```

## Critical idea — save checkpoint to local dir FIRST, pass the PATH

```python
import tempfile, transformers as _transformers_mod
from mlflow.models import ModelSignature
from mlflow.types.schema import Schema, ColSpec

save_dir = tempfile.mkdtemp(prefix='qwen3vl_base_')
model.save_pretrained(save_dir)
processor.save_pretrained(save_dir)

# Unity Catalog requires an explicit signature
signature = ModelSignature(
    inputs=Schema([ColSpec('string', 'prompt'), ColSpec('binary', 'image')]),
    outputs=Schema([ColSpec('string', 'generated_text')]),
)

# Strip CUDA build tag (e.g. "+cu126") from torch version
torch_version = torch.__version__.split('+')[0]

with mlflow.start_run(run_name="register_qwen3vl_base") as run:
    mlflow.log_params({
        'model_name': MODEL_NAME,
        'dtype': str(dtype),
        'type': 'base_model',
    })
    mlflow.transformers.log_model(
        transformers_model=save_dir,             # PATH, not dict
        name='model',
        registered_model_name=uc_model_name,
        task='image-text-to-text',               # literal string
        torch_dtype=dtype,
        signature=signature,
        metadata={'source_model': MODEL_NAME},
        pip_requirements=[
            f'transformers=={_transformers_mod.__version__}',
            f'torch=={torch_version}',
            'accelerate',
            'qwen-vl-utils',
            'Pillow',
        ],
    )
```

**Why save-then-pass-path**: passing a dict of `{"model": ..., "tokenizer": ...,
"image_processor": ...}` triggers MLflow's Pipeline-creation validation,
which fails for multimodal Processor models. Passing the on-disk path
sidesteps that path entirely.

## Reload smoke test

`mlflow.transformers.load_model` does NOT forward `trust_remote_code` — for
custom-arch models like Qwen3-VL you must load the checkpoint manually:

```python
import os, glob

local_dir = mlflow.artifacts.download_artifacts(f"models:/{uc_model_name}/1")
# MLflow may nest the checkpoint inside a subdir — find the config.json
hits = glob.glob(os.path.join(local_dir, '**/config.json'), recursive=True)
model_dir = os.path.dirname(hits[0]) if hits else local_dir

model = AutoModelForImageTextToText.from_pretrained(
    model_dir, device_map='auto', torch_dtype=dtype, trust_remote_code=True,
)
processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
model.eval()
```

## What to copy into `local_inference_notebooks/finetune/notebook.py`

The current `mlflow.transformers.log_model` call there is probably passing
the in-memory model+processor as a dict. Replace it with the
save-to-tempdir-then-pass-path pattern above, and pin the same
`pip_requirements` list.

For Gemma 4, swap the qwen-vl-utils dep for whatever Gemma 4's loader
needs (likely nothing extra — it's plain AutoModelForImageTextToText), and
pin `transformers>=5.5.0`.
