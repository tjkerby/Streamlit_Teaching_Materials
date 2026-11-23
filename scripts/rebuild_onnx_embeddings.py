"""Reproduce the supplied genuine Nomic vectors without an Ollama installation.

Instructor preparation only: uv run --group preparation python scripts/rebuild_onnx_embeddings.py
Downloads the official model weights into an OS temporary cache, outside the repo.
"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np
import onnxruntime as ort
import pandas as pd
import requests
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[1]
REPO = "nomic-ai/nomic-embed-text-v1.5"
# Pin the actual model revision in data/embedding_provenance.json after first preparation.
prior = ROOT / "data/embedding_provenance.json"
metadata = json.loads(prior.read_text()) if prior.exists() else {}
revision = metadata.get("hf_revision")
if not revision:
    response = requests.get(f"https://huggingface.co/api/models/{REPO}", timeout=90)
    response.raise_for_status()
    revision = response.json()["sha"]
cache = Path(tempfile.gettempdir()) / "stat386-nomic" / revision
cache.mkdir(parents=True, exist_ok=True)
for filename in ["tokenizer.json", "onnx/model_quantized.onnx"]:
    target = cache / Path(filename).name
    if not target.exists():
        response = requests.get(f"https://huggingface.co/{REPO}/resolve/{revision}/{filename}", stream=True, timeout=180)
        response.raise_for_status()
        temporary = target.with_suffix(".partial")
        with temporary.open("wb") as handle:
            for chunk in response.iter_content(1024 * 1024):
                handle.write(chunk)
        temporary.replace(target)

tokenizer = Tokenizer.from_file(str(cache / "tokenizer.json"))
tokenizer.enable_padding()
options = ort.SessionOptions()
options.intra_op_num_threads = 4
session = ort.InferenceSession(str(cache / "model_quantized.onnx"), sess_options=options, providers=["CPUExecutionProvider"])
df = pd.read_csv(ROOT / "data/talks.csv")
vectors = []
for start in range(0, len(df), 8):
    texts = ["clustering: " + text for text in df["kicker"].iloc[start:start + 8]]
    encoded = tokenizer.encode_batch(texts)
    inputs = {"input_ids": np.asarray([x.ids for x in encoded], dtype=np.int64),
              "attention_mask": np.asarray([x.attention_mask for x in encoded], dtype=np.int64),
              "token_type_ids": np.asarray([x.type_ids for x in encoded], dtype=np.int64)}
    feed = {x.name: inputs[x.name] for x in session.get_inputs()}
    hidden = session.run(None, feed)[0]
    mask = inputs["attention_mask"][..., None].astype(np.float32)
    pooled = (hidden * mask).sum(axis=1) / mask.sum(axis=1).clip(min=1)
    # Nomic's documented mean pooling, layer normalization, and L2 normalization.
    pooled = (pooled - pooled.mean(axis=1, keepdims=True)) / np.sqrt(pooled.var(axis=1, keepdims=True) + 1e-5)
    pooled /= np.linalg.norm(pooled, axis=1, keepdims=True)
    vectors.append(pooled.astype(np.float32))
    print(f"Embedded {min(start + 8, len(df))}/{len(df)}", flush=True)
X = np.vstack(vectors)
assert X.shape == (len(df), 768) and np.isfinite(X).all()
output = ROOT / "data/talk_embeddings.npz"
np.savez_compressed(output, embeddings=X, talk_id=df["talk_id"].to_numpy(dtype=str),
                    kicker=df["kicker"].to_numpy(dtype=str), model=np.array("nomic-embed-text:v1.5"),
                    prefix=np.array("clustering: "), backend=np.array("Official Nomic quantized ONNX"))
provenance = {"model": "nomic-embed-text:v1.5", "prefix": "clustering: ",
              "backend": "Official Nomic quantized ONNX with ONNX Runtime CPU",
              "hf_repository": REPO, "hf_revision": revision,
              "model_file": "onnx/model_quantized.onnx",
              "model_sha256": hashlib.sha256((cache / "model_quantized.onnx").read_bytes()).hexdigest(),
              "tokenizer_sha256": hashlib.sha256((cache / "tokenizer.json").read_bytes()).hexdigest(),
              "postprocessing": "attention-masked mean pooling; layer norm eps=1e-5; L2 norm; all 768 coordinates",
              "csv_sha256": hashlib.sha256((ROOT / "data/talks.csv").read_bytes()).hexdigest(),
              "embedding_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
              "records": len(df), "created_utc": datetime.now(timezone.utc).isoformat(),
              "onnxruntime_version": ort.__version__,
              "note": "Genuine model inference. Backend and quantization differ from Ollama, so numerical results and PHATE layouts can differ."}
prior.write_text(json.dumps(provenance, indent=2) + "\n")
print("Saved genuine model embeddings and provenance.")
