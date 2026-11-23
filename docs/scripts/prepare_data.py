"""Rebuild or import genuine model outputs. Never substitutes random vectors."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
MODEL = "nomic-embed-text:v1.5"
PREFIX = "clustering: "


def validate(path, df):
    with np.load(path, allow_pickle=False) as saved:
        for key, expected in [("talk_id", df["talk_id"].to_numpy(dtype=str)),
                              ("kicker", df["kicker"].to_numpy(dtype=str))]:
            if not np.array_equal(saved[key], expected):
                raise ValueError(f"Saved {key} does not match the CSV in order and content.")
        if saved["model"].item() != MODEL or saved["prefix"].item() != PREFIX:
            raise ValueError("Saved model or task prefix differs from this lesson.")
        X = saved["embeddings"]
        if X.shape != (len(df), 768) or not np.isfinite(X).all() or np.any(np.linalg.norm(X, axis=1) == 0):
            raise ValueError("Expected finite, nonzero 768-dimensional vectors.")
    return X


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--from-ollama", action="store_true")
    source.add_argument("--import-embeddings", type=Path)
    args = parser.parse_args()
    df = pd.read_csv(ROOT / "data/talks.csv")
    output = ROOT / "data/talk_embeddings.npz"
    if args.import_embeddings:
        validate(args.import_embeddings, df)
        if args.import_embeddings.resolve() != output.resolve():
            shutil.copy2(args.import_embeddings, output)
        backend = "Imported from the instructor's previous lesson. Retain its original provenance."
    else:
        vectors = []
        for i, kicker in enumerate(df["kicker"], 1):
            response = requests.post("http://localhost:11434/api/embed",
                                     json={"model": MODEL, "input": PREFIX + kicker, "truncate": False}, timeout=180)
            response.raise_for_status()
            vectors.append(response.json()["embeddings"][0])
            print(f"Embedded {i}/{len(df)}", flush=True)
        np.savez_compressed(output, embeddings=np.asarray(vectors, dtype=np.float32),
                            talk_id=df["talk_id"].to_numpy(dtype=str), kicker=df["kicker"].to_numpy(dtype=str),
                            model=np.array(MODEL), prefix=np.array(PREFIX), backend=np.array("Ollama"))
        backend = "Ollama local /api/embed"
        validate(output, df)
    provenance = {"model": MODEL, "prefix": PREFIX, "backend": backend,
                  "created_utc": datetime.now(timezone.utc).isoformat(),
                  "csv_sha256": hashlib.sha256((ROOT / "data/talks.csv").read_bytes()).hexdigest(),
                  "embedding_sha256": hashlib.sha256(output.read_bytes()).hexdigest(), "records": len(df)}
    (ROOT / "data/embedding_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print("Saved and validated:", output)
    print("Restart the app. Its file-content cache keys will detect the new artifact.")


if __name__ == "__main__":
    main()
