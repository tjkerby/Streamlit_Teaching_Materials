"""Analysis reused by the final app. Inspect the short apps before this file."""
from io import BytesIO

import numpy as np
import pandas as pd
import phate
import streamlit as st


@st.cache_data(show_spinner="Loading talks and checking their saved embeddings...")
def load_inputs(csv_bytes: bytes, embedding_bytes: bytes):
    """The file contents are explicit inputs, so edited files get a new entry."""
    print("CACHE MISS: loading and validating inputs", flush=True)
    df = pd.read_csv(BytesIO(csv_bytes))
    with np.load(BytesIO(embedding_bytes), allow_pickle=False) as saved:
        required = {"embeddings", "talk_id", "kicker", "model", "prefix"}
        if not required.issubset(saved.files):
            raise ValueError("The embedding file is missing required metadata.")
        if not np.array_equal(saved["talk_id"], df["talk_id"].to_numpy(dtype=str)):
            raise ValueError("Talk IDs or row order differ. Prepare matching embeddings.")
        if not np.array_equal(saved["kicker"], df["kicker"].to_numpy(dtype=str)):
            raise ValueError("Kicker text changed. Prepare matching embeddings.")
        model, prefix = saved["model"].item(), saved["prefix"].item()
        if model != "nomic-embed-text:v1.5" or prefix != "clustering: ":
            raise ValueError("Expected Nomic v1.5 embeddings with the clustering prefix.")
        X = saved["embeddings"].copy()
    if df["talk_id"].duplicated().any():
        raise ValueError("Talk IDs must be unique.")
    if X.ndim != 2 or X.shape[0] != len(df) or X.shape[1] != 768:
        raise ValueError("Expected one 768-dimensional embedding per talk.")
    if not np.isfinite(X).all() or np.any(np.linalg.norm(X, axis=1) == 0):
        raise ValueError("Embeddings must be finite, nonzero vectors.")
    df["conference"] = df["year"].astype(str) + " " + df["month"].map({4: "April", 10: "October"})
    return df, X


@st.cache_data(show_spinner="Computing the full-sample PHATE map...", max_entries=4)
def compute_map(X, random_state=386):
    print("CACHE MISS: fitting PHATE", flush=True)
    return phate.PHATE(random_state=random_state).fit_transform(X)
