"""Checkpoint 1: display the familiar analysis with no caching."""
from pathlib import Path

import numpy as np
import pandas as pd
import phate
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
st.title("Conference message map")
df = pd.read_csv(ROOT / "data/talks.csv")
if not (ROOT / "data/talk_embeddings.npz").exists():
    st.error("Prepare data/talk_embeddings.npz using README.md first.")
    st.stop()
with np.load(ROOT / "data/talk_embeddings.npz", allow_pickle=False) as saved:
    assert np.array_equal(saved["talk_id"], df["talk_id"].to_numpy(dtype=str))
    assert np.array_equal(saved["kicker"], df["kicker"].to_numpy(dtype=str))
    X = saved["embeddings"]
print("FITTING PHATE", flush=True)
coordinates = phate.PHATE(random_state=386).fit_transform(X)
df["PHATE1"] = coordinates[:, 0]
df["PHATE2"] = coordinates[:, 1]
fig = px.scatter(df, x="PHATE1", y="PHATE2", color="role_group", hover_name="title", hover_data=["speaker", "kicker"])
st.plotly_chart(fig, width="stretch")
st.caption("52 selected kickers. Read the text before interpreting a pattern.")
