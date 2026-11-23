"""Checkpoint 3: cache the input loading and full-data PHATE fit."""
from pathlib import Path

import numpy as np
import pandas as pd
import phate
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline import load_inputs, compute_map
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
st.title("Conference message map")
df, X = load_inputs((ROOT / "data/talks.csv").read_bytes(), (ROOT / "data/talk_embeddings.npz").read_bytes())
coordinates = compute_map(X)
df["PHATE1"] = coordinates[:, 0]
df["PHATE2"] = coordinates[:, 1]
years = st.multiselect("Conference years", [2020, 2023, 2025], default=[2020, 2023, 2025])
shown = df.loc[df["year"].isin(years)]
st.metric("Talks displayed", f"{len(shown)} / {len(df)}")
if shown.empty:
    st.info("Select a year to display messages.")
    st.stop()
fig = px.scatter(shown, x="PHATE1", y="PHATE2", color="role_group", hover_name="title", hover_data=["speaker", "kicker"])
fig.update_xaxes(range=[df["PHATE1"].min() - 0.1, df["PHATE1"].max() + 0.1])
fig.update_yaxes(range=[df["PHATE2"].min() - 0.1, df["PHATE2"].max() + 0.1])
st.plotly_chart(fig, width="stretch")
st.dataframe(shown[["title", "speaker", "year", "kicker"]], hide_index=True)
st.caption("52 selected kickers. Read the text before interpreting a pattern.")
