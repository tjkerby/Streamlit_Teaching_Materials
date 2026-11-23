"""Runnable starting point. Add the steps in demo.qmd during Day 2."""
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
st.title("Conference Message Explorer")
df = pd.read_csv(ROOT / "data/talks.csv")
st.caption("52 selected conference kickers. Each row contains one highlighted message.")

# TODO 1: Load the saved vectors, check alignment, fit PHATE, and display its map.
# See day2/01_map.py for the completed first step.

years = st.multiselect("Conference years", [2020, 2023, 2025], default=[2020, 2023, 2025])
shown = df.loc[df["year"].isin(years)]

# TODO 2: Show talks displayed, unique speakers, and conferences represented.
# TODO 3: Draw the selected rows using the FULL-SAMPLE coordinates and axis ranges.
# TODO 4: Extract the loading and PHATE functions, then cache their results.
# TODO 5: Add a readable kicker and its original source link.

st.dataframe(shown[["title", "speaker", "year", "kicker"]], hide_index=True)
