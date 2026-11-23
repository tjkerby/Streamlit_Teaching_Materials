from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
st.title("Explore conference messages")
st.write("Choose the earliest year to include, then read the matching messages.")
df = pd.read_csv(ROOT / "data/talks.csv")
year = st.select_slider("Earliest conference year", options=[2020, 2023, 2025], value=2020)
shown = df.loc[df["year"] >= year]
st.metric("Talks displayed", f"{len(shown)} / {len(df)}")
st.bar_chart(shown.groupby("year").size().rename("Talks"))
st.dataframe(shown[["title", "speaker", "year", "kicker"]], hide_index=True)
st.caption("Selected teaching sample. The counts describe these records, not changes across the full archive.")
