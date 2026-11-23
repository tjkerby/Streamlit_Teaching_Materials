"""Completed Day 2 app. Run from the project: uv run streamlit run app.py"""
from pathlib import Path

import numpy as np
import plotly.express as px
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity

from pipeline import compute_map, load_inputs

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title="Conference Message Explorer", layout="wide")
st.title("Conference Message Explorer")
st.write("Explore highlighted messages, compare speakers and years, and return to the source talks.")
st.caption("52 selected kickers from six conferences in 2020, 2023, and 2025. Each point represents a short highlighted message, rather than an entire talk.")

embedding_path = ROOT / "data/talk_embeddings.npz"
if not embedding_path.exists():
    st.error("The saved embeddings are missing. Follow the preparation steps in README.md before running the map.")
    st.code("uv run python scripts/prepare_data.py --from-ollama", language="bash")
    st.stop()

try:
    # Reading these small files makes their CONTENTS part of the cache key.
    df, X = load_inputs((ROOT / "data/talks.csv").read_bytes(), embedding_path.read_bytes())
except (ValueError, OSError, KeyError) as exc:
    st.error(str(exc))
    st.stop()

coordinates = compute_map(X)
df["PHATE1"] = coordinates[:, 0]
df["PHATE2"] = coordinates[:, 1]

with st.sidebar:
    st.header("Explore the sample")
    years = st.multiselect("Conference years", sorted(df["year"].unique()), default=sorted(df["year"].unique()))
    roles = st.multiselect("Recorded speaker roles", sorted(df["role_group"].unique()), default=sorted(df["role_group"].unique()))
    speakers = st.multiselect("Speakers (empty means all)", sorted(df["speaker"].unique()))
    text = st.text_input("Words in a title or kicker", help="Literal text match, ignoring capitalization. This is not semantic search.")
    color_label = st.selectbox("Color the map by", ["Recorded speaker role", "Conference year"])

keep = df["year"].isin(years) & df["role_group"].isin(roles)
if speakers:
    keep &= df["speaker"].isin(speakers)
if text.strip():
    keep &= (df["title"] + " " + df["kicker"]).str.contains(text.strip(), case=False, regex=False, na=False)
shown = df.loc[keep].copy()

col1, col2, col3 = st.columns(3)
col1.metric("Talks displayed", f"{len(shown)} / {len(df)}")
col2.metric("Speakers represented", shown["speaker"].nunique())
col3.metric("Conferences represented", shown["conference"].nunique())
st.caption("All summaries describe the current filters. Clearing all years or roles selects no talks. An empty speaker selection includes all speakers.")

if shown.empty:
    st.info("No talks match these filters. Select a year and role, or broaden the text or speaker filter.")
    st.stop()

color_column = "role_group" if color_label == "Recorded speaker role" else "conference_year"
df["conference_year"] = df["year"].astype(str)
shown["conference_year"] = shown["year"].astype(str)
categories = sorted(df[color_column].unique())
palette = px.colors.qualitative.Safe
colors = {label: palette[i % len(palette)] for i, label in enumerate(categories)}
fig = px.scatter(
    shown, x="PHATE1", y="PHATE2", color=color_column,
    color_discrete_map=colors, category_orders={color_column: categories},
    hover_name="title", hover_data={"speaker": True, "conference": True, "kicker": True,
                                    "PHATE1": False, "PHATE2": False, color_column: False},
    labels={"role_group": "Recorded speaker role", "conference_year": "Conference year"},
)
# Keep the full map's frame when filtering. No subset refit occurs.
def axis_range(values):
    low, high = float(values.min()), float(values.max())
    pad = max((high - low) * 0.08, 0.1)
    return [low - pad, high + pad]

fig.update_layout(height=490, margin=dict(l=20, r=20, t=20, b=20))
fig.update_xaxes(range=axis_range(df["PHATE1"]))
fig.update_yaxes(range=axis_range(df["PHATE2"]))
st.plotly_chart(fig, width="stretch")
st.caption("Filtering keeps the full-sample coordinates, axis ranges, and color assignments. Nearby points suggest messages to read. PHATE axes have no topic names.")

st.subheader("Read a message")
selected_id = st.selectbox(
    "Choose a displayed talk",
    shown["talk_id"].tolist(),
    format_func=lambda talk_id: df.loc[df["talk_id"].eq(talk_id), "title"].iloc[0]
    + " (" + df.loc[df["talk_id"].eq(talk_id), "speaker"].iloc[0] + ")",
)
selected_index = int(df.index[df["talk_id"].eq(selected_id)][0])
selected = df.loc[selected_index]
st.markdown(f"**{selected['title']}**")
st.write(selected["kicker"])
st.caption(f"{selected['speaker']} · {selected['conference']}")
st.link_button("Read the original talk", selected["url"])

with st.expander("Similar messages among the displayed talks"):
    st.write("Rank messages using cosine similarity of the original Nomic embeddings. The selected talk is excluded.")
    if len(shown) < 2:
        st.info("Display at least two talks to compare messages.")
    else:
        available = len(shown) - 1
        n = 1 if available == 1 else st.slider("Number of similar messages", 1, min(10, available), min(5, available))
        scores = cosine_similarity(X[selected_index:selected_index + 1], X)[0]
        matches = shown.loc[shown["talk_id"].ne(selected_id)].copy()
        matches["cosine_similarity"] = scores[matches.index.to_numpy()]
        st.dataframe(matches.nlargest(n, "cosine_similarity")[["title", "speaker", "kicker", "cosine_similarity"]], hide_index=True)
        st.caption("A similarity score is a model-based comparison, not a probability of identical meaning. Read the messages before describing a theme.")

with st.expander("Supporting records and download"):
    columns = ["title", "speaker", "conference", "role_group", "kicker", "url"]
    st.dataframe(shown[columns], hide_index=True)
    st.download_button("Download displayed records", shown[columns].to_csv(index=False).encode("utf-8"),
                       file_name="displayed_conference_messages.csv", mime="text/csv")

with st.expander("Sources and interpretation"):
    st.write("This purposive teaching sample contains 52 short kickers. It cannot estimate historical trends or represent every theme in the full talks. Recorded speaker roles describe the source pages.")
    st.write("Nomic v1.5 represents the kicker text using the clustering prefix. PHATE fits the full embedding matrix with random_state=386 and its remaining defaults. The map compresses the vectors and can distort relationships.")
    st.write("The saved embeddings preserve analysis results across restarts. Streamlit caching reuses loaded data and the PHATE computation while the app runs.")
