"""Optional local demonstration. Requires a running Ollama server and Nomic."""
import requests
import streamlit as st

st.title("Cache one real embedding call")
st.caption("Local demonstration. Ollama must run on the same machine as this app's Python server.")


@st.cache_data(show_spinner="Requesting a Nomic embedding...", max_entries=100)
def embed_text(text, model, prefix):
    print("CACHE MISS: calling Ollama", flush=True)
    response = requests.post("http://localhost:11434/api/embed",
                             json={"model": model, "input": prefix + text, "truncate": False}, timeout=120)
    response.raise_for_status()
    return response.json()["embeddings"][0]


with st.form("embedding_request"):
    text = st.text_area("Message", "Helping a neighbor through a difficult time.")
    submitted = st.form_submit_button("Embed this message")
if submitted and text.strip():
    try:
        vector = embed_text(text.strip(), "nomic-embed-text:v1.5", "clustering: ")
        st.write("Vector length:", len(vector))
        st.write(vector[:8])
    except requests.RequestException:
        st.error("Start Ollama and run: ollama pull nomic-embed-text:v1.5")
st.caption("Submitting the same text can reuse the result. Cached return values and Ollama's loaded model are separate things.")
