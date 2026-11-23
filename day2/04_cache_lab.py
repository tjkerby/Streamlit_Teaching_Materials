"""A deliberately slow teaching example. It does not generate embeddings."""
import time

import streamlit as st

st.title("What does the cache reuse?")
st.caption("This demonstration deliberately waits one second. The returned value is ordinary arithmetic, not model output.")


def slow_square(number):
    print("UNCACHED function body:", number, flush=True)
    time.sleep(1)
    return number ** 2


@st.cache_data
def cached_square(number):
    print("CACHED function body (cache miss):", number, flush=True)
    time.sleep(1)
    return number ** 2


number = st.slider("Number", 1, 5, 2)
use_cache = st.checkbox("Use the cached function", value=False)
st.selectbox("Unrelated display preference", ["Compact", "Detailed"])
if st.button("Clear this function's cache"):
    cached_square.clear()
start = time.perf_counter()
answer = cached_square(number) if use_cache else slow_square(number)
st.write("Answer:", answer)
st.write(f"Function call took {time.perf_counter() - start:.3f} seconds")
st.write("Try 2, then 3, then 2 again. Change only the unrelated preference. Predict the terminal output each time.")
