import streamlit as st

print("SCRIPT START", flush=True)
st.title("Predict the next run")
st.write("Watch both the browser and the terminal.")
count = 0
name = st.text_input("Your name", value="Ada")
print("The widget returned:", name, flush=True)
if st.button("Add one"):
    count += 1
st.write("Ordinary variable:", count)
st.caption("Click Add one twice. Then edit the name and press Enter. Explain each displayed count.")

with st.expander("A separate counter that remembers"):
    if "remembered_count" not in st.session_state:
        st.session_state.remembered_count = 0
    if st.button("Add one and remember"):
        st.session_state.remembered_count += 1
    st.write("Session State:", st.session_state.remembered_count)
st.caption("Session State belongs to this browser session. A new browser tab starts a separate session.")
