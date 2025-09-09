import streamlit as st
import HW1, HW2

st.set_page_config(page_title="Document QA", layout="wide")

nav = st.navigation({
    "HW": [
        st.Page(H1.run, title="HW 1", url_path="HW-1"),
        st.Page(HW22.run, title="HW 2 (Default)", url_path="HW-2", default=True),
    ]
})

nav.run()