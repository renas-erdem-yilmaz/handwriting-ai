import streamlit as st
from .backend import Backend

@st.cache_resource(show_spinner=False)
def get_backend():
    # No checkpoints or GPU tensors are loaded here. Runner loads on explicit work.
    return Backend()
