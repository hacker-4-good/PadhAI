import streamlit as st 
from web_content import *

st.set_page_config(layout="wide", page_title="PadhAI", page_icon=":books:")

st.markdown(hero_logo, unsafe_allow_html=True)

with st.sidebar:
    st.markdown(sidebar_logo, unsafe_allow_html=True)

st.page_link("pages/chatbot.py", label="Chatbot", icon="💬")
st.page_link("pages/quiz.py", label="Quiz", icon="❓")