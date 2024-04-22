import streamlit as st

from utils import st_def, tab_rag
st_def.st_logo(title='👋Retrieval-Augmented Generation', page_title="RAG🍨")

tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs(
    ["🔰General", "Loading➡️", "Chunking➡️", "Embedding➡️", "Vector➡️", "Retrieval➡️", "Q&A➡️", "Evaluation🏅"])

with tab1:  tab_rag.rag_general()
with tab8:  tab_rag.rag_evaluation()


