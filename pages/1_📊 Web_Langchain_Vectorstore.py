import streamlit as st
from utils.st_def import st_logo, st_load_ML
import openai, PyPDF2, os, time, pandas as pd

st_logo(title='👋 RAG Web!', page_title="RAG",)
# #-----------------------------------------------
st.code("""
    from langchain_community.document_loaders import WebBaseLoader
    from langchain.indexes import VectorstoreIndexCreator
    loader = WebBaseLoader("https://www.promptingguide.ai/techniques/rag")
    index = VectorstoreIndexCreator().from_loaders([loader])
    index.query("Why RAG?")
    """)

with st.spinner('RAGing...'):
    from langchain_community.document_loaders import WebBaseLoader
    from langchain.indexes import VectorstoreIndexCreator
    loader = WebBaseLoader("https://www.promptingguide.ai/techniques/rag")
    index = VectorstoreIndexCreator().from_loaders([loader])
    st.markdown(index.query("Why RAG?"))