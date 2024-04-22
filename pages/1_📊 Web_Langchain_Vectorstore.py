import streamlit as st
from utils import st_def, tag_web

st_def.st_logo(title='👋RAG Web News!', page_title="RAG",)
# #-----------------------------------------------
tab1, tab2, tab3 = st.tabs(["🔰Prototype", "News", "Conclusion🏅"])

with tab2: tag_web.news()
with tab1:
    st.code("""
        # 5 lines only. No LLM, just vectorstore
        from langchain_community.document_loaders import WebBaseLoader
        from langchain.indexes import VectorstoreIndexCreator
        loader = WebBaseLoader("https://www.promptingguide.ai/techniques/rag")
        index = VectorstoreIndexCreator().from_loaders([loader])
        index.query("Why RAG?")
        """)

    # with st.spinner('RAGing...'):
    #     from langchain_community.document_loaders import WebBaseLoader
    #     from langchain.indexes import VectorstoreIndexCreator
    #     loader = WebBaseLoader("https://www.promptingguide.ai/techniques/rag")
    #     index = VectorstoreIndexCreator().from_loaders([loader])
    #     st.markdown(index.query("Why RAG?"))