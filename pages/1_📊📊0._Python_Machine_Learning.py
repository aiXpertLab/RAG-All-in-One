import streamlit as st
from utils import st_def, st_ML
import openai, PyPDF2, os, time, pandas as pd

st_def.st_logo(title='👋 to Machine Learning!', page_title="Machine Learning",)
# contexts = st.session_state['news'] 
# st.write(contexts)

# pdf1 = st.file_uploader('Upload your PDF Document', type='pdf')
# #-----------------------------------------------
# if pdf1:
#     pdfReader = PyPDF2.PdfReader(pdf1)
#     st.session_state['pdfreader'] = pdfReader
#     st.success(" has loaded.")
# else:
#     st.info("waiting for loading ...")

agree = st.checkbox('Continue to ChatGPT?')

if agree:
    st.write('Great!')