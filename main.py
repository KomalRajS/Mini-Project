import streamlit as st
import random
import time
from PyPDF2 import PdfReader

from LegalAI.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from LegalAI.pipeline.stage_02_data_embedding import DataEmbeddingPipeline
from LegalAI.logging import logger
STAGE='stage_02 Data embedding'

try:
    print(f'---------  {STAGE}  started ---------')
    data_embedding_pipeline = DataEmbeddingPipeline()
    data_embedding_pipeline.main()
    print(f'---------  {STAGE} completed  ---------')
except Exception as e:
    print(f'---------  {STAGE} failed  ---------')
    logger.exception(e)
    
    
    
    
def extract_text_from_pdf(uploaded_file):
    pdf_reader = PdfReader(uploaded_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text
def chunk_it(chunk_size,text):
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    return chunks


st.title("Simple chat")


if "messages" not in st.session_state:
    st.session_state.messages = []


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")


    
# Accept user input
if prompt := st.chat_input():
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    
    if uploaded_file:   
        docs=extract_text_from_pdf(uploaded_file)
        docs2=chunk_it(1000000,docs)
        data_embedding_pipeline.data_embedding.push(docs2)
        response=data_embedding_pipeline.data_embedding.query_engine.query(  prompt)
        with st.chat_message("assistant"):
            st.markdown(response)
        
    
    # Add user message to chat history
    