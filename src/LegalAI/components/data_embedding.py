from urllib import request
from zipfile import ZipFile
from LegalAI.logging import logger
from LegalAI.utils.common import get_size
from pathlib import Path
from LegalAI.entity import DataEmbeddingConfig

import os
from pinecone import Pinecone
from llama_index.llms.gemini import Gemini
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.core import StorageContext, VectorStoreIndex, download_loader

from llama_index.core import Settings
from llama_index.core import Document

class DataEmbedder:
    def __init__(self,
                 config:DataEmbeddingConfig):
        logger.info('Loading embedding model')
        GOOGLE_API_KEY = "AIzaSyDLGZxq4w1C3cIehsfBmK7K_T1jxTkGefk"
        PINECONE_API_KEY = "6b699f4f-5b1d-471a-adbe-918747981c1b"

        os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
        os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
        self.llm = Gemini()
        self.config=config
        self.embed_model = GeminiEmbedding(model_name=self.config.model_name)
        self.pinecone_client = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
        self.pinecone_index = self.pinecone_client.Index(self.config.index_name)
        
        Settings.llm = self.llm
        Settings.embed_model = self.embed_model
        Settings.chunk_size = 512
        logger.info('Successfully upserted data to Pinecone')
        
    def push(self,text):
        documents = [Document(text=t) for t in text]
        vector_store = PineconeVectorStore(pinecone_index=self.pinecone_index)
        storage_context = StorageContext.from_defaults(
            vector_store=vector_store
        )
        
        index = VectorStoreIndex.from_documents(
            documents, 
            storage_context=storage_context
        )
        
        index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
        
        self.query_engine = index.as_query_engine()
        
    

