import logging
import os
from langchain.embeddings import Embeddings
from langchain_classic.retrievers import EnsembleRetriever
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_core.runnables import ConfigurableField
from langchain_ollama import OllamaEmbeddings
from langchain_community.retrievers import BM25Retriever

import ssl
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context
import nltk
nltk.download("punkt_tab")

from nltk.tokenize import word_tokenize


class PdfRagService:
    def __init__(self, 
                 pdf_directory: str,
                 chunk_size: int = 1500,
                 chunk_overlap: int = 150,
                 chroma_persist_directory: str = "data/db/chroma",
                 chroma_collection_name: str = "pdf_rag_db",
                 embeddings_model: Embeddings | None = None,
                 vectorstore: Chroma | None = None,
                 retriever: EnsembleRetriever | None = None,
                 k: int = 10):
        
        self.pdf_directory = pdf_directory
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.chroma_persist_directory = chroma_persist_directory
        self.chroma_collection_name = chroma_collection_name

        if embeddings_model is None:
            self.embeddings_model = OllamaEmbeddings(model="bge-m3:latest")
        else:
            self.embeddings_model = embeddings_model

        self.vectorstore = vectorstore
        self.retriever = retriever
        self.k = k

    def __load_pdfs__(self) -> list[Document]:
        loader = PyPDFDirectoryLoader(
            self.pdf_directory,
            mode="single")
        documents = loader.load()
        logging.info(f"Loaded {len(documents)} documents from PDFs.")
        return documents
    
    def __split_documents_texts__(self, documents: list[Document]) -> list[Document]:
        logging.info("Splitting documents into chunks...")
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            encoding_name="cl100k_base",
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", "? ", " ", ""]
        )
        split_docs = text_splitter.split_documents(documents)
        logging.info(f"Splited documents into {len(split_docs)} chunks.")
        return split_docs
    
    def __create_vector_store__(self, splited_docs: list[Document]) -> Chroma:
        persist_directory = self.chroma_persist_directory
        collection_name = self.chroma_collection_name

        if(os.path.exists(persist_directory) and os.listdir(persist_directory)):
            logging.info("Persisted vector store found. Loading existing vector store...")
            self.vectorstore = Chroma(
                collection_name=collection_name,
                persist_directory=persist_directory,
                embedding_function=self.embeddings_model
            )

            logging.info(f"Loaded existing vector store.")
            return self.vectorstore

        logging.info("No persisted vector store found. Creating new vector store...")
        self.vectorstore = Chroma.from_documents(
            documents=splited_docs,
            embedding=self.embeddings_model,
            collection_name=collection_name,
            persist_directory=persist_directory
        )

        logging.info(f"Created vector store.")
        return self.vectorstore
    
    def __configure_retriever__(self, splited_docs: list[Document]) -> EnsembleRetriever:
        bm25_retriever = BM25Retriever.from_documents(
            splited_docs,
            preprocess_func=word_tokenize)
        bm25_retriever.k = self.k

        if(self.vectorstore is None):
            raise ValueError("Vector store is not created. Please create the vector store before creating the retriever.")
        
        vectorstore_retriever = self.vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": self.k, "fetch_k": self.k * 2}
        ).configurable_fields(
            search_kwargs=ConfigurableField(
                id="search_kwargs",
                name="Search Kwargs",
                description="The search kwargs to use"
            )
        )

        self.retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, vectorstore_retriever], 
            weights=[0.2, 0.8])
        
        logging.info(f"Created ensemble retriever.")
        return self.retriever
    
    def create_retriever(self) -> EnsembleRetriever:
        documents = self.__load_pdfs__()
        splited_docs = self.__split_documents_texts__(documents)
        self.__create_vector_store__(splited_docs)
        retriever = self.__configure_retriever__(splited_docs)
        return retriever