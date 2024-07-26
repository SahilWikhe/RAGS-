import os
from typing import List, Tuple
import pandas as pd
from langchain.document_loaders import DataFrameLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.text_splitter import CharacterTextSplitter
from functools import lru_cache

class RAGSystem:
    def __init__(self, cache_dir: str = "./cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.vector_store = None
        self.qa_chain = None

    def process_excel_file(self, file_path: str):
        dataframes = self._load_excel(file_path)
        documents = self._process_dataframes(dataframes)
        self.vector_store = self._create_vector_store(documents)
        
        llm = Ollama(model="llama3:8b")
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=self.vector_store.as_retriever(),
            return_source_documents=True
        )

    @lru_cache(maxsize=100)
    def answer_question(self, query: str) -> Tuple[str, List[str]]:
        if not self.qa_chain:
            raise ValueError("Excel file has not been processed. Call process_excel_file first.")
        
        result = self.qa_chain.invoke({"query": query})
        answer = result['result']
        sources = [doc.page_content for doc in result['source_documents']]
        return answer, sources

    def _load_excel(self, file_path: str) -> List[pd.DataFrame]:
        xls = pd.ExcelFile(file_path)
        return [pd.read_excel(xls, sheet_name=sheet, usecols=lambda x: x is not None) for sheet in xls.sheet_names]

    def _process_dataframes(self, dataframes: List[pd.DataFrame]) -> List[str]:
        documents = []
        for i, df in enumerate(dataframes):
            df['content'] = df.astype(str).agg(' '.join, axis=1)
            loader = DataFrameLoader(df, page_content_column="content")
            docs = loader.load()
            for doc in docs:
                doc.page_content = f"Sheet {i + 1}: {doc.page_content}"
            documents.extend(docs)
        return documents

    def _create_vector_store(self, documents: List[str]) -> FAISS:
        text_splitter = CharacterTextSplitter(chunk_size=1500, chunk_overlap=150)
        texts = text_splitter.split_documents(documents)
        embeddings = HuggingFaceEmbeddings()
        return FAISS.from_documents(texts, embeddings, index_params={"M": 16, "efConstruction": 200})