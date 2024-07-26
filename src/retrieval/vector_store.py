import chromadb
from chromadb.config import Settings
from typing import List, Tuple, Dict
import numpy as np
from functools import lru_cache

class VectorStore:
    def __init__(self, collection_name: str = "excel_data", persist_directory: str = "./chroma_db"):
        self.client = chromadb.Client(Settings(persist_directory=persist_directory, anonymized_telemetry=False))
        self.collection = self.client.get_or_create_collection(name=collection_name)

    def add_embeddings(self, embeddings: Dict[str, np.ndarray], texts: Dict[str, List[Tuple[str, str]]]):
        for sheet_name, sheet_embeddings in embeddings.items():
            sheet_texts = texts[sheet_name]
            ids = [f"{sheet_name}_{i}" for i in range(len(sheet_texts))]
            documents = [text for _, text in sheet_texts]
            metadatas = [{"source": id, "sheet": sheet_name} for id, _ in sheet_texts]
            self.collection.add(
                embeddings=sheet_embeddings.tolist(),
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )

    @lru_cache(maxsize=100)
    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[Tuple[str, str, float, str]]:
        results = self.collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=k
        )
        return [
            (
                results['metadatas'][0][i]['source'],
                results['documents'][0][i],
                results['distances'][0][i],
                results['metadatas'][0][i]['sheet']
            )
            for i in range(len(results['ids'][0]))
        ]

    def clear(self):
        all_ids = self.collection.get()['ids']
        if all_ids:
            self.collection.delete(ids=all_ids)