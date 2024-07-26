import numpy as np
from typing import List
from sentence_transformers import SentenceTransformer

class EmbeddingModel:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', batch_size: int = 32):
        self.model = SentenceTransformer(model_name)
        self.batch_size = batch_size

    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i+self.batch_size]
            batch_embeddings = self.model.encode(batch, show_progress_bar=False)
            embeddings.extend(batch_embeddings)
        return np.array(embeddings)