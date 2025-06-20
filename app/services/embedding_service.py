from sentence_transformers import SentenceTransformer
import torch
from typing import List
import numpy as np
from loguru import logger

class EmbeddingService:
    def __init__(self, model_name: str = 'bkai-foundation-models/vietnamese-bi-encoder'):
        """
        Initializes the embedding service with a specialized Vietnamese model.
        """
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"EmbeddingService is using device: {self.device}")
        
        # Load the model onto the specified device
        self.model = SentenceTransformer(model_name, device=self.device)
        self.dimension = self.model.get_sentence_embedding_dimension()
        logger.info(f"Embedding model loaded. Dimension: {self.dimension}")

    def encode(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Encodes a batch of texts into embeddings.
        """
        # The model expects a list of texts
        return self.model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True  # Normalizing is crucial for cosine similarity
        )

# Create a singleton instance to be used across the application
embedding_service = EmbeddingService() 
 
 
 
 
 
 
 
 
 
 
 
 
 
 