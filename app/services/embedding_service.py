from sentence_transformers import SentenceTransformer
import torch
from typing import List
import numpy as np
from loguru import logger
from langfuse import observe
from ..core.langfuse_client import langfuse

class EmbeddingService:
    def __init__(self, model_name: str = 'bkai-foundation-models/vietnamese-bi-encoder'):
        """
        Initializes the embedding service with a specialized Vietnamese model.
        """
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model_name = model_name
        logger.info(f"EmbeddingService is using device: {self.device}")
        
        # Load the model onto the specified device
        self.model = SentenceTransformer(self.model_name, device=self.device)
        self.dimension = self.model.get_sentence_embedding_dimension()
        logger.info(f"Embedding model loaded. Dimension: {self.dimension}")

    @observe(name="embedding-generation")
    def encode(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Encodes a batch of texts into embeddings.
        This is observed as a 'generation' type span in Langfuse.
        """
        # Manually update the current span with model info, as it's not a direct kwarg
        langfuse.update_current_span(
            metadata={"model": self.model_name, "batch_size": batch_size, "device": self.device},
            input=texts
        )
        
        # The model expects a list of texts
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True  # Normalizing is crucial for cosine similarity
        )
        # We don't log the output here as it's a large vector.
        # The metadata is sufficient for tracing.
        return embeddings

# Create a singleton instance to be used across the application
embedding_service = EmbeddingService() 
 
 
 
 
 
 
 
 
 
 
 
 
 
 