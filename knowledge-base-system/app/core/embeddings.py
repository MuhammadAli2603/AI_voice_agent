"""
Embedding service using sentence-transformers for text vectorization.
Provides efficient batch processing and caching capabilities.
"""

import numpy as np
from typing import List, Optional, Dict
import torch
from sentence_transformers import SentenceTransformer
from functools import lru_cache
import hashlib

from app.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


class EmbeddingService:
    """
    Service for generating text embeddings using sentence-transformers.

    Features:
    - Automatic GPU detection and usage
    - Batch processing for efficiency
    - Embedding caching for repeated texts
    - Progress tracking for large batches
    """

    def __init__(self, model_name: Optional[str] = None):
        """
        Initialize the embedding service.

        Args:
            model_name: Name of the sentence-transformers model.
                       Defaults to config setting.
        """
        self.model_name = model_name or settings.embedding_model_name
        self.embedding_dimension = settings.embedding_dimension
        self.batch_size = settings.embedding_batch_size

        # Detect device (GPU or CPU)
        self.device = self._detect_device()

        logger.info(f"Initializing embedding model: {self.model_name}")
        logger.info(f"Using device: {self.device}")

        try:
            # Load the model
            self.model = SentenceTransformer(self.model_name, device=self.device)
            actual_dimension = self.model.get_sentence_embedding_dimension()

            if actual_dimension != self.embedding_dimension:
                logger.warning(
                    f"Model dimension ({actual_dimension}) differs from "
                    f"configured dimension ({self.embedding_dimension}). "
                    f"Using model dimension."
                )
                self.embedding_dimension = actual_dimension

            logger.info(
                f"Embedding model loaded successfully - "
                f"Dimension: {self.embedding_dimension}"
            )

        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise

        # Cache for embeddings (stores hash -> embedding mapping)
        self._embedding_cache: Dict[str, np.ndarray] = {}
        self._cache_hits = 0
        self._cache_misses = 0

    def _detect_device(self) -> str:
        """
        Detect available device (CUDA GPU or CPU).

        Returns:
            Device string ('cuda' or 'cpu')
        """
        if settings.gpu_enabled and torch.cuda.is_available():
            return 'cuda'
        return 'cpu'

    def encode_text(self, text: str, use_cache: bool = True) -> np.ndarray:
        """
        Encode a single text into an embedding vector.

        Args:
            text: Text to encode
            use_cache: Whether to use cached embeddings

        Returns:
            Embedding vector as numpy array
        """
        if not text or not text.strip():
            logger.warning("Attempted to encode empty text")
            return np.zeros(self.embedding_dimension, dtype=np.float32)

        # Check cache
        if use_cache:
            text_hash = self._hash_text(text)
            if text_hash in self._embedding_cache:
                self._cache_hits += 1
                return self._embedding_cache[text_hash]
            self._cache_misses += 1

        try:
            # Generate embedding
            embedding = self.model.encode(
                text,
                convert_to_numpy=True,
                show_progress_bar=False,
                normalize_embeddings=True  # L2 normalization for better similarity
            )

            # Cache the embedding
            if use_cache:
                text_hash = self._hash_text(text)
                self._embedding_cache[text_hash] = embedding

            return embedding

        except Exception as e:
            logger.error(f"Failed to encode text: {e}")
            # Return zero vector as fallback
            return np.zeros(self.embedding_dimension, dtype=np.float32)

    def encode_batch(
        self,
        texts: List[str],
        batch_size: Optional[int] = None,
        show_progress: bool = False
    ) -> np.ndarray:
        """
        Encode a batch of texts into embedding vectors.

        Args:
            texts: List of texts to encode
            batch_size: Batch size for processing (default from config)
            show_progress: Whether to show progress bar

        Returns:
            Array of embeddings with shape (len(texts), embedding_dimension)
        """
        if not texts:
            logger.warning("Attempted to encode empty batch")
            return np.array([])

        batch_size = batch_size or self.batch_size

        # Filter out empty texts and track their indices
        valid_indices = []
        valid_texts = []
        for idx, text in enumerate(texts):
            if text and text.strip():
                valid_indices.append(idx)
                valid_texts.append(text)

        if not valid_texts:
            logger.warning("All texts in batch are empty")
            return np.zeros((len(texts), self.embedding_dimension), dtype=np.float32)

        try:
            logger.info(f"Encoding batch of {len(valid_texts)} texts")

            # Generate embeddings
            embeddings = self.model.encode(
                valid_texts,
                batch_size=batch_size,
                convert_to_numpy=True,
                show_progress_bar=show_progress,
                normalize_embeddings=True
            )

            # Create full embeddings array with zeros for invalid texts
            full_embeddings = np.zeros(
                (len(texts), self.embedding_dimension),
                dtype=np.float32
            )

            # Fill in valid embeddings
            for idx, embedding in zip(valid_indices, embeddings):
                full_embeddings[idx] = embedding

            logger.info(f"Successfully encoded {len(valid_texts)} texts")

            return full_embeddings

        except Exception as e:
            logger.error(f"Failed to encode batch: {e}")
            # Return zero vectors as fallback
            return np.zeros((len(texts), self.embedding_dimension), dtype=np.float32)

    def get_embedding_dimension(self) -> int:
        """
        Get the embedding dimension.

        Returns:
            Embedding dimension
        """
        return self.embedding_dimension

    def get_cache_stats(self) -> Dict[str, any]:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache stats
        """
        total_requests = self._cache_hits + self._cache_misses
        hit_rate = (
            self._cache_hits / total_requests
            if total_requests > 0
            else 0.0
        )

        return {
            'cache_size': len(self._embedding_cache),
            'cache_hits': self._cache_hits,
            'cache_misses': self._cache_misses,
            'hit_rate': hit_rate
        }

    def clear_cache(self):
        """Clear the embedding cache."""
        self._embedding_cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0
        logger.info("Embedding cache cleared")

    def _hash_text(self, text: str) -> str:
        """
        Generate a hash for text caching.

        Args:
            text: Text to hash

        Returns:
            Hash string
        """
        return hashlib.md5(text.encode('utf-8')).hexdigest()

    def warmup(self, sample_texts: Optional[List[str]] = None):
        """
        Warmup the model with sample texts for better first-query performance.

        Args:
            sample_texts: Optional list of sample texts. If None, uses defaults.
        """
        if sample_texts is None:
            sample_texts = [
                "How can I track my order?",
                "What is your return policy?",
                "I need help with my account."
            ]

        logger.info("Warming up embedding model...")
        self.encode_batch(sample_texts, show_progress=False)
        logger.info("Model warmup complete")
