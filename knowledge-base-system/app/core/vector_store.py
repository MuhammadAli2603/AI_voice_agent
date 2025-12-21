"""
Vector store abstraction with FAISS implementation.
Provides efficient similarity search with metadata filtering.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import faiss
import pickle
from pathlib import Path
import json

from app.models.schemas import DocumentChunk
from app.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


class VectorStore(ABC):
    """Abstract base class for vector storage backends."""

    @abstractmethod
    def add_documents(
        self,
        chunks: List[DocumentChunk],
        embeddings: np.ndarray
    ) -> None:
        """
        Add documents with their embeddings to the store.

        Args:
            chunks: List of document chunks
            embeddings: Corresponding embeddings
        """
        pass

    @abstractmethod
    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[DocumentChunk, float]]:
        """
        Search for similar documents.

        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            filters: Optional metadata filters

        Returns:
            List of (chunk, similarity_score) tuples
        """
        pass

    @abstractmethod
    def delete_by_company(self, company_id: str) -> int:
        """
        Delete all documents for a company.

        Args:
            company_id: Company identifier

        Returns:
            Number of documents deleted
        """
        pass

    @abstractmethod
    def save_index(self, path: str) -> None:
        """
        Save the index to disk.

        Args:
            path: Path to save the index
        """
        pass

    @abstractmethod
    def load_index(self, path: str) -> None:
        """
        Load the index from disk.

        Args:
            path: Path to load the index from
        """
        pass

    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the vector store.

        Returns:
            Dictionary with statistics
        """
        pass


class FAISSVectorStore(VectorStore):
    """
    FAISS-based vector store implementation.

    Uses FAISS IndexFlatIP for exact similarity search with inner product.
    Maintains separate metadata store for filtering.
    """

    def __init__(self, dimension: int):
        """
        Initialize FAISS vector store.

        Args:
            dimension: Embedding dimension
        """
        self.dimension = dimension

        # Initialize FAISS index (using inner product for normalized vectors)
        self.index = faiss.IndexFlatIP(dimension)

        # Store chunks and metadata separately
        self.chunks: List[DocumentChunk] = []
        self.chunk_metadata: List[Dict[str, Any]] = []

        # Mapping from chunk_id to index position
        self.chunk_id_to_idx: Dict[str, int] = {}

        logger.info(f"FAISSVectorStore initialized with dimension {dimension}")

    def add_documents(
        self,
        chunks: List[DocumentChunk],
        embeddings: np.ndarray
    ) -> None:
        """
        Add documents with embeddings to the FAISS index.

        Args:
            chunks: List of document chunks
            embeddings: Numpy array of embeddings (n_chunks, dimension)
        """
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Mismatch between chunks ({len(chunks)}) "
                f"and embeddings ({len(embeddings)})"
            )

        if len(chunks) == 0:
            logger.warning("Attempted to add empty chunk list")
            return

        # Validate embedding dimension
        if embeddings.shape[1] != self.dimension:
            raise ValueError(
                f"Embedding dimension ({embeddings.shape[1]}) "
                f"does not match index dimension ({self.dimension})"
            )

        # Convert to float32 (FAISS requirement)
        embeddings = embeddings.astype('float32')

        # Normalize embeddings for cosine similarity via inner product
        faiss.normalize_L2(embeddings)

        # Add to FAISS index
        self.index.add(embeddings)

        # Store chunks and metadata
        start_idx = len(self.chunks)
        for i, chunk in enumerate(chunks):
            idx = start_idx + i
            self.chunks.append(chunk)
            self.chunk_metadata.append(chunk.metadata.dict())
            self.chunk_id_to_idx[chunk.chunk_id] = idx

        logger.info(f"Added {len(chunks)} documents to index")

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[DocumentChunk, float]]:
        """
        Search for similar documents with optional filtering.

        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            filters: Optional filters (e.g., {'company_id': 'techstore'})

        Returns:
            List of (chunk, similarity_score) tuples
        """
        if self.index.ntotal == 0:
            logger.warning("Search called on empty index")
            return []

        # Normalize query embedding
        query_embedding = query_embedding.astype('float32').reshape(1, -1)
        faiss.normalize_L2(query_embedding)

        # Determine how many results to fetch
        # Fetch more if we need to filter
        fetch_k = top_k * 10 if filters else top_k

        # Search FAISS index
        distances, indices = self.index.search(query_embedding, min(fetch_k, self.index.ntotal))

        # Convert distances to similarity scores (FAISS returns inner products)
        distances = distances[0]
        indices = indices[0]

        # Filter results based on metadata
        results = []
        for idx, distance in zip(indices, distances):
            if idx == -1:  # FAISS returns -1 for missing results
                continue

            chunk = self.chunks[idx]
            metadata = self.chunk_metadata[idx]

            # Apply filters
            if filters:
                if not self._matches_filters(metadata, filters):
                    continue

            # Convert distance to similarity score (already cosine similarity)
            similarity_score = float(distance)

            results.append((chunk, similarity_score))

            # Stop if we have enough results
            if len(results) >= top_k:
                break

        return results

    def delete_by_company(self, company_id: str) -> int:
        """
        Delete all documents for a company.

        Note: FAISS doesn't support deletion, so we need to rebuild the index.

        Args:
            company_id: Company identifier

        Returns:
            Number of documents deleted
        """
        # Find indices to keep
        indices_to_keep = []
        chunks_to_keep = []
        metadata_to_keep = []

        for idx, metadata in enumerate(self.chunk_metadata):
            if metadata.get('company_id') != company_id:
                indices_to_keep.append(idx)
                chunks_to_keep.append(self.chunks[idx])
                metadata_to_keep.append(metadata)

        deleted_count = len(self.chunks) - len(chunks_to_keep)

        if deleted_count == 0:
            logger.info(f"No documents found for company {company_id}")
            return 0

        # Rebuild index with remaining documents
        logger.info(f"Rebuilding index after deleting {deleted_count} documents")

        # Create new index
        self.index = faiss.IndexFlatIP(self.dimension)
        self.chunks = []
        self.chunk_metadata = []
        self.chunk_id_to_idx = {}

        # Re-add remaining documents
        if chunks_to_keep:
            # Get embeddings for remaining chunks (stored in chunks)
            embeddings = []
            for chunk in chunks_to_keep:
                if hasattr(chunk, 'embedding') and chunk.embedding is not None:
                    embeddings.append(chunk.embedding)
                else:
                    # If no embedding stored, use zero vector (shouldn't happen)
                    logger.warning(f"Chunk {chunk.chunk_id} missing embedding")
                    embeddings.append(np.zeros(self.dimension))

            embeddings = np.array(embeddings, dtype='float32')
            self.add_documents(chunks_to_keep, embeddings)

        logger.info(f"Deleted {deleted_count} documents for company {company_id}")
        return deleted_count

    def save_index(self, path: str) -> None:
        """
        Save the FAISS index and metadata to disk.

        Args:
            path: Base path for saving (without extension)
        """
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)

        # Save FAISS index
        index_path = str(path_obj) + ".faiss"
        faiss.write_index(self.index, index_path)

        # Save metadata and chunks
        metadata_path = str(path_obj) + "_metadata.pkl"
        with open(metadata_path, 'wb') as f:
            pickle.dump({
                'chunks': self.chunks,
                'chunk_metadata': self.chunk_metadata,
                'chunk_id_to_idx': self.chunk_id_to_idx
            }, f)

        logger.info(f"Index saved to {index_path}")

    def load_index(self, path: str) -> None:
        """
        Load the FAISS index and metadata from disk.

        Args:
            path: Base path for loading (without extension)
        """
        path_obj = Path(path)

        # Load FAISS index
        index_path = str(path_obj) + ".faiss"
        if not Path(index_path).exists():
            raise FileNotFoundError(f"Index file not found: {index_path}")

        self.index = faiss.read_index(index_path)

        # Load metadata and chunks
        metadata_path = str(path_obj) + "_metadata.pkl"
        if not Path(metadata_path).exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

        with open(metadata_path, 'rb') as f:
            data = pickle.load(f)
            self.chunks = data['chunks']
            self.chunk_metadata = data['chunk_metadata']
            self.chunk_id_to_idx = data['chunk_id_to_idx']

        logger.info(f"Index loaded from {index_path} with {self.index.ntotal} vectors")

    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the vector store.

        Returns:
            Dictionary with statistics
        """
        company_counts = {}
        for metadata in self.chunk_metadata:
            company_id = metadata.get('company_id', 'unknown')
            company_counts[company_id] = company_counts.get(company_id, 0) + 1

        return {
            'total_vectors': self.index.ntotal,
            'dimension': self.dimension,
            'total_chunks': len(self.chunks),
            'companies': company_counts,
            'index_type': 'FAISS IndexFlatIP'
        }

    def _matches_filters(
        self,
        metadata: Dict[str, Any],
        filters: Dict[str, Any]
    ) -> bool:
        """
        Check if metadata matches all filters.

        Args:
            metadata: Chunk metadata
            filters: Filter criteria

        Returns:
            True if all filters match
        """
        for key, value in filters.items():
            if metadata.get(key) != value:
                return False
        return True

    def get_chunk_by_id(self, chunk_id: str) -> Optional[DocumentChunk]:
        """
        Get a chunk by its ID.

        Args:
            chunk_id: Chunk identifier

        Returns:
            DocumentChunk if found, None otherwise
        """
        idx = self.chunk_id_to_idx.get(chunk_id)
        if idx is not None:
            return self.chunks[idx]
        return None
