"""
Tests for the embedding service.
"""

import pytest
import numpy as np
from app.core.embeddings import EmbeddingService


@pytest.fixture
def embedding_service():
    """Create an embedding service instance."""
    return EmbeddingService()


def test_encode_single_text(embedding_service):
    """Test encoding a single text."""
    text = "This is a test sentence."
    embedding = embedding_service.encode_text(text)

    assert isinstance(embedding, np.ndarray)
    assert len(embedding) == embedding_service.get_embedding_dimension()
    assert embedding.dtype == np.float32


def test_encode_batch(embedding_service):
    """Test batch encoding."""
    texts = [
        "First sentence.",
        "Second sentence.",
        "Third sentence."
    ]

    embeddings = embedding_service.encode_batch(texts)

    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape == (3, embedding_service.get_embedding_dimension())


def test_empty_text_handling(embedding_service):
    """Test handling of empty text."""
    embedding = embedding_service.encode_text("")

    # Should return zero vector
    assert isinstance(embedding, np.ndarray)
    assert np.allclose(embedding, np.zeros(embedding_service.get_embedding_dimension()))


def test_cache_functionality(embedding_service):
    """Test embedding caching."""
    # Clear cache first
    embedding_service.clear_cache()

    text = "Test sentence for caching."

    # First call - cache miss
    embedding1 = embedding_service.encode_text(text, use_cache=True)
    stats1 = embedding_service.get_cache_stats()

    # Second call - cache hit
    embedding2 = embedding_service.encode_text(text, use_cache=True)
    stats2 = embedding_service.get_cache_stats()

    # Embeddings should be identical
    assert np.array_equal(embedding1, embedding2)

    # Cache hits should increase
    assert stats2['cache_hits'] > stats1['cache_hits']


def test_similarity_normalized(embedding_service):
    """Test that embeddings are normalized for cosine similarity."""
    text = "This is a test."
    embedding = embedding_service.encode_text(text)

    # L2 norm should be close to 1 (normalized)
    norm = np.linalg.norm(embedding)
    assert abs(norm - 1.0) < 0.01


def test_batch_with_empty_texts(embedding_service):
    """Test batch encoding with some empty texts."""
    texts = [
        "Valid text",
        "",
        "Another valid text"
    ]

    embeddings = embedding_service.encode_batch(texts)

    assert embeddings.shape[0] == 3
    # Empty text should have zero embedding
    assert np.allclose(embeddings[1], np.zeros(embedding_service.get_embedding_dimension()))
