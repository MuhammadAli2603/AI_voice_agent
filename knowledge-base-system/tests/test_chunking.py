"""
Tests for the chunking service.
"""

import pytest
from app.core.chunking import ChunkingService
from app.models.schemas import DocumentType, Priority


@pytest.fixture
def chunking_service():
    """Create a chunking service instance."""
    return ChunkingService()


def test_chunk_short_document(chunking_service):
    """Test chunking of a short document."""
    text = "This is a short document. It has only a few sentences. Not enough to split."
    metadata = {
        "company_id": "test",
        "document_id": "doc1",
        "document_type": DocumentType.GENERAL,
        "category": "test",
        "source_file": "test.txt",
        "priority": Priority.MEDIUM
    }

    chunks = chunking_service.chunk_document(text, metadata)

    assert len(chunks) == 1
    assert chunks[0].text == text
    assert chunks[0].metadata.company_id == "test"


def test_chunk_long_document(chunking_service):
    """Test chunking of a long document."""
    # Create a long document with many sentences
    sentences = [f"This is sentence number {i}." for i in range(100)]
    text = " ".join(sentences)

    metadata = {
        "company_id": "test",
        "document_id": "doc2",
        "document_type": DocumentType.GENERAL,
        "category": "test",
        "source_file": "test.txt",
        "priority": Priority.MEDIUM
    }

    chunks = chunking_service.chunk_document(text, metadata)

    # Should create multiple chunks
    assert len(chunks) > 1

    # All chunks should have metadata
    for chunk in chunks:
        assert chunk.metadata.company_id == "test"
        assert chunk.metadata.document_id == "doc2"


def test_chunk_faq(chunking_service):
    """Test chunking of FAQ data."""
    faqs = [
        {
            "question": "What is your return policy?",
            "answer": "We accept returns within 30 days.",
            "category": "returns"
        },
        {
            "question": "How do I track my order?",
            "answer": "You can track your order using the tracking number.",
            "category": "orders"
        }
    ]

    metadata = {
        "company_id": "test",
        "document_id": "faq1",
        "document_type": DocumentType.FAQ,
        "category": "faq",
        "source_file": "faqs.json",
        "priority": Priority.HIGH
    }

    chunks = chunking_service.chunk_faq(faqs, metadata)

    assert len(chunks) == 2
    assert "What is your return policy?" in chunks[0].text
    assert "How do I track my order?" in chunks[1].text


def test_chunk_validation(chunking_service):
    """Test chunk validation."""
    text = "This is a valid chunk with enough content."
    metadata = {
        "company_id": "test",
        "document_id": "doc3",
        "document_type": DocumentType.GENERAL,
        "category": "test",
        "source_file": "test.txt",
        "priority": Priority.MEDIUM
    }

    chunks = chunking_service.chunk_document(text, metadata)
    chunk = chunks[0]

    # Should pass validation
    assert chunking_service.validate_chunk(chunk) is True


def test_empty_faq_handling(chunking_service):
    """Test handling of empty FAQ entries."""
    faqs = [
        {
            "question": "",
            "answer": "This has no question",
            "category": "test"
        },
        {
            "question": "Valid question?",
            "answer": "Valid answer",
            "category": "test"
        }
    ]

    metadata = {
        "company_id": "test",
        "document_id": "faq2",
        "document_type": DocumentType.FAQ,
        "category": "faq",
        "source_file": "faqs.json",
        "priority": Priority.MEDIUM
    }

    chunks = chunking_service.chunk_faq(faqs, metadata)

    # Should only create chunk for valid FAQ
    assert len(chunks) == 1
    assert "Valid question?" in chunks[0].text
