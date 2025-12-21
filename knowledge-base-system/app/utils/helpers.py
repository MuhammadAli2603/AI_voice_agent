"""
Utility helper functions for the Knowledge Base System.
"""

import hashlib
import time
from typing import Any, Callable, Dict
from functools import wraps
import psutil
import os


def generate_chunk_id(company_id: str, document_id: str, chunk_index: int) -> str:
    """
    Generate a unique chunk ID.

    Args:
        company_id: Company identifier
        document_id: Document identifier
        chunk_index: Index of chunk within document

    Returns:
        Unique chunk ID
    """
    content = f"{company_id}_{document_id}_{chunk_index}"
    return hashlib.md5(content.encode()).hexdigest()[:16]


def generate_document_id(file_path: str) -> str:
    """
    Generate a unique document ID from file path.

    Args:
        file_path: Path to the document file

    Returns:
        Unique document ID
    """
    return hashlib.md5(file_path.encode()).hexdigest()[:16]


def word_count(text: str) -> int:
    """
    Count words in text.

    Args:
        text: Input text

    Returns:
        Number of words
    """
    return len(text.split())


def char_count(text: str) -> int:
    """
    Count characters in text.

    Args:
        text: Input text

    Returns:
        Number of characters
    """
    return len(text)


def normalize_text(text: str) -> str:
    """
    Normalize text by removing extra whitespace and newlines.

    Args:
        text: Input text

    Returns:
        Normalized text
    """
    # Replace multiple spaces with single space
    text = ' '.join(text.split())
    return text.strip()


def get_memory_usage() -> float:
    """
    Get current process memory usage in MB.

    Returns:
        Memory usage in megabytes
    """
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


def timer(func: Callable) -> Callable:
    """
    Decorator to measure function execution time.

    Args:
        func: Function to time

    Returns:
        Wrapped function that logs execution time
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        return result, execution_time
    return wrapper


def format_sources(chunks: list) -> list:
    """
    Extract and format unique sources from chunks.

    Args:
        chunks: List of document chunks

    Returns:
        List of unique source identifiers
    """
    sources = set()
    for chunk in chunks:
        if hasattr(chunk, 'metadata'):
            source = f"{chunk.metadata.document_type}:{chunk.metadata.source_file}"
            sources.add(source)
    return sorted(list(sources))


def calculate_confidence(similarity_scores: list, threshold: float = 0.3) -> float:
    """
    Calculate confidence score based on similarity scores.

    Args:
        similarity_scores: List of similarity scores
        threshold: Minimum similarity threshold

    Returns:
        Confidence score between 0 and 1
    """
    if not similarity_scores:
        return 0.0

    # Filter scores above threshold
    relevant_scores = [s for s in similarity_scores if s >= threshold]

    if not relevant_scores:
        return 0.0

    # Weighted average with exponential decay
    weights = [0.4, 0.3, 0.2, 0.1] + [0.0] * (len(relevant_scores) - 4)
    weighted_sum = sum(score * weight for score, weight in zip(relevant_scores, weights))
    weight_sum = sum(weights[:len(relevant_scores)])

    if weight_sum == 0:
        return 0.0

    confidence = weighted_sum / weight_sum
    return min(confidence, 1.0)


def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """
    Truncate text to maximum length.

    Args:
        text: Input text
        max_length: Maximum length
        suffix: Suffix to add if truncated

    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix


def validate_company_id(company_id: str) -> bool:
    """
    Validate company ID format.

    Args:
        company_id: Company identifier

    Returns:
        True if valid, False otherwise
    """
    # Company ID should be alphanumeric with optional underscores/hyphens
    return company_id.replace('_', '').replace('-', '').isalnum()
