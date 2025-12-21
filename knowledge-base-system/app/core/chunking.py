"""
Document chunking service with hybrid sentence-based chunking strategy.
Implements intelligent text splitting with semantic awareness.
"""

import re
from typing import List, Dict, Any, Tuple
import nltk
from datetime import datetime

from app.models.schemas import (
    DocumentChunk,
    ChunkMetadata,
    DocumentType,
    Priority
)
from app.config import settings
from app.utils.helpers import (
    generate_chunk_id,
    word_count,
    normalize_text
)
from app.utils.logger import get_logger

logger = get_logger(__name__)


# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)


class ChunkingService:
    """
    Service for chunking documents using sentence-based strategy.

    Implements hybrid sentence-based chunking with:
    - Target chunk size: 400-500 words
    - Minimum chunk size: 200 words
    - Maximum chunk size: 600 words
    - Overlap: 50-100 words between chunks
    - Sentence boundary awareness
    - Paragraph preservation where possible
    """

    def __init__(self):
        """Initialize the chunking service."""
        self.min_chunk_size = settings.chunk_size_min
        self.target_chunk_size = settings.chunk_size_target
        self.max_chunk_size = settings.chunk_size_max
        self.overlap_size = settings.chunk_overlap

        logger.info(
            f"ChunkingService initialized - "
            f"Target: {self.target_chunk_size}, "
            f"Min: {self.min_chunk_size}, "
            f"Max: {self.max_chunk_size}, "
            f"Overlap: {self.overlap_size}"
        )

    def chunk_document(
        self,
        text: str,
        metadata: Dict[str, Any]
    ) -> List[DocumentChunk]:
        """
        Chunk a document using sentence-based strategy.

        Args:
            text: Document text to chunk
            metadata: Base metadata for the document

        Returns:
            List of DocumentChunk objects
        """
        # Preprocess text
        text = self._preprocess_text(text)

        if word_count(text) < self.min_chunk_size:
            # Document is small enough to be a single chunk
            return [self._create_single_chunk(text, metadata, 0, 1)]

        # Split into sentences
        sentences = self._split_into_sentences(text)

        # Group sentences into chunks
        chunks = self._group_sentences_into_chunks(sentences)

        # Create DocumentChunk objects
        document_chunks = []
        total_chunks = len(chunks)

        for idx, chunk_text in enumerate(chunks):
            if self._validate_chunk_size(chunk_text):
                doc_chunk = self._create_chunk(
                    chunk_text,
                    metadata,
                    idx,
                    total_chunks
                )
                document_chunks.append(doc_chunk)
            else:
                logger.warning(
                    f"Chunk {idx} failed validation - "
                    f"Size: {word_count(chunk_text)} words"
                )

        logger.info(
            f"Chunked document {metadata.get('document_id', 'unknown')} "
            f"into {len(document_chunks)} chunks"
        )

        return document_chunks

    def chunk_faq(self, faq_list: List[Dict[str, str]], metadata: Dict[str, Any]) -> List[DocumentChunk]:
        """
        Chunk FAQ data - each Q&A pair becomes one chunk.

        Args:
            faq_list: List of FAQ dictionaries with 'question' and 'answer'
            metadata: Base metadata

        Returns:
            List of DocumentChunk objects
        """
        chunks = []
        total_faqs = len(faq_list)

        for idx, faq in enumerate(faq_list):
            question = faq.get('question', '').strip()
            answer = faq.get('answer', '').strip()
            category = faq.get('category', metadata.get('category', 'general'))

            if not question or not answer:
                logger.warning(f"Skipping FAQ {idx} - missing question or answer")
                continue

            # Combine Q&A into single chunk
            chunk_text = f"Q: {question}\n\nA: {answer}"

            # Update metadata for this FAQ
            faq_metadata = metadata.copy()
            faq_metadata['category'] = category
            faq_metadata['document_type'] = DocumentType.FAQ

            chunk = self._create_chunk(
                chunk_text,
                faq_metadata,
                idx,
                total_faqs
            )
            chunks.append(chunk)

        logger.info(f"Created {len(chunks)} FAQ chunks")
        return chunks

    def chunk_json_document(
        self,
        json_data: Dict[str, Any],
        metadata: Dict[str, Any]
    ) -> List[DocumentChunk]:
        """
        Chunk JSON document by converting to text representation.

        Args:
            json_data: JSON data to chunk
            metadata: Base metadata

        Returns:
            List of DocumentChunk objects
        """
        # Convert JSON to readable text
        text = self._json_to_text(json_data)
        return self.chunk_document(text, metadata)

    def _preprocess_text(self, text: str) -> str:
        """
        Preprocess text by normalizing whitespace and cleaning.

        Args:
            text: Input text

        Returns:
            Preprocessed text
        """
        # Normalize line endings
        text = text.replace('\r\n', '\n').replace('\r', '\n')

        # Remove excessive blank lines (more than 2)
        text = re.sub(r'\n{3,}', '\n\n', text)

        # Normalize spaces (but preserve single newlines for paragraph detection)
        lines = text.split('\n')
        cleaned_lines = [' '.join(line.split()) for line in lines]
        text = '\n'.join(cleaned_lines)

        return text.strip()

    def _split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences using NLTK.

        Args:
            text: Input text

        Returns:
            List of sentences
        """
        # Use NLTK sentence tokenizer
        sentences = nltk.sent_tokenize(text)

        # Clean sentences
        sentences = [s.strip() for s in sentences if s.strip()]

        return sentences

    def _group_sentences_into_chunks(self, sentences: List[str]) -> List[str]:
        """
        Group sentences into chunks based on target size with overlap.

        Args:
            sentences: List of sentences

        Returns:
            List of chunk texts
        """
        chunks = []
        current_chunk = []
        current_word_count = 0

        for sentence in sentences:
            sentence_words = word_count(sentence)

            # Check if adding this sentence would exceed max size
            if current_word_count + sentence_words > self.max_chunk_size:
                # Finalize current chunk if it meets minimum size
                if current_word_count >= self.min_chunk_size:
                    chunks.append(' '.join(current_chunk))

                    # Create overlap for next chunk
                    overlap_chunk = self._create_overlap(current_chunk)
                    current_chunk = overlap_chunk
                    current_word_count = sum(word_count(s) for s in current_chunk)
                else:
                    # Chunk is too small, force-add this sentence
                    current_chunk.append(sentence)
                    current_word_count += sentence_words
                    chunks.append(' '.join(current_chunk))
                    current_chunk = []
                    current_word_count = 0
                    continue

            # Add sentence to current chunk
            current_chunk.append(sentence)
            current_word_count += sentence_words

            # Check if we've reached target size
            if current_word_count >= self.target_chunk_size:
                chunks.append(' '.join(current_chunk))

                # Create overlap for next chunk
                overlap_chunk = self._create_overlap(current_chunk)
                current_chunk = overlap_chunk
                current_word_count = sum(word_count(s) for s in current_chunk)

        # Add remaining sentences as final chunk
        if current_chunk and current_word_count >= self.min_chunk_size:
            chunks.append(' '.join(current_chunk))
        elif current_chunk and chunks:
            # Merge small final chunk with previous chunk
            chunks[-1] = chunks[-1] + ' ' + ' '.join(current_chunk)

        return chunks

    def _create_overlap(self, sentences: List[str]) -> List[str]:
        """
        Create overlap sentences for next chunk.

        Args:
            sentences: Current chunk sentences

        Returns:
            Overlap sentences
        """
        overlap_sentences = []
        overlap_words = 0

        # Take sentences from the end until we reach overlap size
        for sentence in reversed(sentences):
            sentence_words = word_count(sentence)
            if overlap_words + sentence_words > self.overlap_size:
                break
            overlap_sentences.insert(0, sentence)
            overlap_words += sentence_words

        return overlap_sentences

    def _validate_chunk_size(self, chunk_text: str) -> bool:
        """
        Validate that chunk meets size requirements.

        Args:
            chunk_text: Chunk text

        Returns:
            True if valid, False otherwise
        """
        words = word_count(chunk_text)
        return self.min_chunk_size <= words <= self.max_chunk_size

    def _create_chunk(
        self,
        text: str,
        metadata: Dict[str, Any],
        chunk_index: int,
        total_chunks: int
    ) -> DocumentChunk:
        """
        Create a DocumentChunk object.

        Args:
            text: Chunk text
            metadata: Base metadata
            chunk_index: Index of chunk
            total_chunks: Total number of chunks

        Returns:
            DocumentChunk object
        """
        company_id = metadata.get('company_id', 'unknown')
        document_id = metadata.get('document_id', 'unknown')

        chunk_id = generate_chunk_id(company_id, document_id, chunk_index)

        chunk_metadata = ChunkMetadata(
            chunk_id=chunk_id,
            company_id=company_id,
            document_id=document_id,
            document_type=metadata.get('document_type', DocumentType.GENERAL),
            category=metadata.get('category', 'general'),
            chunk_index=chunk_index,
            total_chunks=total_chunks,
            source_file=metadata.get('source_file', 'unknown'),
            priority=metadata.get('priority', Priority.MEDIUM)
        )

        return DocumentChunk(
            chunk_id=chunk_id,
            text=text,
            metadata=chunk_metadata
        )

    def _create_single_chunk(
        self,
        text: str,
        metadata: Dict[str, Any],
        chunk_index: int,
        total_chunks: int
    ) -> DocumentChunk:
        """Create a single chunk for short documents."""
        return self._create_chunk(text, metadata, chunk_index, total_chunks)

    def _json_to_text(self, data: Dict[str, Any], prefix: str = "") -> str:
        """
        Convert JSON data to readable text format.

        Args:
            data: JSON data
            prefix: Prefix for nested items

        Returns:
            Text representation
        """
        lines = []

        for key, value in data.items():
            if isinstance(value, dict):
                lines.append(f"{prefix}{key}:")
                lines.append(self._json_to_text(value, prefix + "  "))
            elif isinstance(value, list):
                lines.append(f"{prefix}{key}:")
                for item in value:
                    if isinstance(item, dict):
                        lines.append(self._json_to_text(item, prefix + "  - "))
                    else:
                        lines.append(f"{prefix}  - {item}")
            else:
                lines.append(f"{prefix}{key}: {value}")

        return '\n'.join(lines)

    def validate_chunk(self, chunk: DocumentChunk) -> bool:
        """
        Validate a DocumentChunk.

        Args:
            chunk: DocumentChunk to validate

        Returns:
            True if valid, False otherwise
        """
        # Check text is not empty
        if not chunk.text or not chunk.text.strip():
            return False

        # Check size requirements
        if not self._validate_chunk_size(chunk.text):
            return False

        # Check metadata is present
        if not chunk.metadata:
            return False

        return True
