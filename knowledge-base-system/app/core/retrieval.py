"""
Retrieval service for RAG-based query answering.
Handles query processing, similarity search, re-ranking, and context assembly.
"""

import time
import re
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

from app.models.schemas import (
    QueryResponse,
    RetrievedChunk,
    DocumentChunk,
    Priority
)
from app.core.embeddings import EmbeddingService
from app.core.vector_store import VectorStore
from app.config import settings
from app.utils.helpers import calculate_confidence, format_sources
from app.utils.logger import get_logger

logger = get_logger(__name__)


class RetrievalService:
    """
    Service for retrieving relevant information from the knowledge base.

    Handles:
    - Query preprocessing
    - Embedding generation
    - Similarity search with filtering
    - Result re-ranking
    - Confidence scoring
    - Context assembly for LLM
    """

    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_store: VectorStore
    ):
        """
        Initialize the retrieval service.

        Args:
            embedding_service: Service for generating embeddings
            vector_store: Vector store for similarity search
        """
        self.embedding_service = embedding_service
        self.vector_store = vector_store
        self.similarity_threshold = settings.similarity_threshold
        self.rerank_enabled = settings.rerank_enabled

        logger.info("RetrievalService initialized")

    def retrieve(
        self,
        query: str,
        company_id: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> QueryResponse:
        """
        Retrieve relevant chunks for a query.

        Args:
            query: User query
            company_id: Company identifier
            top_k: Number of results to return
            filters: Optional metadata filters

        Returns:
            QueryResponse with retrieved chunks and metadata
        """
        start_time = time.time()

        # Preprocess query
        processed_query = self.preprocess_query(query)

        # Generate query embedding
        query_embedding = self.embedding_service.encode_text(processed_query)

        # Add company filter
        if filters is None:
            filters = {}
        filters['company_id'] = company_id

        # Search vector store
        results = self.vector_store.search(
            query_embedding,
            top_k=top_k * 2 if self.rerank_enabled else top_k,  # Fetch more for re-ranking
            filters=filters
        )

        # Filter by similarity threshold
        results = [
            (chunk, score)
            for chunk, score in results
            if score >= self.similarity_threshold
        ]

        # Re-rank results if enabled
        if self.rerank_enabled and results:
            results = self.rerank_results(results, query)

        # Take top_k after re-ranking
        results = results[:top_k]

        # Calculate confidence score
        similarity_scores = [score for _, score in results]
        confidence_score = calculate_confidence(similarity_scores, self.similarity_threshold)

        # Convert to RetrievedChunk objects
        retrieved_chunks = []
        for chunk, score in results:
            retrieved_chunk = RetrievedChunk(
                chunk_id=chunk.chunk_id,
                text=chunk.text,
                metadata=chunk.metadata,
                similarity_score=score,
                source=f"{chunk.metadata.document_type}:{chunk.metadata.source_file}"
            )
            retrieved_chunks.append(retrieved_chunk)

        # Get unique sources
        sources = format_sources([chunk for chunk, _ in results])

        retrieval_time = time.time() - start_time

        logger.info(
            f"Retrieved {len(retrieved_chunks)} chunks for company {company_id} "
            f"in {retrieval_time:.3f}s - Confidence: {confidence_score:.2f}"
        )

        return QueryResponse(
            chunks=retrieved_chunks,
            confidence_score=confidence_score,
            sources=sources,
            retrieval_time=retrieval_time,
            company_id=company_id,
            total_results=len(retrieved_chunks)
        )

    def preprocess_query(self, query: str) -> str:
        """
        Preprocess query text.

        Args:
            query: Raw query text

        Returns:
            Preprocessed query
        """
        # Convert to lowercase
        query = query.lower()

        # Remove extra whitespace
        query = ' '.join(query.split())

        # Remove special characters but keep question marks and basic punctuation
        query = re.sub(r'[^\w\s\?\!,\.]', '', query)

        return query.strip()

    def rerank_results(
        self,
        results: List[Tuple[DocumentChunk, float]],
        query: str
    ) -> List[Tuple[DocumentChunk, float]]:
        """
        Re-rank results based on metadata priority and other factors.

        Args:
            results: List of (chunk, similarity_score) tuples
            query: Original query

        Returns:
            Re-ranked list of (chunk, similarity_score) tuples
        """
        scored_results = []

        for chunk, similarity_score in results:
            # Start with similarity score
            final_score = similarity_score

            # Boost based on priority
            priority = chunk.metadata.priority
            if priority == Priority.HIGH:
                final_score *= 1.2
            elif priority == Priority.LOW:
                final_score *= 0.9

            # Boost FAQs slightly (they're usually more direct)
            if chunk.metadata.document_type == "faq":
                final_score *= 1.1

            # Boost if query keywords appear in the chunk text
            keyword_boost = self._calculate_keyword_overlap(query, chunk.text)
            final_score *= (1.0 + keyword_boost * 0.1)

            scored_results.append((chunk, similarity_score, final_score))

        # Sort by final score
        scored_results.sort(key=lambda x: x[2], reverse=True)

        # Return with original similarity scores for transparency
        return [(chunk, sim_score) for chunk, sim_score, _ in scored_results]

    def _calculate_keyword_overlap(self, query: str, text: str) -> float:
        """
        Calculate keyword overlap between query and text.

        Args:
            query: Query text
            text: Chunk text

        Returns:
            Overlap score (0-1)
        """
        # Extract keywords from query (words longer than 3 characters)
        query_keywords = set(
            word.lower()
            for word in query.split()
            if len(word) > 3
        )

        if not query_keywords:
            return 0.0

        text_lower = text.lower()

        # Count how many keywords appear in text
        matches = sum(1 for keyword in query_keywords if keyword in text_lower)

        return matches / len(query_keywords)

    def assemble_context_for_llm(
        self,
        chunks: List[RetrievedChunk],
        query: str,
        company_name: str,
        include_metadata: bool = True
    ) -> str:
        """
        Assemble retrieved chunks into formatted context for LLM.

        Args:
            chunks: Retrieved chunks
            query: Original query
            company_name: Name of the company
            include_metadata: Whether to include source metadata

        Returns:
            Formatted context string
        """
        from datetime import datetime

        # Build context
        context_parts = []

        # Header
        context_parts.append(f"You are a customer service agent for {company_name}.")
        context_parts.append(f"Current date: {datetime.now().strftime('%Y-%m-%d')}")
        context_parts.append("")
        context_parts.append("Relevant Information:")
        context_parts.append("")

        # Add each chunk
        for i, chunk in enumerate(chunks, 1):
            context_parts.append(f"[Source {i}]")
            context_parts.append(chunk.text)

            if include_metadata:
                context_parts.append(
                    f"(Source: {chunk.metadata.document_type} - "
                    f"{chunk.metadata.category}, "
                    f"Confidence: {chunk.similarity_score:.2f})"
                )

            context_parts.append("")

        # Instructions
        context_parts.append("Instructions:")
        context_parts.append(
            "- Only use the above information to answer the customer's question."
        )
        context_parts.append(
            "- If the answer is not in the provided information, "
            "say 'I don't have that information available.'"
        )
        context_parts.append(
            "- Be helpful, concise, and professional."
        )
        context_parts.append(
            "- Reference the source numbers when applicable."
        )
        context_parts.append("")

        # Customer question
        context_parts.append(f"Customer Question: {query}")

        return "\n".join(context_parts)

    def calculate_confidence(self, results: List[Tuple[DocumentChunk, float]]) -> float:
        """
        Calculate overall confidence score for retrieval results.

        Args:
            results: List of (chunk, similarity_score) tuples

        Returns:
            Confidence score (0-1)
        """
        if not results:
            return 0.0

        scores = [score for _, score in results]
        return calculate_confidence(scores, self.similarity_threshold)
