"""
Knowledge service providing high-level business logic for the knowledge base system.
Manages company switching, caching, and coordinates between components.
"""

import time
from typing import Dict, Any, Optional, List
from datetime import datetime
import json
from pathlib import Path

from app.core.embeddings import EmbeddingService
from app.core.chunking import ChunkingService
from app.core.retrieval import RetrievalService
from app.core.vector_store import VectorStore, FAISSVectorStore
from app.services.document_processor import DocumentProcessor
from app.models.schemas import (
    QueryResponse,
    CompanyMetadata,
    ContextResponse
)
from app.config import settings
from app.utils.logger import get_logger
from app.utils.helpers import get_memory_usage

logger = get_logger(__name__)


class KnowledgeService:
    """
    High-level service coordinating all knowledge base operations.

    Manages:
    - Company loading and switching
    - Query processing
    - Caching
    - Statistics tracking
    - Index persistence
    """

    def __init__(self):
        """Initialize the knowledge service and all components."""
        self.start_time = time.time()

        logger.info("Initializing KnowledgeService...")

        # Initialize core components
        self.embedding_service = EmbeddingService()
        self.chunking_service = ChunkingService()
        self.vector_store = FAISSVectorStore(
            dimension=self.embedding_service.get_embedding_dimension()
        )
        self.retrieval_service = RetrievalService(
            self.embedding_service,
            self.vector_store
        )
        self.document_processor = DocumentProcessor(
            self.chunking_service,
            self.embedding_service,
            self.vector_store
        )

        # State management
        self.current_company: Optional[str] = None
        self.loaded_companies: Dict[str, CompanyMetadata] = {}

        # Statistics
        self.query_count = 0
        self.queries_per_company: Dict[str, int] = {}
        self.total_query_time = 0.0

        # Cache (simple in-memory cache)
        self.query_cache: Dict[str, QueryResponse] = {}
        self.cache_hits = 0
        self.cache_misses = 0

        # Warmup embedding model
        self.embedding_service.warmup()

        # Load available companies
        self._discover_companies()

        # Load default company if specified
        if settings.default_company_id:
            try:
                self.load_company(settings.default_company_id)
                logger.info(f"Loaded default company: {settings.default_company_id}")
            except Exception as e:
                logger.warning(f"Could not load default company: {e}")

        logger.info("KnowledgeService initialized successfully")

    def _discover_companies(self):
        """Discover available companies from knowledge base directory."""
        config_file = Path(settings.knowledge_base_dir) / "config.json"

        if config_file.exists():
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
                for company_data in config.get('companies', []):
                    company_metadata = CompanyMetadata(**company_data)
                    self.loaded_companies[company_metadata.company_id] = company_metadata

            logger.info(f"Discovered {len(self.loaded_companies)} companies")

    def load_company(self, company_id: str, force_reindex: bool = False) -> Dict[str, Any]:
        """
        Load and index a company's knowledge base.

        Args:
            company_id: Company identifier
            force_reindex: Whether to force reindexing

        Returns:
            Loading statistics
        """
        logger.info(f"Loading company: {company_id}")

        # Check if company exists
        if company_id not in self.loaded_companies:
            raise ValueError(f"Company not found: {company_id}")

        # Check if index exists and load it
        index_path = str(settings.get_index_path(company_id)).replace('.index', '')

        if not force_reindex and Path(f"{index_path}.faiss").exists():
            try:
                logger.info(f"Loading existing index for {company_id}")
                self.vector_store.load_index(index_path)
                self.current_company = company_id
                stats = self.vector_store.get_stats()

                return {
                    "company_id": company_id,
                    "loaded_from_cache": True,
                    "total_chunks": stats.get('total_vectors', 0),
                    "message": "Loaded from existing index"
                }
            except Exception as e:
                logger.warning(f"Could not load index, will reprocess: {e}")

        # Process company documents
        stats = self.document_processor.process_company(company_id, force_reindex)

        # Save index
        self.save_index(company_id)

        # Set as current company
        self.current_company = company_id

        return {
            **stats,
            "loaded_from_cache": False,
            "message": "Processed and indexed successfully"
        }

    def switch_company(self, company_id: str) -> Dict[str, Any]:
        """
        Switch to a different company (must be already loaded).

        Args:
            company_id: Company identifier

        Returns:
            Switch status
        """
        if company_id not in self.loaded_companies:
            raise ValueError(f"Company not found: {company_id}")

        # Check if company is indexed
        stats = self.vector_store.get_stats()
        if company_id not in stats.get('companies', {}):
            # Need to load company first
            return self.load_company(company_id)

        self.current_company = company_id
        logger.info(f"Switched to company: {company_id}")

        return {
            "company_id": company_id,
            "success": True,
            "message": f"Switched to {company_id}"
        }

    def query(
        self,
        query: str,
        company_id: Optional[str] = None,
        top_k: int = 5,
        use_cache: bool = True
    ) -> QueryResponse:
        """
        Query the knowledge base.

        Args:
            query: User query
            company_id: Company to query (uses current if None)
            top_k: Number of results
            use_cache: Whether to use cache

        Returns:
            QueryResponse with results
        """
        # Use current company if not specified
        if company_id is None:
            company_id = self.current_company

        if company_id is None:
            raise ValueError("No company specified and no current company set")

        # Check cache
        cache_key = f"{company_id}:{query}:{top_k}"
        if use_cache and cache_key in self.query_cache:
            self.cache_hits += 1
            logger.info(f"Cache hit for query: {query[:50]}")
            return self.query_cache[cache_key]

        self.cache_misses += 1

        # Execute query
        start_time = time.time()
        response = self.retrieval_service.retrieve(
            query=query,
            company_id=company_id,
            top_k=top_k
        )
        query_time = time.time() - start_time

        # Update statistics
        self.query_count += 1
        self.total_query_time += query_time
        self.queries_per_company[company_id] = self.queries_per_company.get(company_id, 0) + 1

        # Cache result
        if use_cache:
            self.query_cache[cache_key] = response

        logger.info(
            f"Query processed for {company_id}: "
            f"{len(response.chunks)} results in {query_time:.3f}s"
        )

        return response

    def get_context_for_llm(
        self,
        query: str,
        company_id: Optional[str] = None,
        top_k: int = 5,
        include_metadata: bool = True
    ) -> ContextResponse:
        """
        Get formatted context for LLM consumption.

        Args:
            query: User query
            company_id: Company to query
            top_k: Number of chunks to retrieve
            include_metadata: Whether to include source metadata

        Returns:
            ContextResponse with formatted context
        """
        # Get company name
        if company_id is None:
            company_id = self.current_company

        company_metadata = self.loaded_companies.get(company_id)
        company_name = company_metadata.company_name if company_metadata else company_id

        # Query knowledge base
        query_response = self.query(query, company_id, top_k)

        # Assemble context
        formatted_context = self.retrieval_service.assemble_context_for_llm(
            chunks=query_response.chunks,
            query=query,
            company_name=company_name,
            include_metadata=include_metadata
        )

        return ContextResponse(
            formatted_context=formatted_context,
            company_name=company_name,
            query=query,
            num_chunks=len(query_response.chunks),
            confidence_score=query_response.confidence_score,
            retrieval_time=query_response.retrieval_time
        )

    def get_companies(self) -> List[CompanyMetadata]:
        """
        Get list of all available companies.

        Returns:
            List of CompanyMetadata
        """
        return list(self.loaded_companies.values())

    def get_current_company(self) -> Optional[str]:
        """
        Get current active company.

        Returns:
            Company ID or None
        """
        return self.current_company

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get usage statistics.

        Returns:
            Statistics dictionary
        """
        uptime = time.time() - self.start_time
        avg_query_time = (
            self.total_query_time / self.query_count
            if self.query_count > 0
            else 0.0
        )

        cache_requests = self.cache_hits + self.cache_misses
        cache_hit_rate = (
            self.cache_hits / cache_requests
            if cache_requests > 0
            else 0.0
        )

        vs_stats = self.vector_store.get_stats()

        return {
            "total_queries": self.query_count,
            "total_companies": len(self.loaded_companies),
            "total_chunks": vs_stats.get('total_vectors', 0),
            "average_query_time": avg_query_time,
            "cache_hit_rate": cache_hit_rate,
            "uptime_seconds": uptime,
            "queries_per_company": self.queries_per_company,
            "memory_usage_mb": get_memory_usage(),
            "current_company": self.current_company,
            "vector_store_stats": vs_stats,
            "embedding_cache_stats": self.embedding_service.get_cache_stats()
        }

    def get_health(self) -> Dict[str, Any]:
        """
        Get system health status.

        Returns:
            Health status dictionary
        """
        vs_stats = self.vector_store.get_stats()
        uptime = time.time() - self.start_time

        return {
            "status": "healthy",
            "current_company": self.current_company,
            "total_chunks": vs_stats.get('total_vectors', 0),
            "memory_usage_mb": get_memory_usage(),
            "uptime_seconds": uptime,
            "companies_loaded": len(vs_stats.get('companies', {})),
            "embedding_model": settings.embedding_model_name
        }

    def clear_cache(self):
        """Clear query cache and embedding cache."""
        self.query_cache.clear()
        self.embedding_service.clear_cache()
        self.cache_hits = 0
        self.cache_misses = 0
        logger.info("Caches cleared")

    def save_index(self, company_id: Optional[str] = None):
        """
        Save vector index to disk.

        Args:
            company_id: Company ID (uses current if None)
        """
        if company_id is None:
            company_id = self.current_company

        if company_id is None:
            raise ValueError("No company specified")

        index_path = str(settings.get_index_path(company_id)).replace('.index', '')
        self.vector_store.save_index(index_path)
        logger.info(f"Index saved for company: {company_id}")

    def delete_company(self, company_id: str) -> Dict[str, Any]:
        """
        Delete a company's data from the system.

        Args:
            company_id: Company identifier

        Returns:
            Deletion status
        """
        deleted_count = self.vector_store.delete_by_company(company_id)

        # Delete index files
        index_path = settings.get_index_path(company_id)
        try:
            if Path(f"{index_path}.faiss").exists():
                Path(f"{index_path}.faiss").unlink()
            if Path(f"{index_path}_metadata.pkl").exists():
                Path(f"{index_path}_metadata.pkl").unlink()
        except Exception as e:
            logger.warning(f"Error deleting index files: {e}")

        # Clear from current company if it was active
        if self.current_company == company_id:
            self.current_company = None

        logger.info(f"Deleted company {company_id}: {deleted_count} chunks removed")

        return {
            "company_id": company_id,
            "chunks_deleted": deleted_count,
            "success": True
        }
