"""
FastAPI API endpoints for the Knowledge Base System.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from typing import List

from app.models.schemas import (
    QueryRequest,
    QueryResponse,
    CompanySwitchRequest,
    CompanySwitchResponse,
    IngestRequest,
    IngestResponse,
    HealthResponse,
    StatisticsResponse,
    CompanyListResponse,
    CompanyMetadata,
    ContextRequest,
    ContextResponse,
    ErrorResponse
)
from app.services.knowledge_service import KnowledgeService
from app.api.dependencies import get_knowledge_service, verify_company_exists
from app.utils.logger import get_logger, set_request_id

logger = get_logger(__name__)

router = APIRouter()


@router.post(
    "/company/load",
    response_model=IngestResponse,
    status_code=status.HTTP_200_OK,
    summary="Load and index a company's knowledge base",
    tags=["Company Management"]
)
async def load_company(
    request: IngestRequest,
    knowledge_service: KnowledgeService = Depends(get_knowledge_service)
):
    """
    Load and index a company's documents into the vector store.

    - **company_id**: Company identifier to load
    - **force_reindex**: Whether to force reindexing even if already indexed

    Returns statistics about the loading process including number of documents and chunks processed.
    """
    set_request_id()
    logger.info(f"Loading company: {request.company_id}")

    try:
        # Verify company exists
        verify_company_exists(request.company_id, knowledge_service)

        # Load company
        result = knowledge_service.load_company(
            request.company_id,
            request.force_reindex
        )

        return IngestResponse(
            success=True,
            company_id=request.company_id,
            chunks_processed=result.get('chunks_created', 0),
            documents_processed=result.get('documents_processed', 0),
            processing_time=result.get('processing_time', 0.0),
            message=result.get('message', 'Company loaded successfully')
        )

    except Exception as e:
        logger.error(f"Error loading company {request.company_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.post(
    "/company/switch",
    response_model=CompanySwitchResponse,
    summary="Switch active company",
    tags=["Company Management"]
)
async def switch_company(
    request: CompanySwitchRequest,
    knowledge_service: KnowledgeService = Depends(get_knowledge_service)
):
    """
    Switch the active company context for queries.

    - **company_id**: Company identifier to switch to

    The company must already be loaded/indexed before switching.
    """
    set_request_id()
    logger.info(f"Switching to company: {request.company_id}")

    try:
        # Verify company exists
        verify_company_exists(request.company_id, knowledge_service)

        # Switch company
        result = knowledge_service.switch_company(request.company_id)

        # Get company metadata
        companies = knowledge_service.get_companies()
        company_meta = next(
            (c for c in companies if c.company_id == request.company_id),
            None
        )

        stats = knowledge_service.get_statistics()

        return CompanySwitchResponse(
            success=True,
            company_id=request.company_id,
            company_name=company_meta.company_name if company_meta else request.company_id,
            message=result.get('message', 'Company switched successfully'),
            total_chunks=stats.get('total_chunks', 0)
        )

    except Exception as e:
        logger.error(f"Error switching company: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.post(
    "/query",
    response_model=QueryResponse,
    summary="Query the knowledge base",
    tags=["Query"]
)
async def query_knowledge_base(
    request: QueryRequest,
    knowledge_service: KnowledgeService = Depends(get_knowledge_service)
):
    """
    Query the knowledge base and retrieve relevant information.

    - **company_id**: Company to query
    - **query**: Question or search query
    - **top_k**: Number of results to return (1-20)
    - **filters**: Optional metadata filters

    Returns relevant document chunks with similarity scores and sources.
    """
    set_request_id()
    logger.info(f"Query for {request.company_id}: {request.query[:100]}")

    try:
        # Verify company exists
        verify_company_exists(request.company_id, knowledge_service)

        # Execute query
        response = knowledge_service.query(
            query=request.query,
            company_id=request.company_id,
            top_k=request.top_k
        )

        return response

    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.post(
    "/context",
    response_model=ContextResponse,
    summary="Get formatted context for LLM",
    tags=["Query"]
)
async def get_context_for_llm(
    request: ContextRequest,
    knowledge_service: KnowledgeService = Depends(get_knowledge_service)
):
    """
    Get formatted context ready for LLM consumption.

    - **company_id**: Company to query
    - **query**: Customer question
    - **top_k**: Number of chunks to retrieve
    - **include_metadata**: Whether to include source metadata

    Returns a formatted context string with instructions for the LLM.
    """
    set_request_id()
    logger.info(f"Context request for {request.company_id}: {request.query[:100]}")

    try:
        # Verify company exists
        verify_company_exists(request.company_id, knowledge_service)

        # Get context
        context_response = knowledge_service.get_context_for_llm(
            query=request.query,
            company_id=request.company_id,
            top_k=request.top_k,
            include_metadata=request.include_metadata
        )

        return context_response

    except Exception as e:
        logger.error(f"Error generating context: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get(
    "/companies",
    response_model=CompanyListResponse,
    summary="List all available companies",
    tags=["Company Management"]
)
async def list_companies(
    knowledge_service: KnowledgeService = Depends(get_knowledge_service)
):
    """
    Get a list of all available companies in the system.

    Returns company metadata including IDs, names, and categories.
    """
    set_request_id()

    try:
        companies = knowledge_service.get_companies()

        return CompanyListResponse(
            companies=companies,
            total=len(companies)
        )

    except Exception as e:
        logger.error(f"Error listing companies: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="System health check",
    tags=["System"]
)
async def health_check(
    knowledge_service: KnowledgeService = Depends(get_knowledge_service)
):
    """
    Get system health status and metrics.

    Returns information about current state, resource usage, and loaded data.
    """
    try:
        health = knowledge_service.get_health()

        return HealthResponse(**health)

    except Exception as e:
        logger.error(f"Error getting health status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.post(
    "/ingest",
    response_model=IngestResponse,
    summary="Ingest or re-ingest company documents",
    tags=["Company Management"]
)
async def ingest_documents(
    request: IngestRequest,
    knowledge_service: KnowledgeService = Depends(get_knowledge_service)
):
    """
    Ingest or re-ingest all documents for a company.

    - **company_id**: Company to ingest
    - **force_reindex**: Whether to force reindexing

    This processes all documents in the company's directory and updates the vector index.
    """
    # This is the same as load_company endpoint
    return await load_company(request, knowledge_service)


@router.delete(
    "/company/{company_id}",
    status_code=status.HTTP_200_OK,
    summary="Delete a company's data",
    tags=["Company Management"]
)
async def delete_company(
    company_id: str,
    knowledge_service: KnowledgeService = Depends(get_knowledge_service)
):
    """
    Remove a company's data from the system.

    Deletes all indexed chunks and clears from the vector store.
    """
    set_request_id()
    logger.info(f"Deleting company: {company_id}")

    try:
        result = knowledge_service.delete_company(company_id)

        return {
            "success": True,
            "company_id": company_id,
            "chunks_deleted": result.get('chunks_deleted', 0),
            "message": f"Company {company_id} deleted successfully"
        }

    except Exception as e:
        logger.error(f"Error deleting company: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get(
    "/stats",
    response_model=StatisticsResponse,
    summary="Get usage statistics",
    tags=["System"]
)
async def get_statistics(
    knowledge_service: KnowledgeService = Depends(get_knowledge_service)
):
    """
    Get detailed usage statistics and metrics.

    Returns information about queries, performance, cache hit rates, and more.
    """
    set_request_id()

    try:
        stats = knowledge_service.get_statistics()

        return StatisticsResponse(
            total_queries=stats['total_queries'],
            total_companies=stats['total_companies'],
            total_chunks=stats['total_chunks'],
            average_query_time=stats['average_query_time'],
            cache_hit_rate=stats['cache_hit_rate'],
            uptime_seconds=stats['uptime_seconds'],
            queries_per_company=stats['queries_per_company']
        )

    except Exception as e:
        logger.error(f"Error getting statistics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.post(
    "/cache/clear",
    status_code=status.HTTP_200_OK,
    summary="Clear all caches",
    tags=["System"]
)
async def clear_cache(
    knowledge_service: KnowledgeService = Depends(get_knowledge_service)
):
    """
    Clear query cache and embedding cache.

    Use this to free up memory or force fresh results.
    """
    set_request_id()
    logger.info("Clearing caches")

    try:
        knowledge_service.clear_cache()

        return {
            "success": True,
            "message": "Caches cleared successfully"
        }

    except Exception as e:
        logger.error(f"Error clearing cache: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )
