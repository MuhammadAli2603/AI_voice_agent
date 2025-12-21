"""
FastAPI dependencies for dependency injection.
"""

from fastapi import Depends, HTTPException, status
from typing import Optional

from app.services.knowledge_service import KnowledgeService
from app.utils.logger import get_logger, set_request_id

logger = get_logger(__name__)

# Global knowledge service instance
_knowledge_service: Optional[KnowledgeService] = None


def get_knowledge_service() -> KnowledgeService:
    """
    Get or create the knowledge service instance.

    Returns:
        KnowledgeService instance
    """
    global _knowledge_service
    if _knowledge_service is None:
        logger.info("Initializing KnowledgeService...")
        _knowledge_service = KnowledgeService()
    return _knowledge_service


def verify_company_exists(
    company_id: str,
    knowledge_service: KnowledgeService = Depends(get_knowledge_service)
) -> str:
    """
    Verify that a company exists in the system.

    Args:
        company_id: Company identifier
        knowledge_service: Knowledge service instance

    Returns:
        Company ID if valid

    Raises:
        HTTPException: If company doesn't exist
    """
    companies = knowledge_service.get_companies()
    company_ids = [c.company_id for c in companies]

    if company_id not in company_ids:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Company '{company_id}' not found. Available companies: {company_ids}"
        )

    return company_id
