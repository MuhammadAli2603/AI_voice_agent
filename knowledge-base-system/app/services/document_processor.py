"""
Document processing service for ingesting and indexing knowledge base documents.
Handles file parsing, chunking, embedding, and storage.
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import time

from app.core.chunking import ChunkingService
from app.core.embeddings import EmbeddingService
from app.core.vector_store import VectorStore
from app.models.schemas import DocumentChunk, DocumentType, Priority
from app.config import settings
from app.utils.helpers import generate_document_id
from app.utils.logger import get_logger

logger = get_logger(__name__)


class DocumentProcessor:
    """
    Service for processing and indexing documents into the vector store.

    Handles:
    - Loading documents from file system
    - Parsing different file formats
    - Chunking documents
    - Generating embeddings
    - Storing in vector database
    """

    def __init__(
        self,
        chunking_service: ChunkingService,
        embedding_service: EmbeddingService,
        vector_store: VectorStore
    ):
        """
        Initialize the document processor.

        Args:
            chunking_service: Service for chunking documents
            embedding_service: Service for generating embeddings
            vector_store: Vector store for indexing
        """
        self.chunking_service = chunking_service
        self.embedding_service = embedding_service
        self.vector_store = vector_store
        self.knowledge_base_dir = Path(settings.knowledge_base_dir)

        logger.info("DocumentProcessor initialized")

    def process_company(self, company_id: str, force_reindex: bool = False) -> Dict[str, Any]:
        """
        Process all documents for a company.

        Args:
            company_id: Company identifier
            force_reindex: Whether to reindex even if already indexed

        Returns:
            Dictionary with processing statistics
        """
        start_time = time.time()

        logger.info(f"Processing company: {company_id}")

        # Check if company directory exists
        company_dir = self.knowledge_base_dir / company_id
        if not company_dir.exists():
            raise FileNotFoundError(f"Company directory not found: {company_dir}")

        # If force reindex, delete existing data
        if force_reindex:
            logger.info(f"Force reindex enabled - removing existing data for {company_id}")
            self.vector_store.delete_by_company(company_id)

        # Load company metadata
        metadata_file = company_dir / "metadata.json"
        if metadata_file.exists():
            with open(metadata_file, 'r', encoding='utf-8') as f:
                company_metadata = json.load(f)
        else:
            company_metadata = {"company_id": company_id}

        # Load all documents
        documents = self.load_company_documents(company_id)
        logger.info(f"Loaded {len(documents)} documents for {company_id}")

        # Process each document
        all_chunks = []
        for doc in documents:
            try:
                chunks = self.process_document(doc, company_id, company_metadata)
                all_chunks.extend(chunks)
            except Exception as e:
                logger.error(f"Error processing document {doc.get('file_path')}: {e}")

        # Generate embeddings for all chunks
        logger.info(f"Generating embeddings for {len(all_chunks)} chunks")
        texts = [chunk.text for chunk in all_chunks]
        embeddings = self.embedding_service.encode_batch(texts, show_progress=True)

        # Store embeddings in chunks for later use (e.g., when rebuilding index)
        for chunk, embedding in zip(all_chunks, embeddings):
            chunk.embedding = embedding

        # Add to vector store
        logger.info(f"Adding {len(all_chunks)} chunks to vector store")
        self.vector_store.add_documents(all_chunks, embeddings)

        processing_time = time.time() - start_time

        stats = {
            "company_id": company_id,
            "documents_processed": len(documents),
            "chunks_created": len(all_chunks),
            "processing_time": processing_time,
            "success": True
        }

        logger.info(
            f"Completed processing {company_id}: "
            f"{len(documents)} docs, {len(all_chunks)} chunks "
            f"in {processing_time:.2f}s"
        )

        return stats

    def load_company_documents(self, company_id: str) -> List[Dict[str, Any]]:
        """
        Load all documents for a company from file system.

        Args:
            company_id: Company identifier

        Returns:
            List of document dictionaries
        """
        company_dir = self.knowledge_base_dir / company_id
        documents = []

        # Scan directory for supported file types
        for file_path in company_dir.glob("**/*"):
            if file_path.is_file() and file_path.name != "metadata.json":
                try:
                    doc = self.parse_document(str(file_path), company_id)
                    if doc:
                        documents.append(doc)
                except Exception as e:
                    logger.error(f"Error parsing {file_path}: {e}")

        return documents

    def parse_document(self, file_path: str, company_id: str) -> Optional[Dict[str, Any]]:
        """
        Parse a document file into structured format.

        Args:
            file_path: Path to document file
            company_id: Company identifier

        Returns:
            Parsed document dictionary or None if unsupported
        """
        path = Path(file_path)
        suffix = path.suffix.lower()

        if suffix == ".json":
            return self.parse_json_document(file_path, company_id)
        elif suffix == ".txt":
            return self.parse_text_document(file_path, company_id)
        else:
            logger.warning(f"Unsupported file type: {suffix} for {file_path}")
            return None

    def parse_json_document(self, file_path: str, company_id: str) -> Dict[str, Any]:
        """
        Parse JSON document.

        Args:
            file_path: Path to JSON file
            company_id: Company identifier

        Returns:
            Parsed document dictionary
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        path = Path(file_path)
        filename = path.name

        # Determine document type from filename
        if "faq" in filename.lower():
            doc_type = DocumentType.FAQ
            category = "faq"
        elif "product" in filename.lower() or "service" in filename.lower() or "course" in filename.lower():
            doc_type = DocumentType.PRODUCT
            category = "products"
        else:
            doc_type = DocumentType.GENERAL
            category = "general"

        return {
            "file_path": file_path,
            "filename": filename,
            "data": data,
            "document_type": doc_type,
            "category": category,
            "company_id": company_id,
            "format": "json"
        }

    def parse_text_document(self, file_path: str, company_id: str) -> Dict[str, Any]:
        """
        Parse text document.

        Args:
            file_path: Path to text file
            company_id: Company identifier

        Returns:
            Parsed document dictionary
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        path = Path(file_path)
        filename = path.name

        # Determine document type from filename
        if "policy" in filename.lower() or "policies" in filename.lower():
            doc_type = DocumentType.POLICY
            category = "policies"
            priority = Priority.HIGH
        else:
            doc_type = DocumentType.GENERAL
            category = "general"
            priority = Priority.MEDIUM

        return {
            "file_path": file_path,
            "filename": filename,
            "content": content,
            "document_type": doc_type,
            "category": category,
            "company_id": company_id,
            "priority": priority,
            "format": "text"
        }

    def process_document(
        self,
        document: Dict[str, Any],
        company_id: str,
        company_metadata: Dict[str, Any]
    ) -> List[DocumentChunk]:
        """
        Process a single document into chunks.

        Args:
            document: Document dictionary
            company_id: Company identifier
            company_metadata: Company metadata

        Returns:
            List of DocumentChunk objects
        """
        doc_id = generate_document_id(document['file_path'])

        # Build metadata for chunks
        metadata = {
            "company_id": company_id,
            "document_id": doc_id,
            "document_type": document.get("document_type", DocumentType.GENERAL),
            "category": document.get("category", "general"),
            "source_file": document.get("filename", "unknown"),
            "priority": document.get("priority", Priority.MEDIUM)
        }

        # Process based on format
        if document.get("format") == "json":
            return self.process_json_document(document, metadata)
        elif document.get("format") == "text":
            return self.process_text_document(document, metadata)
        else:
            logger.warning(f"Unknown document format: {document.get('format')}")
            return []

    def process_json_document(
        self,
        document: Dict[str, Any],
        metadata: Dict[str, Any]
    ) -> List[DocumentChunk]:
        """
        Process JSON document into chunks.

        Args:
            document: Document dictionary
            metadata: Base metadata

        Returns:
            List of DocumentChunk objects
        """
        data = document['data']

        # Check if it's FAQ format
        if 'faqs' in data:
            return self.chunking_service.chunk_faq(data['faqs'], metadata)

        # Check if it's products/services/courses format
        for key in ['products', 'services', 'courses']:
            if key in data:
                chunks = []
                for idx, item in enumerate(data[key]):
                    # Each item becomes a chunk
                    text = f"{item.get('name', 'Unknown')}\n\n{item.get('description', '')}"
                    item_metadata = metadata.copy()
                    item_metadata['category'] = item.get('category', metadata['category'])

                    chunk_chunks = self.chunking_service.chunk_document(text, item_metadata)
                    chunks.extend(chunk_chunks)

                return chunks

        # Generic JSON processing
        return self.chunking_service.chunk_json_document(data, metadata)

    def process_text_document(
        self,
        document: Dict[str, Any],
        metadata: Dict[str, Any]
    ) -> List[DocumentChunk]:
        """
        Process text document into chunks.

        Args:
            document: Document dictionary
            metadata: Base metadata

        Returns:
            List of DocumentChunk objects
        """
        content = document['content']
        return self.chunking_service.chunk_document(content, metadata)

    def get_processing_stats(self, company_id: str) -> Dict[str, Any]:
        """
        Get processing statistics for a company.

        Args:
            company_id: Company identifier

        Returns:
            Statistics dictionary
        """
        company_dir = self.knowledge_base_dir / company_id

        if not company_dir.exists():
            return {
                "company_id": company_id,
                "exists": False
            }

        # Count files
        file_count = len(list(company_dir.glob("**/*.[jt][xs][ot][n]*")))

        # Get vector store stats
        vs_stats = self.vector_store.get_stats()
        company_chunks = vs_stats.get('companies', {}).get(company_id, 0)

        return {
            "company_id": company_id,
            "exists": True,
            "total_files": file_count,
            "indexed_chunks": company_chunks,
            "directory": str(company_dir)
        }
