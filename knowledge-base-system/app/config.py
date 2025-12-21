"""
Configuration management for the Knowledge Base System.
Loads settings from environment variables with sensible defaults.
"""

from typing import Optional
from pydantic_settings import BaseSettings
from functools import lru_cache
import os
from pathlib import Path


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Application
    app_name: str = "Knowledge Base System"
    app_version: str = "1.0.0"
    debug: bool = False
    log_level: str = "INFO"

    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 4

    # Embedding Configuration
    embedding_model_name: str = "all-MiniLM-L6-v2"
    embedding_dimension: int = 384
    embedding_batch_size: int = 32

    # Vector Database
    vector_db_type: str = "faiss"
    index_dir: str = "./indexes"

    # Chunking Configuration
    chunk_size_min: int = 200
    chunk_size_target: int = 450
    chunk_size_max: int = 600
    chunk_overlap: int = 75

    # Retrieval Configuration
    top_k_results: int = 5
    similarity_threshold: float = 0.3
    rerank_enabled: bool = True

    # Cache Configuration
    cache_enabled: bool = True
    cache_ttl: int = 3600
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0

    # Paths
    knowledge_base_dir: str = "./knowledge_bases"
    cache_dir: str = "./cache"
    log_dir: str = "./logs"

    # Default Company
    default_company_id: Optional[str] = "techstore"

    # Performance
    max_concurrent_requests: int = 50
    query_timeout: int = 30
    gpu_enabled: bool = False

    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = "ignore"

    def get_company_dir(self, company_id: str) -> Path:
        """Get the directory path for a specific company."""
        return Path(self.knowledge_base_dir) / company_id

    def get_index_path(self, company_id: str) -> Path:
        """Get the index file path for a specific company."""
        return Path(self.index_dir) / f"{company_id}.index"

    def get_metadata_path(self, company_id: str) -> Path:
        """Get the metadata file path for a specific company."""
        return Path(self.index_dir) / f"{company_id}_metadata.json"

    def ensure_directories(self):
        """Create necessary directories if they don't exist."""
        directories = [
            self.knowledge_base_dir,
            self.index_dir,
            self.cache_dir,
            self.log_dir
        ]
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    settings = Settings()
    settings.ensure_directories()
    return settings


# Global settings instance
settings = get_settings()
