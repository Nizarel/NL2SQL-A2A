"""
Configuration management for NL2SQL Multi-Agent System
Centralized settings using Pydantic for validation and caching
"""

import os
from functools import lru_cache
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field

# Get the directory where this config.py file is located
_CONFIG_DIR = os.path.dirname(os.path.abspath(__file__))


class Settings(BaseSettings):
    """
    Application settings with environment variable integration
    """
    # Azure OpenAI Configuration
    azure_openai_endpoint: Optional[str] = Field(None, env="AZURE_OPENAI_ENDPOINT")
    azure_openai_api_key: Optional[str] = Field(None, env="AZURE_OPENAI_API_KEY")
    azure_openai_deployment_name: Optional[str] = Field(None, env="AZURE_OPENAI_DEPLOYMENT_NAME")
    azure_openai_mini_deployment_name: Optional[str] = Field(None, env="AZURE_OPENAI_MINI_DEPLOYMENT_NAME")
    azure_openai_api_version: str = Field("2024-12-01-preview", env="AZURE_OPENAI_API_VERSION")
    
    # OpenAI Configuration
    openai_api_key: Optional[str] = Field(None, env="OPENAI_API_KEY")
    openai_model: str = Field("gpt-4o", env="OPENAI_MODEL")
    
    # MCP Server Configuration
    mcp_server_url: str = Field(
        "https://azsql-fastmcpserv2.jollyfield-479bc951.eastus2.azurecontainerapps.io/mcp/",
        env="MCP_SERVER_URL"
    )
    
    # Performance Settings
    schema_cache_ttl: int = Field(3600, env="SCHEMA_CACHE_TTL")  # 1 hour in seconds
    query_timeout: int = Field(30, env="QUERY_TIMEOUT")  # 30 seconds
    max_result_rows: int = Field(1000, env="MAX_RESULT_ROWS")
    enable_query_cache: bool = Field(True, env="ENABLE_QUERY_CACHE")
    query_cache_ttl: int = Field(300, env="QUERY_CACHE_TTL")  # 5 minutes
    query_cache_size: int = Field(100, env="QUERY_CACHE_SIZE")
    
    # Agent Settings
    max_tokens_intent: int = Field(500, env="MAX_TOKENS_INTENT")
    max_tokens_sql: int = Field(800, env="MAX_TOKENS_SQL")
    max_tokens_summary: int = Field(1500, env="MAX_TOKENS_SUMMARY")
    temperature: float = Field(0.1, env="TEMPERATURE")
    
    # API Settings
    enable_compression: bool = Field(True, env="ENABLE_COMPRESSION")
    compression_min_size: int = Field(1000, env="COMPRESSION_MIN_SIZE")
    max_response_size: int = Field(10000, env="MAX_RESPONSE_SIZE")  # Characters
    
    class Config:
        env_file = os.path.join(_CONFIG_DIR, ".env")  # Look for .env in same directory as config.py
        env_file_encoding = "utf-8"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached application settings
    This function is cached to avoid re-reading environment variables
    """
    return Settings()


def reload_settings():
    """
    Force reload of settings (useful for testing or dynamic config changes)
    """
    get_settings.cache_clear()
    return get_settings()
