from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional
import time

class QueryRequest(BaseModel):
    """Request model for natural language queries"""
    question: str = Field(..., description="Natural language question about the data")
    execute: bool = Field(True, description="Whether to execute the generated SQL query")
    limit: int = Field(100, ge=1, le=1000, description="Maximum number of rows to return")
    include_summary: bool = Field(True, description="Whether to generate AI summary and insights")
    context: str = Field("", description="Optional additional context for the question")

class SQLGenerationRequest(BaseModel):
    """Request model for SQL generation"""
    question: str = Field(..., description="Natural language question")
    context: str = Field("", description="Optional additional context")

class SQLExecutionRequest(BaseModel):
    """Request model for SQL execution"""
    sql_query: str = Field(..., description="SQL query to execute")
    limit: int = Field(100, ge=1, le=1000, description="Maximum number of rows to return")
    timeout: int = Field(30, ge=5, le=300, description="Query timeout in seconds")

class SummarizationRequest(BaseModel):
    """Request model for data summarization"""
    raw_results: str = Field(..., description="Raw query results")
    formatted_results: Dict[str, Any] = Field(..., description="Formatted query results")
    sql_query: str = Field(..., description="Original SQL query")
    question: str = Field(..., description="Original user question")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Query execution metadata")

class APIResponse(BaseModel):
    """Standard API response model"""
    success: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    timestamp: str = Field(default_factory=lambda: time.strftime("%Y-%m-%d %H:%M:%S UTC"))
