from pydantic import BaseModel
from typing import List, Dict, Any

class SchemaAnalysisResult(BaseModel):
    """Pydantic model for schema analysis results"""
    relevant_tables: List[str]
    relationships: Dict[str, Any]
    business_context: Dict[str, Any]
    key_metrics: List[str]
    join_strategy: Dict[str, Any]
    optimized_schema: str
    performance_hints: List[str]
    confidence_score: float
