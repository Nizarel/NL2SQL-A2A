"""
Schema Analyst Agent - Analyzes database schema and provides intelligent context for SQL generation
Modular design using SchemaService and InMemoryCache for better architecture
"""

import time
import asyncio
from datetime import datetime
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.embedding_generator_base import EmbeddingGeneratorBase

from agents.base_agent import BaseAgent
from services.schema_service import SchemaService
from services.schema_analysis_cache_service import InMemorySchemaCache


@dataclass
class SchemaAnalysisResult:
    """Data class for schema analysis results"""
    relevant_tables: List[str]
    relationships: Dict[str, Any]
    business_context: Dict[str, Any]
    key_metrics: List[str]
    join_strategy: Dict[str, Any]
    optimized_schema: str
    performance_hints: List[str]
    confidence_score: float


class SchemaAnalystAgent(BaseAgent):
    """
    Agent responsible for analyzing database schema and providing intelligent context for SQL generation
    
    Goals:
    1. Analyze and understand database structure, table relationships, and constraints
    2. Provide clear, comprehensive schema context for SQL generation
    3. Identify and explain foreign key relationships and join possibilities
    4. Understand data types, constraints, and business logic embedded in schema
    5. Cache analysis results using semantic similarity for performance
    """
    
    def __init__(self, kernel: Kernel, schema_service: SchemaService):
        super().__init__(kernel, "SchemaAnalystAgent")
        self.schema_service = schema_service
        
        # Initialize modular cache service
        self.cache_service = InMemorySchemaCache()
        
        # Get embedding service from kernel
        self.embedding_service = self._get_embedding_service()
        
        print("🔍 Schema Analyst Agent initialized with modular cache service")
        if self.embedding_service:
            print("✅ Embedding service available for semantic caching")
        else:
            print("⚠️ No embedding service - using exact matching only")
    
    def _get_embedding_service(self) -> Optional[EmbeddingGeneratorBase]:
        """Get embedding service from kernel"""
        try:
            # Look for embedding service in kernel
            for service in self.kernel.services.values():
                if isinstance(service, EmbeddingGeneratorBase):
                    return service
            return None
        except Exception as e:
            print(f"⚠️ No embedding service found: {e}")
            return None
    
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze schema context for a given question using modular cache service
        
        Args:
            input_data: Dictionary containing:
                - question: Natural language question
                - context: Optional additional context
                - use_cache: Whether to use cached analysis (default: True)
                - similarity_threshold: Threshold for semantic similarity (default: 0.85)
                
        Returns:
            Dictionary containing schema analysis results
        """
        try:
            question = input_data.get("question", "")
            context = input_data.get("context", "")
            use_cache = input_data.get("use_cache", True)
            similarity_threshold = input_data.get("similarity_threshold", 0.85)
            
            if not question:
                return self._create_result(
                    success=False,
                    error="No question provided for schema analysis"
                )
            
            print(f"🔍 Schema Analysis for: {question[:50]}...")
            analysis_start = time.time()
            
            # Try cache lookup first
            if use_cache:
                # Try exact match first
                exact_match = await self.cache_service.get_exact_match(question, context)
                if exact_match:
                    return self._create_result(
                        success=True,
                        data=exact_match,
                        metadata={
                            "cache_hit": True,
                            "cache_type": "exact",
                            "analysis_time": 0
                        }
                    )
                
                # Try semantic match if embedding service available
                if self.embedding_service:
                    semantic_match = await self.cache_service.get_semantic_match(
                        question, context, self.embedding_service, similarity_threshold
                    )
                    if semantic_match:
                        return self._create_result(
                            success=True,
                            data=semantic_match["data"],
                            metadata={
                                "cache_hit": True,
                                "cache_type": "semantic",
                                "semantic_similarity": semantic_match.get("similarity", 0),
                                "analysis_time": 0
                            }
                        )
            
            # Perform fresh analysis
            analysis_result = await self._perform_schema_analysis(question, context)
            analysis_time = time.time() - analysis_start
            
            # Cache the result
            if use_cache:
                await self.cache_service.store_analysis(
                    question, context, analysis_result.__dict__, self.embedding_service
                )
            
            return self._create_result(
                success=True,
                data=analysis_result.__dict__,
                metadata={
                    "analysis_time": round(analysis_time, 3),
                    "cache_hit": False,
                    "confidence_score": analysis_result.confidence_score
                }
            )
            
        except Exception as e:
            return self._create_result(
                success=False,
                error=f"Schema analysis failed: {str(e)}"
            )
    
    async def _perform_schema_analysis(self, question: str, context: str = "") -> SchemaAnalysisResult:
        """Perform comprehensive schema analysis"""
        
        # Run analysis tasks concurrently for performance
        analysis_tasks = await asyncio.gather(
            self._identify_relevant_tables(question, context),
            self._analyze_table_relationships(question),
            self._extract_business_context(question),
            self._identify_key_metrics(question),
            return_exceptions=True
        )
        
        # Process results
        relevant_tables = analysis_tasks[0] if not isinstance(analysis_tasks[0], Exception) else []
        relationships = analysis_tasks[1] if not isinstance(analysis_tasks[1], Exception) else {}
        business_context = analysis_tasks[2] if not isinstance(analysis_tasks[2], Exception) else {}
        key_metrics = analysis_tasks[3] if not isinstance(analysis_tasks[3], Exception) else []
        
        # Generate derived analysis
        join_strategy = self._recommend_join_strategy(relevant_tables, relationships)
        optimized_schema = self._build_optimized_schema_context(relevant_tables)
        performance_hints = self._generate_performance_hints(relevant_tables, question)
        confidence_score = self._calculate_confidence_score(relevant_tables, relationships, business_context)
        
        return SchemaAnalysisResult(
            relevant_tables=relevant_tables,
            relationships=relationships,
            business_context=business_context,
            key_metrics=key_metrics,
            join_strategy=join_strategy,
            optimized_schema=optimized_schema,
            performance_hints=performance_hints,
            confidence_score=confidence_score
        )
    
    async def _identify_relevant_tables(self, question: str, context: str = "") -> List[str]:
        """Identify tables relevant to the question using schema service"""
        return self.schema_service.identify_relevant_tables(question, context)
    
    async def _analyze_table_relationships(self, question: str) -> Dict[str, Any]:
        """Analyze table relationships relevant to the question"""
        # Delegate to schema service instead of duplicating relationship logic
        return self.schema_service.relationships
    
    async def _extract_business_context(self, question: str) -> Dict[str, Any]:
        """Extract business context relevant to the question using schema service"""
        return self.schema_service.extract_business_context(question)
    
    async def _identify_key_metrics(self, question: str) -> List[str]:
        """Identify key metrics mentioned in the question using schema service"""
        return self.schema_service.identify_key_metrics(question)
    
    def _recommend_join_strategy(self, tables: List[str], relationships: Dict[str, Any]) -> Dict[str, Any]:
        """Recommend optimal join strategy for the tables"""
        if not tables or len(tables) < 2:
            return {
                "strategy": "single_table",
                "primary_table": tables[0] if tables else "segmentacion",
                "joins": [],
                "estimated_performance": "optimal"
            }
        
        # Determine if this is a star schema query (fact table + dimensions)
        has_fact_table = "segmentacion" in tables
        
        strategy = {
            "strategy": "star_schema" if has_fact_table else "dimension_lookup",
            "primary_table": "segmentacion" if has_fact_table else tables[0],
            "joins": [],
            "join_order": [],
            "estimated_performance": "good"
        }
        
        # Build join recommendations for star schema
        if has_fact_table:
            join_order = ["segmentacion"]  # Start with fact table
            
            # Add dimension table joins
            if "cliente" in tables:
                strategy["joins"].append({
                    "type": "INNER JOIN",
                    "table": "dev.cliente",
                    "condition": "segmentacion.customer_id = cliente.customer_id",
                    "purpose": "Customer details"
                })
                join_order.append("cliente")
            
            if "producto" in tables:
                strategy["joins"].append({
                    "type": "INNER JOIN",
                    "table": "dev.producto",
                    "condition": "segmentacion.material_id = producto.Material",
                    "purpose": "Product details"
                })
                join_order.append("producto")
            
            if "tiempo" in tables:
                strategy["joins"].append({
                    "type": "INNER JOIN",
                    "table": "dev.tiempo",
                    "condition": "segmentacion.calday = tiempo.Fecha",
                    "purpose": "Date dimensions"
                })
                join_order.append("tiempo")
        
        return strategy
    
    def _build_optimized_schema_context(self, relevant_tables: List[str]) -> str:
        """Build optimized schema context for only relevant tables"""
        if not relevant_tables:
            return self.schema_service.get_full_schema_summary()
        
        # Use schema service method instead of duplicating logic
        return self.schema_service.get_schema_for_query(relevant_tables)
    
    def _generate_performance_hints(self, tables: List[str], question: str) -> List[str]:
        """Generate performance optimization hints"""
        hints = []
        question_lower = question.lower()
        
        # Table-specific hints
        if "segmentacion" in tables:
            hints.append("segmentacion is the main fact table - use it as the primary table for joins")
            hints.append("Consider date filters on segmentacion.calday for time-based queries")
        
        if len(tables) > 3:
            hints.append("Multiple table joins detected - consider query optimization")
            hints.append("Use appropriate indexes on join columns for better performance")
        
        # Query pattern hints
        if any(word in question_lower for word in ["top", "limit", "first"]):
            hints.append("Use TOP clause for limiting results in SQL Server")
        
        if any(word in question_lower for word in ["sum", "total", "aggregate", "count"]):
            hints.append("Use appropriate GROUP BY clauses for aggregations")
            hints.append("Consider using HAVING clause for aggregate filtering")
        
        return hints
    
    def _calculate_confidence_score(self, tables: List[str], relationships: Dict[str, Any], 
                                   business_context: Dict[str, Any]) -> float:
        """Calculate confidence score for the analysis"""
        score = 0.5  # Base score
        
        # Table relevance scoring
        if tables:
            score += 0.2
            if "segmentacion" in tables:  # Has fact table
                score += 0.1
            if len(tables) >= 2:  # Multiple tables for richer analysis
                score += 0.1
        
        # Relationship scoring
        if relationships:
            score += 0.1
        
        # Business context scoring
        if business_context.get("query_type") != "general":
            score += 0.1
        
        return min(score, 1.0)
    
    async def get_cache_statistics(self) -> Dict[str, Any]:
        """Get cache statistics for monitoring"""
        return self.cache_service.get_statistics()
    
    def clear_all_caches(self):
        """Clear all caches"""
        self.cache_service.clear_cache()
        print("🧹 All caches cleared")
    
    def _create_result(self, success: bool, data: Dict[str, Any] = None, 
                      error: str = None, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """Create standardized result dictionary"""
        return {
            "success": success,
            "data": data or {},
            "error": error,
            "metadata": metadata or {},
            "timestamp": datetime.now().isoformat()
        }