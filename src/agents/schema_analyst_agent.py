"""
Generic Schema Analyst Agent - Analyzes database schema and provides intelligent context
Fully configurable and reusable for any database project
"""

import time
import asyncio
from datetime import datetime
from typing import Dict, Any, List, Optional, Protocol
from dataclasses import dataclass, field

from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.embedding_generator_base import EmbeddingGeneratorBase

from agents.base_agent import BaseAgent
from services.schema_analysis_cache_service import SchemaAnalysisCache
from Models.schema_analysis_result import SchemaAnalysisResult


class SchemaProviderProtocol(Protocol):
    """Protocol for schema providers to ensure compatibility"""
    async def get_tables(self) -> List[str]:
        """Get list of table names"""
        ...
    
    async def get_table_schema(self, table_name: str) -> Dict[str, Any]:
        """Get schema for a specific table"""
        ...
    
    async def get_relationships(self) -> Dict[str, Any]:
        """Get table relationships"""
        ...


@dataclass
class SchemaAnalystConfig:
    """Configuration for Schema Analyst Agent"""
    # Cache settings
    use_cache: bool = True
    cache_max_size: int = 100
    cache_max_age_hours: int = 24
    similarity_threshold: float = 0.85
    
    # Analysis settings
    max_relevant_tables: int = 5
    include_relationships: bool = True
    include_performance_hints: bool = True
    
    # Schema settings
    schema_prefix: str = ""
    default_schema: str = "public"
    
    # Business context (optional)
    domain: str = "Generic Database"
    business_context: Dict[str, Any] = field(default_factory=dict)


class GenericSchemaAnalystAgent(BaseAgent):
    """
    Generic Schema Analyst Agent for any database project
    
    Features:
    - Configurable for any database schema
    - Pluggable schema providers
    - Intelligent caching with semantic similarity
    - Extensible analysis pipeline
    """
    
    def __init__(
        self, 
        kernel: Kernel, 
        schema_provider: SchemaProviderProtocol,
        config: Optional[SchemaAnalystConfig] = None,
        name: str = "SchemaAnalystAgent"
    ):
        super().__init__(kernel, name)
        self.schema_provider = schema_provider
        self.config = config or SchemaAnalystConfig()

        # Initialize cache if enabled
        self.cache_service = SchemaAnalysisCache() if self.config.use_cache else None

        # Get embedding service from kernel
        self.embedding_service = self._get_embedding_service()

        # Schema metadata cache
        self._schema_metadata: Optional[Dict[str, Any]] = None

        print(f"🔍 {name} initialized")
        if self.embedding_service and self.config.use_cache:
            print("✅ Semantic caching enabled")
    
    def _get_embedding_service(self) -> Optional[EmbeddingGeneratorBase]:
        """Get embedding service from kernel"""
        try:
            for service in self.kernel.services.values():
                if isinstance(service, EmbeddingGeneratorBase):
                    return service
            return None
        except Exception:
            return None
    
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process schema analysis request
        
        Args:
            input_data: {
                "question": str,
                "context": Optional[str],
                "analysis_type": Optional[str] - "tables", "relationships", "full"
                "options": Optional[Dict[str, Any]]
            }
        """
        try:
            question = input_data.get("question", "")
            context = input_data.get("context", "")
            analysis_type = input_data.get("analysis_type", "full")
            options = input_data.get("options", {})
            
            if not question:
                return self._create_result(
                    success=False,
                    error="No question provided for schema analysis"
                )
            
            # Check cache if enabled
            if self.config.use_cache and self.cache_service:
                cached_result = await self._check_cache(question, context)
                if cached_result:
                    return cached_result
            
            # Perform analysis
            analysis_start = time.time()
            analysis_result = await self._perform_analysis(
                question, context, analysis_type, options
            )
            analysis_time = time.time() - analysis_start
            
            # Cache result if enabled
            if self.config.use_cache and self.cache_service:
                await self.cache_service.store_analysis(
                    question, context, analysis_result.__dict__, self.embedding_service
                )
            
            return self._create_result(
                success=True,
                data=analysis_result.__dict__,
                metadata={
                    "analysis_time": round(analysis_time, 3),
                    "cache_hit": False,
                    "analysis_type": analysis_type
                }
            )
            
        except Exception as e:
            return self._create_result(
                success=False,
                error=f"Schema analysis failed: {str(e)}"
            )
    
    async def _check_cache(self, question: str, context: str) -> Optional[Dict[str, Any]]:
        """Check cache for existing analysis"""
        # Try cache (exact or semantic)
        result = await self.cache_service.get_analysis(question, context, self.embedding_service)
        if result:
            return self._create_result(
                success=True,
                data=result,
                metadata={"cache_hit": True, "cache_type": "analysis"}
            )

        # Try semantic match
        if self.embedding_service:
            semantic_match = await self.cache_service.get_semantic_match(
                question, context, self.embedding_service, self.config.similarity_threshold
            )
            if semantic_match:
                return self._create_result(
                    success=True,
                    data=semantic_match["data"],
                    metadata={
                        "cache_hit": True,
                        "cache_type": "semantic",
                        "similarity": semantic_match.get("similarity", 0)
                    }
                )

        return None
    
    async def _perform_analysis(
        self, 
        question: str, 
        context: str,
        analysis_type: str,
        options: Dict[str, Any]
    ) -> SchemaAnalysisResult:
        """Perform schema analysis based on type"""
        
        # Ensure schema metadata is loaded
        if not self._schema_metadata:
            await self._load_schema_metadata()
        
        # Run analysis components
        tasks = []
        
        if analysis_type in ["tables", "full"]:
            tasks.append(self._identify_relevant_tables(question, context, options))
        
        if analysis_type in ["relationships", "full"] and self.config.include_relationships:
            tasks.append(self._analyze_relationships(question, options))
        
        if analysis_type == "full":
            tasks.extend([
                self._extract_context(question, options),
                self._identify_metrics(question, options)
            ])
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        relevant_tables = results[0] if len(results) > 0 and not isinstance(results[0], Exception) else []
        relationships = results[1] if len(results) > 1 and not isinstance(results[1], Exception) else {}
        business_context = results[2] if len(results) > 2 and not isinstance(results[2], Exception) else {}
        key_metrics = results[3] if len(results) > 3 and not isinstance(results[3], Exception) else []
        
        # Generate additional insights
        join_strategy = self._generate_join_strategy(relevant_tables, relationships) if relationships else {}
        optimized_schema = await self._build_optimized_schema(relevant_tables)
        performance_hints = self._generate_performance_hints(relevant_tables, question) if self.config.include_performance_hints else []
        confidence_score = self._calculate_confidence(len(relevant_tables), bool(relationships))
        
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
    
    async def _load_schema_metadata(self):
        """Load schema metadata from provider"""
        tables = await self.schema_provider.get_tables()
        relationships = await self.schema_provider.get_relationships()
        
        self._schema_metadata = {
            "tables": tables,
            "relationships": relationships,
            "schema_prefix": self.config.schema_prefix,
            "default_schema": self.config.default_schema
        }
    
    async def _identify_relevant_tables(
        self, 
        question: str, 
        context: str,
        options: Dict[str, Any]
    ) -> List[str]:
        """Identify relevant tables using configurable strategies"""
        # This is a simplified version - extend with ML/NLP if needed
        question_lower = question.lower()
        context_lower = context.lower()
        combined = f"{question_lower} {context_lower}"
        
        relevant_tables = []
        
        # Simple keyword matching - override this method for custom logic
        for table in self._schema_metadata.get("tables", []):
            if table.lower() in combined:
                relevant_tables.append(table)
        
        # Limit to max configured tables
        return relevant_tables[:self.config.max_relevant_tables]
    
    async def _analyze_relationships(
        self, 
        question: str,
        options: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze relationships from schema provider"""
        return self._schema_metadata.get("relationships", {})
    
    async def _extract_context(
        self, 
        question: str,
        options: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Extract business context - override for custom logic"""
        return {
            "domain": self.config.domain,
            "query_intent": self._detect_query_intent(question),
            "custom_context": self.config.business_context
        }
    
    async def _identify_metrics(
        self, 
        question: str,
        options: Dict[str, Any]
    ) -> List[str]:
        """Identify metrics mentioned in question - override for custom logic"""
        # Simple implementation - extend as needed
        metric_keywords = ["count", "sum", "average", "total", "revenue", "sales"]
        question_lower = question.lower()
        
        return [kw for kw in metric_keywords if kw in question_lower]
    
    def _detect_query_intent(self, question: str) -> str:
        """Detect query intent from question"""
        question_lower = question.lower()
        
        intents = {
            "ranking": ["top", "best", "highest", "lowest", "rank"],
            "aggregation": ["sum", "total", "count", "average"],
            "comparison": ["compare", "versus", "vs", "difference"],
            "temporal": ["trend", "over time", "growth", "change"],
            "filtering": ["where", "filter", "only", "specific"]
        }
        
        for intent, keywords in intents.items():
            if any(kw in question_lower for kw in keywords):
                return intent
        
        return "general"
    
    def _generate_join_strategy(
        self, 
        tables: List[str], 
        relationships: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate join strategy based on relationships"""
        if len(tables) < 2:
            return {"strategy": "single_table", "joins": []}
        
        # Simple strategy - extend for complex scenarios
        return {
            "strategy": "multi_table",
            "primary_table": tables[0],
            "joins": self._build_join_path(tables, relationships)
        }
    
    def _build_join_path(
        self, 
        tables: List[str], 
        relationships: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Build join path between tables"""
        joins = []
        
        # Simple implementation - extend with graph algorithms for complex schemas
        for i in range(len(tables) - 1):
            from_table = tables[i]
            to_table = tables[i + 1]
            
            # Check if direct relationship exists
            if from_table in relationships:
                for rel in relationships[from_table]:
                    if rel.get("table") == to_table:
                        joins.append({
                            "from": from_table,
                            "to": to_table,
                            "type": rel.get("type", "INNER JOIN"),
                            "condition": rel.get("key", "")
                        })
                        break
        
        return joins
    
    async def _build_optimized_schema(self, tables: List[str]) -> str:
        """Build optimized schema context for relevant tables"""
        if not tables:
            return "No specific tables identified"
        
        schema_parts = []
        
        for table in tables:
            try:
                table_schema = await self.schema_provider.get_table_schema(table)
                schema_parts.append(f"Table: {self._format_table_name(table)}")
                schema_parts.append(str(table_schema))
                schema_parts.append("")
            except Exception as e:
                schema_parts.append(f"Error loading schema for {table}: {str(e)}")
        
        return "\n".join(schema_parts)
    
    def _format_table_name(self, table: str) -> str:
        """Format table name with schema prefix"""
        if self.config.schema_prefix:
            return f"{self.config.schema_prefix}.{table}"
        return table
    
    def _generate_performance_hints(
        self, 
        tables: List[str], 
        question: str
    ) -> List[str]:
        """Generate performance hints - override for database-specific hints"""
        hints = []
        
        if len(tables) > 3:
            hints.append("Consider query optimization for multiple table joins")
        
        question_lower = question.lower()
        if any(word in question_lower for word in ["top", "limit", "first"]):
            hints.append("Use appropriate limiting clauses for better performance")
        
        if any(word in question_lower for word in ["sum", "total", "count"]):
            hints.append("Ensure proper indexing on aggregation columns")
        
        return hints
    
    def _calculate_confidence(self, table_count: int, has_relationships: bool) -> float:
        """Calculate confidence score for analysis"""
        score = 0.5  # Base score
        
        if table_count > 0:
            score += min(0.3, table_count * 0.1)
        
        if has_relationships:
            score += 0.2
        
        return min(score, 1.0)
    
    async def get_statistics(self) -> Dict[str, Any]:
        """Get agent statistics"""
        stats = {"agent": self.agent_name}
        if self.cache_service:
            stats["cache"] = self.cache_service.get_info()
        return stats
    
    def _create_result(
        self, 
        success: bool, 
        data: Dict[str, Any] = None,
        error: str = None,
        metadata: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Create standardized result"""
        return {
            "success": success,
            "data": data or {},
            "error": error,
            "metadata": metadata or {},
            "timestamp": datetime.now().isoformat()
        }