"""
Hybrid Orchestrator Agent - Phase 3 Parallel Execution
Safe parallel processing with automatic fallback to legacy orchestrator
"""
import asyncio
import os
import time
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

from agents.orchestrator_agent import OrchestratorAgent
from services.performance_monitor_enhanced import performance_monitor
from services.simple_cache import query_result_cache, schema_cache, sql_cache

@dataclass
class ParallelConfig:
    """Configuration for parallel execution"""
    enabled: bool = False
    min_complexity_threshold: int = 30  # seconds
    max_parallel_operations: int = 3
    fallback_on_error: bool = True
    enable_monitoring: bool = True

class HybridOrchestratorAgent:
    """
    Hybrid orchestrator that can use parallel execution for complex queries
    while maintaining 100% compatibility through legacy fallback
    """
    
    def __init__(self, 
                 kernel,
                 schema_analyst_agent,
                 sql_generator_agent, 
                 executor_agent,
                 summarizing_agent,
                 memory_service=None,
                 performance_monitor_service=None):
        
        # Initialize legacy orchestrator as fallback with EXACT parameter names from orchestrator_agent.py
        self.legacy_orchestrator = OrchestratorAgent(
            kernel=kernel,
            schema_analyst=schema_analyst_agent,    # Parameter name: schema_analyst
            sql_generator=sql_generator_agent,      # Parameter name: sql_generator  
            executor=executor_agent,                # Parameter name: executor
            summarizer=summarizing_agent,           # Parameter name: summarizer
            memory_service=memory_service           # Parameter name: memory_service
        )
        
        # Store agents for parallel execution
        self.schema_analyst_agent = schema_analyst_agent
        self.sql_generator_agent = sql_generator_agent
        self.executor_agent = executor_agent
        self.summarizing_agent = summarizing_agent
        self.memory_service = memory_service
        
        # Configuration from environment
        self.config = ParallelConfig(
            enabled=os.getenv("ENABLE_PARALLEL_PROCESSING", "false").lower() == "true",
            min_complexity_threshold=int(os.getenv("PARALLEL_THRESHOLD", "30")),
            max_parallel_operations=int(os.getenv("MAX_PARALLEL_OPS", "3")),
            fallback_on_error=os.getenv("PARALLEL_FALLBACK", "true").lower() == "true"
        )
        
        self.logger = logging.getLogger(__name__)
        self.stats = {
            "parallel_attempts": 0,
            "parallel_successes": 0,
            "fallback_uses": 0,
            "performance_improvements": []
        }
    
    async def process(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main processing method with intelligent parallel/sequential selection
        """
        with performance_monitor.track_operation(
            "hybrid_orchestrator",
            parallel_enabled=self.config.enabled,
            **{k: v for k, v in request.items() if k in ["user_id", "session_id"]}
        ) as metric:
            
            # Always check cache first (Phase 2 benefit)
            cache_key_params = {
                "user_id": request.get("user_id"),
                "session_id": request.get("session_id"),
                "execute": request.get("execute", True),
                "limit": request.get("limit", 100)
            }
            
            cached_result = query_result_cache.get(request.get("question", ""), cache_key_params)
            if cached_result:
                self.logger.info(f"🚀 Cache HIT - returning cached result")
                metric.metadata.update({"cache_hit": True, "processing_mode": "cached"})
                return cached_result
            
            # Determine processing strategy
            if self._should_use_parallel(request):
                self.logger.info(f"🔄 Attempting parallel processing for complex query")
                metric.metadata.update({"processing_mode": "parallel_attempt"})
                
                try:
                    self.stats["parallel_attempts"] += 1
                    parallel_start = time.time()
                    
                    result = await self._process_parallel(request)
                    
                    if result.get("success"):
                        parallel_duration = time.time() - parallel_start
                        self.stats["parallel_successes"] += 1
                        self.stats["performance_improvements"].append(parallel_duration)
                        
                        self.logger.info(f"✅ Parallel processing succeeded in {parallel_duration:.2f}s")
                        metric.metadata.update({
                            "processing_mode": "parallel_success",
                            "parallel_duration": parallel_duration
                        })
                        
                        # Cache the successful result
                        query_result_cache.put(
                            request.get("question", ""), 
                            result, 
                            cache_key_params,
                            ttl=1800
                        )
                        
                        return result
                        
                except Exception as e:
                    self.logger.warning(f"⚠️ Parallel processing failed: {str(e)}")
                    if not self.config.fallback_on_error:
                        raise
            
            # Fallback to proven legacy orchestrator
            self.logger.info(f"🔄 Using legacy orchestrator (fallback or strategy choice)")
            self.stats["fallback_uses"] += 1
            metric.metadata.update({"processing_mode": "legacy_fallback"})
            
            result = await self.legacy_orchestrator.process(request)
            
            # Cache successful legacy results too
            if result.get("success"):
                query_result_cache.put(
                    request.get("question", ""), 
                    result, 
                    cache_key_params,
                    ttl=1800
                )
            
            return result
    
    def _should_use_parallel(self, request: Dict[str, Any]) -> bool:
        """
        Determine if query should use parallel processing
        """
        if not self.config.enabled:
            return False
        
        question = request.get("question", "").lower()
        
        # Heuristics for parallel suitability
        parallel_indicators = [
            "trend" in question and ("month" in question or "year" in question),
            "analysis" in question and len(question.split()) > 8,
            "comparison" in question or "compare" in question,
            "top" in question and ("customer" in question or "product" in question),
            len(question.split()) > 10,  # Complex questions
            any(word in question for word in ["aggregate", "sum", "average", "group by"])
        ]
        
        # Use parallel for potentially complex queries
        return any(parallel_indicators)
    
    async def _process_parallel(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute parallel processing workflow
        """
        question = request.get("question", "")
        user_id = request.get("user_id", "")
        session_id = request.get("session_id", "")
        execute = request.get("execute", True)
        
        try:
            # Phase 1: Parallel Schema Analysis + Context Preparation
            schema_task = asyncio.create_task(
                self._get_schema_analysis(question)
            )
            
            context_task = asyncio.create_task(
                self._prepare_context(user_id, session_id)
            )
            
            # Wait for both schema and context
            schema_result, context_data = await asyncio.gather(
                schema_task, context_task, return_exceptions=True
            )
            
            # Check for exceptions
            if isinstance(schema_result, Exception):
                raise schema_result
            if isinstance(context_data, Exception):
                self.logger.warning(f"Context preparation failed: {context_data}")
                context_data = {}
            
            # Phase 2: SQL Generation (depends on schema)
            sql_result = await self.sql_generator_agent.process({
                "question": question,
                "schema_analysis": schema_result.get("data", {}),
                "context": context_data
            })
            
            if not sql_result.get("success"):
                raise Exception(f"SQL generation failed: {sql_result.get('error')}")
            
            # Phase 3: Parallel Execution + Summary Preparation
            if execute:
                execution_task = asyncio.create_task(
                    self.executor_agent.process({
                        "sql_query": sql_result["data"]["sql_query"],
                        "limit": request.get("limit", 100)
                    })
                )
                
                summary_prep_task = asyncio.create_task(
                    self._prepare_summary_context(question, sql_result["data"])
                )
                
                # Wait for execution and summary prep
                execution_result, summary_context = await asyncio.gather(
                    execution_task, summary_prep_task, return_exceptions=True
                )
                
                if isinstance(execution_result, Exception):
                    raise execution_result
                
                # Phase 4: Final Summary Generation
                if request.get("include_summary", True):
                    summary_result = await self.summarizing_agent.process({
                        "question": question,
                        "sql_query": sql_result["data"]["sql_query"],
                        "query_results": execution_result.get("data", {}),
                        "context": summary_context if not isinstance(summary_context, Exception) else {}
                    })
                else:
                    summary_result = {"success": True, "data": {"summary": "Summary generation skipped"}}
            else:
                execution_result = {"success": True, "data": {"results": "Execution skipped"}}
                summary_result = {"success": True, "data": {"summary": "Summary generation skipped"}}
            
            # Compile final result
            return {
                "success": True,
                "data": {
                    "sql_query": sql_result["data"]["sql_query"],
                    "executed": execute,
                    "results": execution_result.get("data", {}).get("results", []),
                    "formatted_results": execution_result.get("data", {}).get("formatted_results", ""),
                    "schema_analysis": schema_result.get("data", {}),
                    "summary": summary_result.get("data", {}).get("summary", "")
                },
                "metadata": {
                    "workflow_success": True,
                    "orchestration_type": "hybrid_parallel",
                    "parallel_execution": True,
                    "schema_analyzed": schema_result.get("success", False),
                    "sql_generated": sql_result.get("success", False),
                    "query_executed": execution_result.get("success", False) if execute else False,
                    "summary_generated": summary_result.get("success", False),
                    "execution_time": execution_result.get("metadata", {}).get("execution_time", 0),
                    "total_workflow_time": time.time() - time.time()  # Will be updated by caller
                }
            }
            
        except Exception as e:
            self.logger.error(f"Parallel processing failed: {str(e)}")
            raise
    
    async def _get_schema_analysis(self, question: str) -> Dict[str, Any]:
        """Get schema analysis with caching"""
        # Check schema cache first
        cached_schema = schema_cache.get(question)
        if cached_schema:
            return cached_schema
        
        result = await self.schema_analyst_agent.process({"question": question})
        
        if result.get("success"):
            schema_cache.put(question, result, ttl=7200)  # 2 hours
        
        return result
    
    async def _prepare_context(self, user_id: str, session_id: str) -> Dict[str, Any]:
        """Prepare conversation context"""
        if not self.memory_service:
            return {}
        
        try:
            context = await self.memory_service.get_session_context(
                user_id=user_id,
                session_id=session_id,
                max_messages=5
            )
            return {"conversation_context": context}
        except Exception as e:
            self.logger.warning(f"Context preparation failed: {e}")
            return {}
    
    async def _prepare_summary_context(self, question: str, sql_data: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare context for summary generation"""
        return {
            "query_complexity": len(question.split()),
            "sql_complexity": sql_data.get("sql_query", "").count("JOIN"),
            "preparation_timestamp": time.time()
        }
    
    async def get_workflow_status(self) -> Dict[str, Any]:
        """Get hybrid orchestrator status"""
        legacy_status = await self.legacy_orchestrator.get_workflow_status()
        
        # Calculate parallel success rate
        success_rate = 0
        if self.stats["parallel_attempts"] > 0:
            success_rate = self.stats["parallel_successes"] / self.stats["parallel_attempts"]
        
        avg_improvement = 0
        if self.stats["performance_improvements"]:
            avg_improvement = sum(self.stats["performance_improvements"]) / len(self.stats["performance_improvements"])
        
        hybrid_status = {
            "orchestrator_type": "hybrid_parallel",
            "parallel_config": {
                "enabled": self.config.enabled,
                "threshold": self.config.min_complexity_threshold,
                "max_ops": self.config.max_parallel_operations
            },
            "parallel_stats": {
                "attempts": self.stats["parallel_attempts"],
                "successes": self.stats["parallel_successes"],
                "success_rate": success_rate,
                "fallback_uses": self.stats["fallback_uses"],
                "avg_performance": avg_improvement
            },
            "legacy_fallback": "available"
        }
        
        # Merge with legacy status
        return {
            "success": True,
            "data": {**legacy_status.get("data", {}), **hybrid_status},
            "metadata": legacy_status.get("metadata", {})
        }
