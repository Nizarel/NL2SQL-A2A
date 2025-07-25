"""
Optimized orchestrator agent with parallel execution and unified interface
"""

import asyncio
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import semantic_kernel as sk

try:
    # Try relative imports first (when used as module)
    from .agent_interface import AgentMessage, WorkflowStage, WorkflowConfig
    from .optimized_base_agent import OptimizedBaseAgent, agent_pool
    from ..services.performance_monitor import track_async_performance, perf_monitor
except ImportError:
    # Fall back to absolute imports (when used directly)
    from agents.agent_interface import AgentMessage, WorkflowStage, WorkflowConfig
    from agents.optimized_base_agent import OptimizedBaseAgent, agent_pool
    from services.performance_monitor import track_async_performance, perf_monitor


class OptimizedOrchestratorAgent(OptimizedBaseAgent):
    """
    Optimized orchestrator with parallel execution capabilities
    while maintaining backward compatibility
    """
    
    def __init__(self, kernel: sk.Kernel, memory_service=None, 
                 schema_analyst_agent=None, sql_generator_agent=None, 
                 executor_agent=None, summarizing_agent=None):
        
        super().__init__(
            agent_name="OptimizedOrchestrator",
            capabilities=[
                "workflow_orchestration", 
                "parallel_execution", 
                "conversation_management",
                "context_awareness",
                "performance_optimization"
            ],
            kernel=kernel
        )
        
        # Existing agents for backward compatibility
        self.memory_service = memory_service
        self.schema_analyst_agent = schema_analyst_agent
        self.sql_generator_agent = sql_generator_agent
        self.executor_agent = executor_agent
        self.summarizing_agent = summarizing_agent
        
        # Optimization features
        self.parallel_execution_enabled = True
        self.max_parallel_agents = 3
        self.workflow_cache = {}
        
        # Performance tracking
        self.workflow_metrics = {
            "sequential_workflows": 0,
            "parallel_workflows": 0,
            "cache_hits": 0,
            "total_time_saved": 0.0
        }
    
    async def _initialize_agent(self, config: WorkflowConfig):
        """Initialize the optimized orchestrator"""
        # Configure parallel execution based on config
        if hasattr(config, 'parallel_execution'):
            self.parallel_execution_enabled = config.parallel_execution
        
        if hasattr(config, 'max_parallel_agents'):
            self.max_parallel_agents = config.max_parallel_agents
        
        # Add agents to pool if they exist
        if self.schema_analyst_agent:
            agent_pool.add_agent(self.schema_analyst_agent)
        if self.sql_generator_agent:
            agent_pool.add_agent(self.sql_generator_agent)
        if self.executor_agent:
            agent_pool.add_agent(self.executor_agent)
        if self.summarizing_agent:
            agent_pool.add_agent(self.summarizing_agent)
    
    async def _process_message_impl(self, message: AgentMessage) -> AgentMessage:
        """Process message using optimized workflow"""
        
        # Extract content
        content = message.content
        user_input = content.get("user_input", "")
        session_id = content.get("session_id")
        conversation_context = content.get("conversation_context", "")
        
        # Determine workflow type based on query
        workflow_type = await self._determine_workflow_type(user_input, conversation_context)
        
        if workflow_type == "parallel":
            return await self._execute_parallel_workflow(user_input, session_id, conversation_context)
        else:
            return await self._execute_sequential_workflow(user_input, session_id, conversation_context)
    
    @track_async_performance("workflow_type_determination")
    async def _determine_workflow_type(self, user_input: str, conversation_context: str) -> str:
        """Determine if query can be processed in parallel"""
        
        # Simple heuristics for parallel processing eligibility
        parallel_indicators = [
            "show", "list", "get", "find", "analyze", "compare", "summarize"
        ]
        
        # Follow-up queries usually benefit from sequential processing for context
        sequential_indicators = [
            "what about", "also", "and", "more details", "explain", "why"
        ]
        
        user_lower = user_input.lower()
        
        # Check for sequential indicators first
        if any(indicator in user_lower for indicator in sequential_indicators):
            return "sequential"
        
        # Check for parallel indicators
        if any(indicator in user_lower for indicator in parallel_indicators):
            return "parallel"
        
        # Default to sequential for safety
        return "sequential"
    
    @track_async_performance("parallel_workflow_execution")
    async def _execute_parallel_workflow(self, user_input: str, session_id: str, 
                                       conversation_context: str) -> AgentMessage:
        """Execute workflow with parallel agent processing"""
        
        try:
            workflow_start = datetime.now()
            
            # Check workflow cache
            cache_key = f"parallel_{hash(user_input + (conversation_context or ''))}"
            if cache_key in self.workflow_cache:
                self.workflow_metrics["cache_hits"] += 1
                cached_result = self.workflow_cache[cache_key]
                cached_result["cache_hit"] = True
                return self._create_success_response(cached_result)
            
            # Stage 1: Prepare parallel tasks
            tasks = []
            
            # Task 1: Context enhancement (if memory service available)
            if self.memory_service:
                context_task = asyncio.create_task(
                    self._enhance_context_parallel(user_input, session_id, conversation_context)
                )
                tasks.append(("context", context_task))
            
            # Task 2: Schema analysis (if schema analyst available)
            if self.schema_analyst_agent:
                schema_task = asyncio.create_task(
                    self._analyze_schema_parallel(user_input)
                )
                tasks.append(("schema", schema_task))
            
            # Execute parallel tasks
            if tasks:
                results = await asyncio.gather(*[task[1] for task in tasks], return_exceptions=True)
                
                # Process results
                enhanced_context = conversation_context
                schema_result = None
                
                for i, (task_type, _) in enumerate(tasks):
                    if not isinstance(results[i], Exception):
                        if task_type == "context":
                            enhanced_context = results[i]
                        elif task_type == "schema":
                            schema_result = results[i]
            
            # Stage 2: SQL Generation (sequential, depends on previous results)
            sql_result = await self._generate_sql_optimized(user_input, enhanced_context, schema_result)
            
            if not sql_result.get("success"):
                return self._create_error_response(sql_result.get("error", "SQL generation failed"))
            
            # Stage 3: Execute SQL and Summarize in parallel
            parallel_stage2_tasks = []
            
            # Execute SQL
            if self.executor_agent:
                execute_task = asyncio.create_task(
                    self._execute_sql_optimized(sql_result)
                )
                parallel_stage2_tasks.append(("execute", execute_task))
            
            # Execute parallel stage 2
            if parallel_stage2_tasks:
                stage2_results = await asyncio.gather(
                    *[task[1] for task in parallel_stage2_tasks], 
                    return_exceptions=True
                )
                
                execution_result = None
                for i, (task_type, _) in enumerate(parallel_stage2_tasks):
                    if not isinstance(stage2_results[i], Exception):
                        if task_type == "execute":
                            execution_result = stage2_results[i]
            
            if not execution_result or not execution_result.get("success"):
                return self._create_error_response(execution_result.get("error", "SQL execution failed"))
            
            # Stage 4: Summarization
            summary_result = await self._summarize_results_optimized(
                user_input, execution_result, enhanced_context
            )
            
            # Log conversation
            if self.memory_service:
                asyncio.create_task(
                    self._log_conversation_async(user_input, summary_result, session_id)
                )
            
            # Calculate performance metrics
            workflow_time = (datetime.now() - workflow_start).total_seconds()
            self.workflow_metrics["parallel_workflows"] += 1
            
            # Prepare final result
            final_result = {
                "response": summary_result.get("summary", ""),
                "sql_query": sql_result.get("sql_query", ""),
                "data": execution_result.get("data", []),
                "success": True,
                "workflow_type": "parallel",
                "execution_time": workflow_time,
                "performance_optimized": True
            }
            
            # Cache result
            self.workflow_cache[cache_key] = final_result.copy()
            
            return self._create_success_response(final_result, {
                "workflow_type": "parallel",
                "execution_time": workflow_time,
                "cache_stored": True
            })
            
        except Exception as e:
            return self._create_error_response(f"Parallel workflow failed: {str(e)}")
    
    @track_async_performance("sequential_workflow_execution")
    async def _execute_sequential_workflow(self, user_input: str, session_id: str,
                                         conversation_context: str) -> AgentMessage:
        """Execute traditional sequential workflow for complex queries"""
        
        try:
            # Use existing orchestrator logic but with performance tracking
            if hasattr(self, '_execute_manual_sequential_workflow'):
                # Call existing method for backward compatibility
                result = await self._execute_manual_sequential_workflow(
                    user_input, session_id, conversation_context
                )
                
                self.workflow_metrics["sequential_workflows"] += 1
                
                # Convert to new message format
                if result.get("success"):
                    return self._create_success_response(result, {
                        "workflow_type": "sequential",
                        "backward_compatible": True
                    })
                else:
                    return self._create_error_response(result.get("error", "Sequential workflow failed"))
            
            # Fallback implementation
            return await self._execute_fallback_workflow(user_input, session_id, conversation_context)
            
        except Exception as e:
            return self._create_error_response(f"Sequential workflow failed: {str(e)}")
    
    async def _enhance_context_parallel(self, user_input: str, session_id: str, 
                                      conversation_context: str) -> str:
        """Enhance context using memory service in parallel"""
        if not self.memory_service:
            return conversation_context
        
        try:
            enhanced_context = await self.memory_service.get_conversation_context_with_summary(
                session_id, user_input
            )
            return enhanced_context or conversation_context
        except Exception:
            return conversation_context
    
    async def _analyze_schema_parallel(self, user_input: str) -> Dict[str, Any]:
        """Analyze schema in parallel"""
        if not self.schema_analyst_agent:
            return {}
        
        try:
            # Create optimized schema analysis message
            schema_message = AgentMessage(
                message_type="schema_analysis",
                content={"user_input": user_input},
                metadata={"parallel_execution": True}
            )
            
            # Process using new interface if available
            if hasattr(self.schema_analyst_agent, 'process_message'):
                response = await self.schema_analyst_agent.process_message(schema_message)
                return response.content
            else:
                # Fallback to legacy interface
                return await self.schema_analyst_agent.process(user_input)
                
        except Exception:
            return {}
    
    async def _generate_sql_optimized(self, user_input: str, context: str, 
                                    schema_result: Dict[str, Any]) -> Dict[str, Any]:
        """Generate SQL with optimization"""
        if not self.sql_generator_agent:
            return {"success": False, "error": "SQL generator not available"}
        
        try:
            # Use legacy interface for now
            return await self.sql_generator_agent.process(
                user_input, 
                conversation_context=context,
                schema_context=schema_result
            )
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _execute_sql_optimized(self, sql_result: Dict[str, Any]) -> Dict[str, Any]:
        """Execute SQL with optimization"""
        if not self.executor_agent:
            return {"success": False, "error": "Executor not available"}
        
        try:
            return await self.executor_agent.process(sql_result.get("sql_query", ""))
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _summarize_results_optimized(self, user_input: str, execution_result: Dict[str, Any],
                                         context: str) -> Dict[str, Any]:
        """Summarize results with optimization"""
        if not self.summarizing_agent:
            return {"summary": "Results retrieved successfully"}
        
        try:
            return await self.summarizing_agent.process(
                user_input,
                data=execution_result.get("data", []),
                conversation_context=context
            )
        except Exception as e:
            return {"summary": f"Results retrieved (summarization failed: {str(e)})"}
    
    async def _log_conversation_async(self, user_input: str, summary_result: Dict[str, Any], 
                                    session_id: str):
        """Log conversation asynchronously"""
        try:
            if self.memory_service and hasattr(self.memory_service, 'log_conversation'):
                await self.memory_service.log_conversation(
                    session_id=session_id,
                    user_message=user_input,
                    ai_response=summary_result.get("summary", ""),
                    metadata={"optimized": True}
                )
        except Exception:
            # Don't fail the main workflow for logging errors
            pass
    
    async def _execute_fallback_workflow(self, user_input: str, session_id: str,
                                       conversation_context: str) -> AgentMessage:
        """Fallback workflow implementation"""
        return self._create_success_response({
            "response": "I understand your request, but I'm currently in optimization mode. Please try again.",
            "workflow_type": "fallback",
            "success": True
        })
    
    async def get_workflow_metrics(self) -> Dict[str, Any]:
        """Get workflow performance metrics"""
        total_workflows = (
            self.workflow_metrics["sequential_workflows"] + 
            self.workflow_metrics["parallel_workflows"]
        )
        
        parallel_percentage = (
            (self.workflow_metrics["parallel_workflows"] / total_workflows * 100)
            if total_workflows > 0 else 0
        )
        
        return {
            **self.workflow_metrics,
            "total_workflows": total_workflows,
            "parallel_percentage": round(parallel_percentage, 1),
            "cache_hit_rate": round(
                (self.workflow_metrics["cache_hits"] / total_workflows * 100)
                if total_workflows > 0 else 0, 1
            ),
            "parallel_execution_enabled": self.parallel_execution_enabled,
            "max_parallel_agents": self.max_parallel_agents
        }
    
    def enable_parallel_execution(self, max_agents: int = 3):
        """Enable parallel execution"""
        self.parallel_execution_enabled = True
        self.max_parallel_agents = max_agents
    
    def disable_parallel_execution(self):
        """Disable parallel execution"""
        self.parallel_execution_enabled = False
