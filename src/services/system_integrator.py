"""
Integration helper for optimized components with existing system
Provides seamless migration from legacy to optimized architecture
"""

import asyncio
from typing import Dict, Any, Optional
from datetime import datetime
import semantic_kernel as sk

try:
    # Try relative imports first (when used as module)
    from ..agents.optimized_orchestrator_agent import OptimizedOrchestratorAgent
    from ..agents.optimized_base_agent import OptimizedBaseAgent, agent_pool
    from ..agents.agent_interface import AgentMessage, WorkflowConfig
    from ..services.performance_monitor import track_async_performance, perf_monitor
except ImportError:
    # Fall back to absolute imports (when used directly)
    from agents.optimized_orchestrator_agent import OptimizedOrchestratorAgent
    from agents.optimized_base_agent import OptimizedBaseAgent, agent_pool
    from agents.agent_interface import AgentMessage, WorkflowConfig
    from services.performance_monitor import track_async_performance, perf_monitor


class SystemIntegrator:
    """
    Handles integration between optimized and legacy components
    """
    
    def __init__(self):
        self.migration_mode = False
        self.performance_comparison = {
            "legacy_executions": 0,
            "optimized_executions": 0,
            "legacy_total_time": 0.0,
            "optimized_total_time": 0.0
        }
        self.fallback_enabled = True
    
    async def create_optimized_orchestrator(self, kernel: sk.Kernel, existing_orchestrator) -> OptimizedOrchestratorAgent:
        """
        Create optimized orchestrator that wraps existing functionality
        """
        # Extract existing agents from current orchestrator
        memory_service = getattr(existing_orchestrator, 'memory_service', None)
        schema_analyst = getattr(existing_orchestrator, 'schema_analyst_agent', None)
        sql_generator = getattr(existing_orchestrator, 'sql_generator_agent', None)
        executor = getattr(existing_orchestrator, 'executor_agent', None)
        summarizing = getattr(existing_orchestrator, 'summarizing_agent', None)
        
        # Create optimized orchestrator
        optimized_orchestrator = OptimizedOrchestratorAgent(
            kernel=kernel,
            memory_service=memory_service,
            schema_analyst_agent=schema_analyst,
            sql_generator_agent=sql_generator,
            executor_agent=executor,
            summarizing_agent=summarizing
        )
        
        # Initialize with default config
        config = WorkflowConfig(
            parallel_execution=True,
            max_parallel_agents=3,
            enable_caching=True,
            performance_monitoring=True
        )
        
        await optimized_orchestrator.initialize(config)
        
        # Copy existing workflow method for backward compatibility
        if hasattr(existing_orchestrator, '_execute_manual_sequential_workflow'):
            optimized_orchestrator._execute_manual_sequential_workflow = (
                existing_orchestrator._execute_manual_sequential_workflow
            )
        
        return optimized_orchestrator
    
    async def integrate_with_existing_system(self, nl2sql_system) -> bool:
        """
        Integrate optimized components with existing NL2SQL system
        
        Args:
            nl2sql_system: Existing NL2SQLMultiAgentSystem instance
            
        Returns:
            bool: True if integration successful
        """
        try:
            print("🔄 Starting system integration...")
            
            # Create optimized orchestrator
            optimized_orchestrator = await self.create_optimized_orchestrator(
                nl2sql_system.kernel, 
                nl2sql_system.orchestrator_agent
            )
            
            # Create wrapper that handles both interfaces
            wrapped_orchestrator = OrchestratorWrapper(
                legacy_orchestrator=nl2sql_system.orchestrator_agent,
                optimized_orchestrator=optimized_orchestrator,
                integrator=self
            )
            
            # Replace orchestrator in system
            nl2sql_system.orchestrator_agent = wrapped_orchestrator
            
            print("✅ System integration completed successfully")
            return True
            
        except Exception as e:
            print(f"❌ System integration failed: {str(e)}")
            return False
    
    def enable_migration_mode(self):
        """Enable migration mode for gradual transition"""
        self.migration_mode = True
        print("🔄 Migration mode enabled - running both legacy and optimized in parallel")
    
    def disable_migration_mode(self):
        """Disable migration mode - use optimized only"""
        self.migration_mode = False
        print("🚀 Migration mode disabled - using optimized components only")
    
    def get_performance_comparison(self) -> Dict[str, Any]:
        """Get performance comparison between legacy and optimized"""
        legacy_avg = (
            self.performance_comparison["legacy_total_time"] / 
            self.performance_comparison["legacy_executions"]
        ) if self.performance_comparison["legacy_executions"] > 0 else 0
        
        optimized_avg = (
            self.performance_comparison["optimized_total_time"] / 
            self.performance_comparison["optimized_executions"]
        ) if self.performance_comparison["optimized_executions"] > 0 else 0
        
        improvement = (
            ((legacy_avg - optimized_avg) / legacy_avg * 100)
            if legacy_avg > 0 and optimized_avg > 0 else 0
        )
        
        return {
            "legacy_executions": self.performance_comparison["legacy_executions"],
            "optimized_executions": self.performance_comparison["optimized_executions"],
            "legacy_avg_time": round(legacy_avg, 3),
            "optimized_avg_time": round(optimized_avg, 3),
            "performance_improvement_percent": round(improvement, 1),
            "migration_mode": self.migration_mode
        }


class OrchestratorWrapper:
    """
    Wrapper that provides both legacy and optimized orchestrator interfaces
    """
    
    def __init__(self, legacy_orchestrator, optimized_orchestrator: OptimizedOrchestratorAgent, 
                 integrator: SystemIntegrator):
        self.legacy_orchestrator = legacy_orchestrator
        self.optimized_orchestrator = optimized_orchestrator
        self.integrator = integrator
        
        # Copy essential attributes from legacy
        for attr in ['memory_service', 'schema_analyst_agent', 'sql_generator_agent', 
                    'executor_agent', 'summarizing_agent']:
            if hasattr(legacy_orchestrator, attr):
                setattr(self, attr, getattr(legacy_orchestrator, attr))
    
    @track_async_performance("orchestrator_wrapper_process")
    async def process(self, user_input: str, session_id: str = None, 
                     conversation_context: str = None, **kwargs) -> Dict[str, Any]:
        """
        Process request using appropriate orchestrator
        """
        
        if self.integrator.migration_mode:
            # Run both for comparison
            return await self._run_migration_mode(user_input, session_id, conversation_context, **kwargs)
        else:
            # Use optimized only
            return await self._run_optimized_only(user_input, session_id, conversation_context, **kwargs)
    
    async def _run_migration_mode(self, user_input: str, session_id: str, 
                                conversation_context: str, **kwargs) -> Dict[str, Any]:
        """Run both legacy and optimized for comparison"""
        
        try:
            # Run both in parallel
            legacy_task = asyncio.create_task(
                self._run_legacy_with_timing(user_input, session_id, conversation_context, **kwargs)
            )
            optimized_task = asyncio.create_task(
                self._run_optimized_with_timing(user_input, session_id, conversation_context, **kwargs)
            )
            
            # Wait for both
            legacy_result, optimized_result = await asyncio.gather(
                legacy_task, optimized_task, return_exceptions=True
            )
            
            # Compare results and return optimized if successful, otherwise legacy
            if not isinstance(optimized_result, Exception) and optimized_result.get("success"):
                print("✅ Using optimized result in migration mode")
                return optimized_result
            elif not isinstance(legacy_result, Exception) and legacy_result.get("success"):
                print("⚠️ Falling back to legacy result in migration mode")
                return legacy_result
            else:
                # Both failed
                return {
                    "success": False,
                    "error": "Both legacy and optimized processing failed",
                    "legacy_error": str(legacy_result) if isinstance(legacy_result, Exception) else legacy_result.get("error"),
                    "optimized_error": str(optimized_result) if isinstance(optimized_result, Exception) else optimized_result.get("error")
                }
                
        except Exception as e:
            return {"success": False, "error": f"Migration mode execution failed: {str(e)}"}
    
    async def _run_optimized_only(self, user_input: str, session_id: str, 
                                conversation_context: str, **kwargs) -> Dict[str, Any]:
        """Run optimized orchestrator only"""
        
        try:
            return await self._run_optimized_with_timing(user_input, session_id, conversation_context, **kwargs)
        except Exception as e:
            if self.integrator.fallback_enabled:
                print(f"⚠️ Optimized execution failed, falling back to legacy: {str(e)}")
                return await self._run_legacy_with_timing(user_input, session_id, conversation_context, **kwargs)
            else:
                return {"success": False, "error": f"Optimized execution failed: {str(e)}"}
    
    async def _run_legacy_with_timing(self, user_input: str, session_id: str, 
                                    conversation_context: str, **kwargs) -> Dict[str, Any]:
        """Run legacy orchestrator with performance timing"""
        start_time = datetime.now()
        
        try:
            result = await self.legacy_orchestrator.process(
                user_input, session_id, conversation_context, **kwargs
            )
            
            execution_time = (datetime.now() - start_time).total_seconds()
            self.integrator.performance_comparison["legacy_executions"] += 1
            self.integrator.performance_comparison["legacy_total_time"] += execution_time
            
            # Add timing metadata
            if isinstance(result, dict):
                result["execution_time"] = execution_time
                result["orchestrator_type"] = "legacy"
            
            return result
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            self.integrator.performance_comparison["legacy_executions"] += 1
            self.integrator.performance_comparison["legacy_total_time"] += execution_time
            
            return {
                "success": False,
                "error": str(e),
                "execution_time": execution_time,
                "orchestrator_type": "legacy"
            }
    
    async def _run_optimized_with_timing(self, user_input: str, session_id: str, 
                                       conversation_context: str, **kwargs) -> Dict[str, Any]:
        """Run optimized orchestrator with performance timing"""
        start_time = datetime.now()
        
        try:
            result = await self.optimized_orchestrator.process(
                user_input, session_id, conversation_context, **kwargs
            )
            
            execution_time = (datetime.now() - start_time).total_seconds()
            self.integrator.performance_comparison["optimized_executions"] += 1
            self.integrator.performance_comparison["optimized_total_time"] += execution_time
            
            # Add timing metadata
            if isinstance(result, dict):
                result["execution_time"] = execution_time
                result["orchestrator_type"] = "optimized"
            
            return result
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            self.integrator.performance_comparison["optimized_executions"] += 1
            self.integrator.performance_comparison["optimized_total_time"] += execution_time
            
            return {
                "success": False,
                "error": str(e),
                "execution_time": execution_time,
                "orchestrator_type": "optimized"
            }
    
    # Delegate other methods to legacy for backward compatibility
    def __getattr__(self, name):
        """Delegate unknown attributes to legacy orchestrator"""
        return getattr(self.legacy_orchestrator, name)


# Global integrator instance
system_integrator = SystemIntegrator()
