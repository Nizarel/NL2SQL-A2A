#!/usr/bin/env python3
"""
Enhanced Orchestrator Agent - Phase 3A Optimization Example
Demonstrates integration of all enhanced services for maximum efficiency
"""

import re
import time
import asyncio
from typing import Dict, Any, List, Optional
from semantic_kernel import Kernel
from semantic_kernel.agents import ChatCompletionAgent
from semantic_kernel.agents.strategies import SequentialSelectionStrategy
from semantic_kernel.agents.group_chat.agent_group_chat import AgentGroupChat
from semantic_kernel.contents import ChatMessageContent, AuthorRole
from semantic_kernel.connectors.ai.open_ai import AzureChatPromptExecutionSettings
from semantic_kernel.connectors.ai.function_choice_behavior import FunctionChoiceBehavior

from agents.base_agent import BaseAgent
from agents.sql_generator_agent import SQLGeneratorAgent
from agents.executor_agent import ExecutorAgent
from agents.summarizing_agent import SummarizingAgent
from agents.schema_analyst_agent import SchemaAnalystAgent
from services.orchestrator_memory_service import OrchestratorMemoryService
from Models.agent_response import FormattedResults, AgentResponse

# ============================================
# ENHANCED SERVICES INTEGRATION
# ============================================
from services.error_handling_service import ErrorHandlingService, ErrorCategory, ErrorSeverity
from services.sql_utility_service import SQLUtilityService
from services.monitoring_service import monitoring_service
from services.configuration_service import config_service
from services.template_service import TemplateService


class EnhancedOrchestratorAgent(BaseAgent):
    """
    Enhanced Orchestrator Agent with full service integration
    
    Phase 3A Optimizations:
    - Integrated ErrorHandlingService for standardized error handling
    - Integrated SQLUtilityService for consistent SQL processing
    - Integrated MonitoringService for real-time performance tracking
    - Integrated ConfigurationService for centralized settings
    - Integrated TemplateService for template management consistency
    """
    
    def __init__(self, kernel: Kernel, schema_analyst: SchemaAnalystAgent, 
                 sql_generator: SQLGeneratorAgent, executor: ExecutorAgent, 
                 summarizer: SummarizingAgent, memory_service: Optional[OrchestratorMemoryService] = None):
        super().__init__(kernel, "EnhancedOrchestratorAgent")
        
        # Store agent references
        self.schema_analyst = schema_analyst
        self.sql_generator = sql_generator
        self.executor = executor
        self.summarizer = summarizer
        
        # Memory service for conversation logging
        self.memory_service = memory_service
        
        # ============================================
        # ENHANCED SERVICES INITIALIZATION
        # ============================================
        
        # Initialize template service for consistency
        self.template_service = TemplateService()
        
        # Get orchestrator configuration
        self.orchestrator_config = config_service.get_config("orchestrator")
        
        # Initialize monitoring metrics
        self._initialize_monitoring_metrics()
        
        # Initialize Semantic Kernel AgentGroupChat for orchestration
        self.agent_group_chat: Optional[AgentGroupChat] = None
        self._sk_orchestration_active = False
        self._initialize_sk_orchestration()
        
        print("🚀 Enhanced Orchestrator Agent initialized with full service integration")
        
    def _initialize_monitoring_metrics(self):
        """Initialize performance monitoring metrics"""
        metrics_to_track = [
            ("orchestrator_workflow_time", "ms"),
            ("orchestrator_step_time", "ms"),
            ("orchestrator_error_rate", "%"),
            ("orchestrator_cache_hit_rate", "%"),
            ("orchestrator_sql_extraction_time", "ms"),
            ("orchestrator_success_rate", "%")
        ]
        
        for metric_name, unit in metrics_to_track:
            monitoring_service.record_metric(metric_name, 0.0)
        
        print("📊 Orchestrator monitoring metrics initialized")
    
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enhanced process method with full service integration
        
        Args:
            input_data: Dictionary containing workflow parameters
            
        Returns:
            Enhanced response with comprehensive error handling and monitoring
        """
        workflow_start_time = time.time()
        workflow_context = None
        correlation_id = input_data.get("correlation_id", f"workflow_{int(time.time())}")
        
        # Record workflow start
        monitoring_service.record_metric("orchestrator_workflow_time", 0)
        
        try:
            # ============================================
            # ENHANCED INPUT VALIDATION
            # ============================================
            
            # Use ConfigurationService for default values
            orchestrator_settings = config_service.get_config("orchestrator")
            
            question = input_data.get("question", "")
            user_id = input_data.get("user_id", "default_user")
            session_id = input_data.get("session_id", "default_session")
            context = input_data.get("context", "")
            execute = input_data.get("execute", orchestrator_settings.get("default_execute", True))
            limit = input_data.get("limit", orchestrator_settings.get("default_limit", 100))
            include_summary = input_data.get("include_summary", orchestrator_settings.get("default_include_summary", True))
            enable_conversation_logging = input_data.get("enable_conversation_logging", orchestrator_settings.get("default_logging", True))
            
            # Enhanced input validation with ErrorHandlingService
            if not question:
                return ErrorHandlingService.create_enhanced_error_response(
                    error=ValueError("No question provided for processing"),
                    context={"operation": "orchestrator_input_validation", "correlation_id": correlation_id}
                )
            
            print(f"🎯 Enhanced Orchestrating workflow for: {question}")
            
            # ============================================
            # ENHANCED WORKFLOW SESSION MANAGEMENT
            # ============================================
            
            # Start workflow session with enhanced error handling
            if self.memory_service and enable_conversation_logging:
                try:
                    workflow_context = await self.memory_service.start_workflow_session(
                        user_id=user_id,
                        user_input=question,
                        session_id=session_id
                    )
                    print(f"📝 Started enhanced workflow session: {workflow_context.workflow_id}")
                except Exception as e:
                    # Use ErrorHandlingService for standardized error handling
                    error_response = ErrorHandlingService.handle_agent_processing_error(
                        error=e,
                        agent_name="EnhancedOrchestratorAgent",
                        input_data={"operation": "start_workflow_session"},
                        step="workflow_initialization"
                    )
                    print(f"⚠️ Workflow session creation failed: {error_response['error']}")
                    workflow_context = None
            
            # ============================================
            # ENHANCED WORKFLOW EXECUTION
            # ============================================
            
            # Use enhanced manual sequential workflow with full service integration
            result = await self._execute_enhanced_sequential_workflow(
                input_data, workflow_context, enable_conversation_logging, correlation_id
            )
            
            # ============================================
            # ENHANCED WORKFLOW COMPLETION
            # ============================================
            
            # Complete workflow session with enhanced logging
            if self.memory_service and enable_conversation_logging and workflow_context and result.get("success"):
                try:
                    conversation_log = await self._complete_enhanced_workflow_with_logging(
                        workflow_context=workflow_context,
                        result=result,
                        question=question,
                        workflow_time=time.time() - workflow_start_time,
                        correlation_id=correlation_id
                    )
                    
                    if conversation_log:
                        result["conversation_log_id"] = conversation_log.id
                        print(f"📝 Enhanced conversation logged: {conversation_log.id}")
                    
                except Exception as e:
                    # Use ErrorHandlingService for logging errors
                    error_response = ErrorHandlingService.handle_agent_processing_error(
                        error=e,
                        agent_name="EnhancedOrchestratorAgent",
                        input_data={"operation": "complete_workflow_logging"},
                        step="workflow_completion"
                    )
                    print(f"⚠️ Enhanced conversation logging failed: {error_response['error']}")
            
            # ============================================
            # ENHANCED METADATA AND MONITORING
            # ============================================
            
            # Record workflow performance metrics
            workflow_time = time.time() - workflow_start_time
            monitoring_service.record_metric("orchestrator_workflow_time", workflow_time * 1000)  # Convert to ms
            
            # Record success rate
            success_rate = 100.0 if result.get("success") else 0.0
            monitoring_service.record_metric("orchestrator_success_rate", success_rate)
            
            # Enhanced metadata with service integration info
            if "metadata" not in result:
                result["metadata"] = {}
            
            result["metadata"].update({
                "total_workflow_time": round(workflow_time, 3),
                "orchestration_pattern": "enhanced_sequential",
                "workflow_steps": self._get_workflow_steps(execute, include_summary),
                "conversation_logged": bool(
                    self.memory_service and enable_conversation_logging and workflow_context and result.get("success")
                ),
                "services_integrated": {
                    "error_handling": True,
                    "sql_utility": True,
                    "monitoring": True,
                    "configuration": True,
                    "template": True
                },
                "correlation_id": correlation_id,
                "service_performance": self._get_service_performance_summary()
            })
            
            return result
            
        except Exception as e:
            # Enhanced error handling with full context
            workflow_time = time.time() - workflow_start_time
            monitoring_service.record_metric("orchestrator_workflow_time", workflow_time * 1000)
            monitoring_service.record_metric("orchestrator_success_rate", 0.0)
            
            return ErrorHandlingService.create_enhanced_error_response(
                error=e,
                context={
                    "operation": "enhanced_orchestrator_workflow",
                    "correlation_id": correlation_id,
                    "workflow_time": workflow_time,
                    "question": question[:100] if 'question' in locals() else "unknown"
                }
            )
    
    async def _execute_enhanced_sequential_workflow(
        self, 
        params: Dict[str, Any], 
        workflow_context=None, 
        enable_conversation_logging=True,
        correlation_id: str = ""
    ) -> Dict[str, Any]:
        """
        Enhanced sequential workflow with full service integration
        """
        workflow_results = {
            "schema_analysis": None,
            "sql_generation": None,
            "execution": None, 
            "summarization": None
        }
        
        step_start_time = time.time()
        
        try:
            # ============================================
            # STEP 0: ENHANCED SCHEMA ANALYSIS
            # ============================================
            
            print("🔍 Enhanced Step 0/4: Analyzing schema context...")
            step_start = time.time()
            
            try:
                schema_analysis = await self.schema_analyst.process({
                    "question": params["question"],
                    "context": params.get("context", ""),
                    "use_cache": True,
                    "similarity_threshold": config_service.get_config("schema_analyst").get("similarity_threshold", 0.85)
                })
                
                step_time = (time.time() - step_start) * 1000
                monitoring_service.record_metric("orchestrator_step_time", step_time)
                
                workflow_results["schema_analysis"] = schema_analysis
                
                # Enhanced logging with monitoring
                if schema_analysis["success"]:
                    cache_info = schema_analysis.get("metadata", {})
                    if cache_info.get("cache_hit"):
                        cache_hit_rate = 100.0
                        cache_type = cache_info.get("cache_type", "unknown")
                        print(f"✅ Enhanced schema analysis: Cache HIT ({cache_type})")
                        if cache_type == "semantic":
                            similarity = cache_info.get("semantic_similarity", 0)
                            print(f"   Semantic similarity: {similarity:.3f}")
                    else:
                        cache_hit_rate = 0.0
                        print(f"✅ Enhanced schema analysis: Fresh analysis ({cache_info.get('analysis_time', 0):.3f}s)")
                    
                    monitoring_service.record_metric("orchestrator_cache_hit_rate", cache_hit_rate)
                    
                else:
                    print(f"⚠️ Enhanced schema analysis failed: {schema_analysis['error']}")
                    monitoring_service.record_metric("orchestrator_error_rate", 100.0)
                
            except Exception as e:
                # Enhanced error handling for schema analysis step
                schema_analysis = ErrorHandlingService.handle_agent_processing_error(
                    error=e,
                    agent_name="SchemaAnalystAgent", 
                    input_data=params,
                    step="enhanced_schema_analysis"
                )
                workflow_results["schema_analysis"] = schema_analysis
                monitoring_service.record_metric("orchestrator_error_rate", 100.0)
            
            # ============================================
            # STEP 1: ENHANCED SQL GENERATION
            # ============================================
            
            print("🧠 Enhanced Step 1/4: Generating SQL query with optimized context...")
            step_start = time.time()
            
            try:
                # Extract optimized schema context with enhanced error handling
                if schema_analysis["success"]:
                    analysis_data = schema_analysis["data"]
                    optimized_schema_context = analysis_data.get("optimized_schema", "")
                else:
                    # Fallback to full schema with error logging
                    optimized_schema_context = self.schema_analyst.schema_service.get_full_schema_summary()
                    print("⚠️ Using fallback schema context due to analysis failure")
                
                sql_result = await self.sql_generator.process({
                    "question": params["question"],
                    "context": params.get("context", ""),
                    "optimized_schema_context": optimized_schema_context,
                    "schema_analysis": schema_analysis["data"] if schema_analysis["success"] else None
                })
                
                step_time = (time.time() - step_start) * 1000
                monitoring_service.record_metric("orchestrator_step_time", step_time)
                
                workflow_results["sql_generation"] = sql_result
                
                if not sql_result["success"]:
                    return ErrorHandlingService.create_enhanced_error_response(
                        error=Exception(f"Enhanced SQL generation failed: {sql_result['error']}"),
                        context={
                            "operation": "enhanced_sql_generation",
                            "correlation_id": correlation_id,
                            "workflow_data": workflow_results
                        }
                    )
                
                # ============================================
                # ENHANCED SQL PROCESSING WITH SQLUtilityService
                # ============================================
                
                generated_sql = sql_result["data"]["sql_query"]
                
                # Use SQLUtilityService for enhanced SQL processing
                extraction_start = time.time()
                
                # Validate SQL using SQLUtilityService
                validation_result = SQLUtilityService.validate_sql_syntax(generated_sql)
                if not validation_result["valid"]:
                    return ErrorHandlingService.handle_sql_error(
                        error=Exception(f"SQL validation failed: {validation_result['error']}"),
                        sql_query=generated_sql,
                        operation="enhanced_validation"
                    )
                
                # Clean SQL using SQLUtilityService
                cleaned_sql = SQLUtilityService.clean_sql_query(generated_sql)
                
                extraction_time = (time.time() - extraction_start) * 1000
                monitoring_service.record_metric("orchestrator_sql_extraction_time", extraction_time)
                
                print(f"✅ Enhanced SQL Generated and Validated: {cleaned_sql[:100]}...")
                
                # Update sql_result with cleaned SQL
                sql_result["data"]["sql_query"] = cleaned_sql
                
            except Exception as e:
                # Enhanced error handling for SQL generation step
                sql_result = ErrorHandlingService.handle_agent_processing_error(
                    error=e,
                    agent_name="SQLGeneratorAgent",
                    input_data=params,
                    step="enhanced_sql_generation"
                )
                workflow_results["sql_generation"] = sql_result
                monitoring_service.record_metric("orchestrator_error_rate", 100.0)
                return sql_result
            
            # ============================================
            # STEP 2: ENHANCED SQL EXECUTION
            # ============================================
            
            execution_result = None
            if params.get("execute", True):
                print("⚡ Enhanced Step 2/4: Executing SQL query...")
                step_start = time.time()
                
                try:
                    execution_result = await self.executor.process({
                        "sql_query": cleaned_sql,
                        "limit": params.get("limit", config_service.get_config("executor").get("default_limit", 100)),
                        "timeout": config_service.get_config("executor").get("default_timeout", 30)
                    })
                    
                    step_time = (time.time() - step_start) * 1000
                    monitoring_service.record_metric("orchestrator_step_time", step_time)
                    
                    workflow_results["execution"] = execution_result
                    
                    if not execution_result["success"]:
                        return ErrorHandlingService.handle_sql_error(
                            error=Exception(f"Enhanced SQL execution failed: {execution_result['error']}"),
                            sql_query=cleaned_sql,
                            operation="enhanced_execution"
                        )
                    
                    print("✅ Enhanced SQL executed successfully")
                    
                except Exception as e:
                    # Enhanced error handling for execution step
                    execution_result = ErrorHandlingService.handle_agent_processing_error(
                        error=e,
                        agent_name="ExecutorAgent",
                        input_data={"sql_query": cleaned_sql},
                        step="enhanced_sql_execution"
                    )
                    workflow_results["execution"] = execution_result
                    monitoring_service.record_metric("orchestrator_error_rate", 100.0)
                    return execution_result
            
            # ============================================
            # STEP 3: ENHANCED SUMMARIZATION
            # ============================================
            
            summarization_result = None
            if params.get("include_summary", True) and execution_result and execution_result["success"]:
                print("📊 Enhanced Step 3/4: Generating insights and summary...")
                step_start = time.time()
                
                try:
                    summarization_result = await self.summarizer.process({
                        "raw_results": execution_result["data"]["raw_results"],
                        "formatted_results": execution_result["data"]["formatted_results"],
                        "sql_query": cleaned_sql,
                        "question": params["question"],
                        "metadata": execution_result["metadata"],
                        "schema_analysis": schema_analysis["data"] if schema_analysis["success"] else None
                    })
                    
                    step_time = (time.time() - step_start) * 1000
                    monitoring_service.record_metric("orchestrator_step_time", step_time)
                    
                    workflow_results["summarization"] = summarization_result
                    
                    if summarization_result["success"]:
                        print("✅ Enhanced summary and insights generated")
                    else:
                        print(f"⚠️ Enhanced summary generation had issues: {summarization_result['error']}")
                        
                except Exception as e:
                    # Enhanced error handling for summarization step
                    summarization_result = ErrorHandlingService.handle_agent_processing_error(
                        error=e,
                        agent_name="SummarizingAgent",
                        input_data={"question": params["question"]},
                        step="enhanced_summarization"
                    )
                    workflow_results["summarization"] = summarization_result
                    # Don't fail the entire workflow for summarization errors
                    print(f"⚠️ Enhanced summarization failed but continuing: {summarization_result['error']}")
            
            # ============================================
            # ENHANCED RESULTS COMPILATION
            # ============================================
            
            # Compile final results with enhanced metadata
            final_result = self._compile_enhanced_workflow_results(workflow_results, params, correlation_id)
            
            # Enhanced optimization metadata
            final_result["metadata"].update({
                "cache_info": cache_info if 'cache_info' in locals() else "unknown",
                "schema_optimization": "enhanced_enabled",
                "schema_source": "enhanced_optimized",
                "sql_processing": "enhanced_utility_service",
                "error_handling": "enhanced_service_integrated",
                "monitoring": "real_time_enabled"
            })
            
            return final_result
            
        except Exception as e:
            # Final enhanced error handling
            return ErrorHandlingService.create_enhanced_error_response(
                error=e,
                context={
                    "operation": "enhanced_sequential_workflow",
                    "correlation_id": correlation_id,
                    "workflow_results": workflow_results
                }
            )
    
    def _compile_enhanced_workflow_results(self, workflow_results: Dict[str, Any], params: Dict[str, Any], correlation_id: str) -> Dict[str, Any]:
        """
        Enhanced results compilation with comprehensive service integration
        """
        schema_analysis = workflow_results.get("schema_analysis")
        sql_generation = workflow_results.get("sql_generation")
        execution = workflow_results.get("execution") 
        summarization = workflow_results.get("summarization")
        
        # Enhanced response data with service metadata
        response_data = {
            "sql_query": sql_generation["data"]["sql_query"] if sql_generation and sql_generation["success"] else None,
            "executed": False,
            "results": None,
            "formatted_results": None,
            "service_integration": {
                "error_handling_enhanced": True,
                "sql_utility_integrated": True,
                "monitoring_enabled": True,
                "configuration_managed": True
            }
        }
        
        # Enhanced schema analysis results
        if schema_analysis and schema_analysis["success"]:
            response_data["schema_analysis"] = {
                "relevant_tables": schema_analysis["data"].get("relevant_tables", []),
                "join_strategy": schema_analysis["data"].get("join_strategy", {}),
                "performance_hints": schema_analysis["data"].get("performance_hints", []),
                "confidence_score": schema_analysis["data"].get("confidence_score", 0)
            }
        
        # Enhanced execution results
        if execution and execution["success"]:
            response_data.update({
                "executed": True,
                "results": execution["data"]["raw_results"],
                "formatted_results": execution["data"]["formatted_results"]
            })
        elif execution and not execution["success"]:
            response_data["execution_error"] = execution["error"]
        
        # Enhanced summary results
        if summarization and summarization["success"]:
            response_data["summary"] = {
                "executive_summary": summarization["data"]["executive_summary"],
                "key_insights": summarization["data"]["key_insights"], 
                "recommendations": summarization["data"]["recommendations"],
                "data_overview": summarization["data"]["data_overview"],
                "technical_summary": summarization["data"]["technical_summary"]
            }
        
        # Enhanced metadata compilation
        metadata = {
            "workflow_success": True,
            "orchestration_type": "enhanced_schema_optimized",
            "steps_completed": [step for step, result in workflow_results.items() if result and result["success"]],
            "schema_analyzed": bool(schema_analysis and schema_analysis["success"]),
            "sql_generated": bool(sql_generation and sql_generation["success"]),
            "query_executed": bool(execution and execution["success"]),
            "summary_generated": bool(summarization and summarization["success"]),
            "correlation_id": correlation_id,
            "service_integration_level": "phase_3a_complete"
        }
        
        # Enhanced performance metadata
        if schema_analysis and schema_analysis["success"]:
            schema_metadata = schema_analysis.get("metadata", {})
            metadata.update({
                "schema_cache_hit": schema_metadata.get("cache_hit", False),
                "schema_cache_type": schema_metadata.get("cache_type"),
                "schema_analysis_time": schema_metadata.get("analysis_time"),
                "schema_confidence": schema_analysis["data"].get("confidence_score")
            })
        
        if execution and execution["success"]:
            exec_metadata = execution.get("metadata", {})
            metadata.update({
                "execution_time": exec_metadata.get("execution_time"),
                "row_count": exec_metadata.get("row_count"),
                "query_type": exec_metadata.get("query_type")
            })
        
        return ErrorHandlingService.create_success_response(
            data=response_data,
            metadata=metadata,
            message="Enhanced workflow completed successfully"
        )
    
    def _get_service_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary from all integrated services"""
        return {
            "monitoring_metrics_active": len(monitoring_service.metrics),
            "configuration_sections_loaded": len(config_service.config_cache),
            "template_functions_available": len(self.template_service._template_functions) if hasattr(self.template_service, '_template_functions') else 0,
            "error_handling_categories": len(ErrorCategory),
            "sql_utility_functions": 7  # Number of key functions in SQLUtilityService
        }
    
    async def _complete_enhanced_workflow_with_logging(
        self, 
        workflow_context, 
        result: Dict[str, Any], 
        question: str, 
        workflow_time: float,
        correlation_id: str
    ) -> Optional[Any]:
        """
        Enhanced workflow completion with comprehensive service integration
        """
        try:
            # Enhanced result extraction with better error handling
            data = result.get("data", {})
            
            # Create enhanced FormattedResults
            formatted_results = None
            if "formatted_results" in data and data["formatted_results"]:
                if isinstance(data["formatted_results"], dict):
                    formatted_results = FormattedResults(**data["formatted_results"])
                else:
                    formatted_results = data["formatted_results"]
            
            # Enhanced summary data processing
            summary_data = data.get("summary", {})
            
            if summary_data:
                # Enhanced insight and recommendation processing
                key_insights = summary_data.get("key_insights", ["Enhanced data retrieved and analyzed"])
                recommendations = summary_data.get("recommendations", ["Review enhanced results for insights"])
                
                # Convert structured data to strings with enhanced processing
                insights_strings = []
                for insight in key_insights:
                    if isinstance(insight, dict):
                        insight_text = insight.get("finding", str(insight))
                        insights_strings.append(insight_text)
                    else:
                        insights_strings.append(str(insight))
                
                recommendations_strings = []
                for rec in recommendations:
                    if isinstance(rec, dict):
                        rec_text = rec.get("action", rec.get("recommendation", str(rec)))
                        recommendations_strings.append(rec_text)
                    else:
                        recommendations_strings.append(str(rec))
                
                # Enhanced AgentResponse with service metadata
                agent_response = AgentResponse(
                    agent_type="enhanced_orchestrator",
                    response="Enhanced workflow completed successfully",
                    success=result.get("success", False),
                    executive_summary=summary_data.get("executive_summary", "Enhanced query processed successfully"),
                    key_insights=insights_strings,
                    recommendations=recommendations_strings,
                    confidence_level=summary_data.get("confidence_level", "high"),
                    processing_time_ms=int(workflow_time * 1000)
                )
            else:
                # Enhanced fallback response
                agent_response = AgentResponse(
                    agent_type="enhanced_orchestrator",
                    response="Enhanced query processed",
                    success=result.get("success", False),
                    key_insights=["Enhanced data processing completed"],
                    recommendations=["Review enhanced results for insights"],
                    confidence_level="medium",
                    processing_time_ms=int(workflow_time * 1000)
                )
            
            # Enhanced workflow session completion
            conversation_log = await self.memory_service.complete_workflow_session(
                workflow_context=workflow_context,
                formatted_results=formatted_results,
                agent_response=agent_response,
                sql_query=data.get("sql_query"),
                processing_time_ms=int(workflow_time * 1000)
            )
            
            # Record conversation logging success
            if conversation_log:
                monitoring_service.record_metric("orchestrator_logging_success_rate", 100.0)
            else:
                monitoring_service.record_metric("orchestrator_logging_success_rate", 0.0)
            
            return conversation_log
            
        except Exception as e:
            # Enhanced error logging with service integration
            error_response = ErrorHandlingService.handle_agent_processing_error(
                error=e,
                agent_name="EnhancedOrchestratorAgent",
                input_data={"operation": "enhanced_workflow_completion", "correlation_id": correlation_id},
                step="enhanced_logging_completion"
            )
            
            monitoring_service.record_metric("orchestrator_logging_success_rate", 0.0)
            print(f"❌ Enhanced workflow logging error: {error_response['error']}")
            return None
    
    def _initialize_sk_orchestration(self):
        """Initialize Semantic Kernel orchestration (unchanged from original)"""
        try:
            # Create execution settings for all agents
            settings = AzureChatPromptExecutionSettings(
                max_tokens=2000,
                temperature=0.1,
                top_p=0.9,
                function_choice_behavior=FunctionChoiceBehavior.Auto()
            )
            
            # Create ChatCompletionAgents that wrap our specialized agents
            schema_analysis_agent = ChatCompletionAgent(
                kernel=self.kernel,
                name="SchemaAnalystAgent",
                instructions="""You are an enhanced schema analysis specialist with full service integration..."""
            )
            
            sql_generation_agent = ChatCompletionAgent(
                kernel=self.kernel,
                name="SQLGeneratorAgent", 
                instructions="""You are an enhanced SQL generation specialist with advanced service integration..."""
            )
            
            query_executor_agent = ChatCompletionAgent(
                kernel=self.kernel,
                name="ExecutorAgent",
                instructions="""You are an enhanced SQL execution specialist with monitoring integration..."""
            )
            
            data_analysis_agent = ChatCompletionAgent(
                kernel=self.kernel, 
                name="SummarizingAgent",
                instructions="""You are an enhanced data analysis specialist with template service integration..."""
            )
            
            # Create AgentGroupChat with Sequential Selection Strategy
            self.agent_group_chat = AgentGroupChat(
                agents=[schema_analysis_agent, sql_generation_agent, query_executor_agent, data_analysis_agent],
                selection_strategy=SequentialSelectionStrategy()
            )
            
            print("✅ Enhanced Semantic Kernel Sequential Orchestration initialized with service integration")
            
        except Exception as e:
            print(f"❌ Failed to initialize enhanced SK orchestration: {str(e)}")
            print("⚠️ Will use enhanced manual sequential orchestration as fallback")
            self.agent_group_chat = None
    
    # ============================================
    # ENHANCED UTILITY METHODS
    # ============================================
    
    def _get_workflow_steps(self, execute: bool, include_summary: bool) -> List[str]:
        """Get enhanced workflow steps"""
        steps = ["enhanced_schema_analysis", "enhanced_sql_generation"]
        if execute:
            steps.append("enhanced_execution")
            if include_summary:
                steps.append("enhanced_summarization")
        return steps
    
    async def get_enhanced_workflow_status(self) -> Dict[str, Any]:
        """Get enhanced workflow status with service integration info"""
        # Get system health from MonitoringService
        system_health = monitoring_service.get_system_health()
        
        return ErrorHandlingService.create_success_response(
            data={
                "orchestrator": "enhanced_active",
                "agents": {
                    "schema_analyst": "enhanced_ready",
                    "sql_generator": "enhanced_ready", 
                    "executor": "enhanced_ready",
                    "summarizer": "enhanced_ready"
                },
                "orchestration_mode": "enhanced_schema_optimized",
                "workflow_capabilities": {
                    "enhanced_schema_analysis": True,
                    "intelligent_caching": True,
                    "optimized_context": True,
                    "enhanced_sql_generation": True,
                    "monitored_query_execution": True,
                    "enhanced_data_summarization": True,
                    "sequential_processing": True,
                    "real_time_monitoring": True,
                    "centralized_configuration": True,
                    "standardized_error_handling": True
                },
                "service_integration": {
                    "error_handling_service": True,
                    "sql_utility_service": True,
                    "monitoring_service": True,
                    "configuration_service": True,
                    "template_service": True
                },
                "system_health": system_health["overall_status"],
                "performance_metrics": self._get_service_performance_summary()
            },
            message="Enhanced orchestrator with full service integration active"
        )


# ============================================
# USAGE EXAMPLE
# ============================================

async def demo_enhanced_orchestrator():
    """Demonstrate the enhanced orchestrator with full service integration"""
    
    print("🚀 Enhanced Orchestrator Agent Demo")
    print("=" * 60)
    
    # This would be initialized with actual kernel and agents in production
    # enhanced_orchestrator = EnhancedOrchestratorAgent(
    #     kernel=kernel,
    #     schema_analyst=schema_analyst,
    #     sql_generator=sql_generator,
    #     executor=executor,
    #     summarizer=summarizer
    # )
    
    # Example enhanced workflow call
    # result = await enhanced_orchestrator.process({
    #     "question": "What are the top 5 products by revenue for each CEDI?",
    #     "user_id": "demo_user",
    #     "session_id": "demo_session",
    #     "execute": True,
    #     "include_summary": True,
    #     "correlation_id": "demo_001"
    # })
    
    print("✅ Enhanced orchestrator demo setup complete")
    print("📊 Service integration benefits:")
    print("   - Standardized error handling across all operations")
    print("   - Real-time performance monitoring and metrics")
    print("   - Centralized configuration management")
    print("   - Enhanced SQL processing with validation")
    print("   - Comprehensive logging with correlation IDs")

if __name__ == "__main__":
    import asyncio
    asyncio.run(demo_enhanced_orchestrator())
