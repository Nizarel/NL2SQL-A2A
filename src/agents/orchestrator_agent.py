"""
Orchestrator Agent - Coordinates multi-agent workflow using Semantic Kernel
"""

import time
from typing import Dict, Any, List
from semantic_kernel import Kernel

from agents.base_agent import BaseAgent
from agents.sql_generator_agent import SQLGeneratorAgent
from agents.executor_agent import ExecutorAgent
from agents.summarizing_agent import SummarizingAgent


class OrchestratorAgent(BaseAgent):
    """
    Agent responsible for orchestrating the multi-agent workflow
    Uses Semantic Kernel's InProcessRuntime for agent coordination
    """
    
    def __init__(self, kernel: Kernel, sql_generator: SQLGeneratorAgent, 
                 executor: ExecutorAgent, summarizer: SummarizingAgent):
        super().__init__(kernel, "OrchestratorAgent")
        self.sql_generator = sql_generator
        self.executor = executor
        self.summarizer = summarizer
        
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Orchestrate the complete NL2SQL workflow
        
        Args:
            input_data: Dictionary containing:
                - question: User's natural language question
                - context: Optional additional context
                - execute: Whether to execute the query (default: True)
                - limit: Row limit for results (default: 100)
                - include_summary: Whether to generate summary (default: True)
                
        Returns:
            Dictionary containing complete workflow results
        """
        workflow_start_time = time.time()
        
        try:
            # Extract parameters
            question = input_data.get("question", "")
            context = input_data.get("context", "")
            execute = input_data.get("execute", True)
            limit = input_data.get("limit", 100)
            include_summary = input_data.get("include_summary", True)
            
            if not question:
                return self._create_result(
                    success=False,
                    error="No question provided for processing"
                )
            
            # Execute the sequential workflow
            workflow_result = await self._execute_sequential_workflow({
                "question": question,
                "context": context,
                "execute": execute,
                "limit": limit,
                "include_summary": include_summary
            })
            
            # Add workflow metadata
            workflow_result["metadata"]["total_workflow_time"] = round(time.time() - workflow_start_time, 3)
            workflow_result["metadata"]["workflow_steps"] = self._get_workflow_steps(execute, include_summary)
            
            return workflow_result
            
        except Exception as e:
            return self._create_result(
                success=False,
                error=f"Workflow orchestration failed: {str(e)}",
                metadata={
                    "total_workflow_time": round(time.time() - workflow_start_time, 3),
                    "failed_at": "orchestration"
                }
            )
    
    async def _execute_sequential_workflow(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the sequential agent workflow
        """
        workflow_results = {
            "sql_generation": None,
            "execution": None,
            "summarization": None
        }
        
        try:
            # Step 1: SQL Generation
            print("🧠 Step 1: Analyzing question and generating SQL...")
            sql_gen_result = await self.sql_generator.process({
                "question": params["question"],
                "context": params["context"]
            })
            
            workflow_results["sql_generation"] = sql_gen_result
            
            if not sql_gen_result["success"]:
                return self._create_result(
                    success=False,
                    error=f"SQL generation failed: {sql_gen_result['error']}",
                    data=workflow_results,
                    metadata={"failed_at": "sql_generation"}
                )
            
            generated_sql = sql_gen_result["data"]["sql_query"]
            print(f"✅ SQL Generated: {generated_sql}")
            
            # Step 2: Query Execution (if requested)
            execution_result = None
            if params["execute"]:
                print("⚡ Step 2: Executing SQL query...")
                execution_result = await self.executor.process({
                    "sql_query": generated_sql,
                    "limit": params["limit"]
                })
                
                workflow_results["execution"] = execution_result
                
                if not execution_result["success"]:
                    return self._create_result(
                        success=False,
                        error=f"Query execution failed: {execution_result['error']}",
                        data=workflow_results,
                        metadata={"failed_at": "execution", "sql_query": generated_sql}
                    )
                
                print(f"✅ Query executed successfully: {execution_result['metadata']['row_count']} rows returned")
            
            # Step 3: Summarization (if requested and execution was successful)
            summarization_result = None
            if params["include_summary"] and execution_result and execution_result["success"]:
                print("📊 Step 3: Generating insights and summary...")
                summarization_result = await self.summarizer.process({
                    "raw_results": execution_result["data"]["raw_results"],
                    "formatted_results": execution_result["data"]["formatted_results"],
                    "sql_query": generated_sql,
                    "question": params["question"],
                    "metadata": execution_result["metadata"]
                })
                
                workflow_results["summarization"] = summarization_result
                
                if summarization_result["success"]:
                    print("✅ Summary and insights generated successfully")
                else:
                    print(f"⚠️ Summarization failed: {summarization_result['error']}")
            
            # Compile final results
            return self._compile_workflow_results(
                params["question"],
                workflow_results,
                params["execute"],
                params["include_summary"]
            )
            
        except Exception as e:
            return self._create_result(
                success=False,
                error=f"Workflow execution failed: {str(e)}",
                data=workflow_results,
                metadata={"failed_at": "workflow_execution"}
            )
    
    def _compile_workflow_results(self, question: str, workflow_results: Dict[str, Any], 
                                execute: bool, include_summary: bool) -> Dict[str, Any]:
        """
        Compile results from all workflow steps into final response
        """
        try:
            sql_gen = workflow_results["sql_generation"]
            execution = workflow_results["execution"]
            summarization = workflow_results["summarization"]
            
            # Base response structure
            response_data = {
                "question": question,
                "sql_query": sql_gen["data"]["sql_query"] if sql_gen and sql_gen["success"] else None,
                "intent_analysis": sql_gen["data"].get("intent_analysis") if sql_gen and sql_gen["success"] else None
            }
            
            # Add execution results if available
            if execute and execution:
                if execution["success"]:
                    response_data.update({
                        "executed": True,
                        "results": execution["data"]["raw_results"],
                        "formatted_results": execution["data"]["formatted_results"],
                        "row_count": execution["metadata"]["row_count"],
                        "execution_time": execution["metadata"]["execution_time"]
                    })
                else:
                    response_data.update({
                        "executed": False,
                        "execution_error": execution["error"]
                    })
            else:
                response_data["executed"] = False
            
            # Add summarization results if available
            if include_summary and summarization:
                if summarization["success"]:
                    response_data.update({
                        "summary": {
                            "executive_summary": summarization["data"]["executive_summary"],
                            "key_insights": summarization["data"]["key_insights"],
                            "recommendations": summarization["data"]["recommendations"],
                            "data_overview": summarization["data"]["data_overview"],
                            "technical_summary": summarization["data"]["technical_summary"]
                        },
                        "summary_metadata": summarization["metadata"]
                    })
                else:
                    response_data["summary_error"] = summarization["error"]
            
            # Compile metadata
            metadata = {
                "workflow_success": True,
                "steps_completed": [],
                "agent_results": {
                    "sql_generator": sql_gen["success"] if sql_gen else False,
                    "executor": execution["success"] if execution else None,
                    "summarizer": summarization["success"] if summarization else None
                }
            }
            
            # Track completed steps
            if sql_gen and sql_gen["success"]:
                metadata["steps_completed"].append("sql_generation")
            if execution and execution["success"]:
                metadata["steps_completed"].append("execution")
            if summarization and summarization["success"]:
                metadata["steps_completed"].append("summarization")
            
            # Add timing information
            if execution and execution["success"]:
                metadata["execution_time"] = execution["metadata"]["execution_time"]
                metadata["row_count"] = execution["metadata"]["row_count"]
            
            return self._create_result(
                success=True,
                data=response_data,
                metadata=metadata
            )
            
        except Exception as e:
            return self._create_result(
                success=False,
                error=f"Failed to compile workflow results: {str(e)}",
                data=workflow_results,
                metadata={"failed_at": "result_compilation"}
            )
    
    def _get_workflow_steps(self, execute: bool, include_summary: bool) -> List[str]:
        """
        Get list of workflow steps based on parameters
        """
        steps = ["sql_generation"]
        
        if execute:
            steps.append("execution")
            
            if include_summary:
                steps.append("summarization")
        
        return steps
    
    async def get_workflow_status(self) -> Dict[str, Any]:
        """
        Get current workflow status and agent health
        """
        try:
            status = {
                "orchestrator": "healthy",
                "agents": {
                    "sql_generator": "healthy" if self.sql_generator else "not_initialized",
                    "executor": "healthy" if self.executor else "not_initialized",
                    "summarizer": "healthy" if self.summarizer else "not_initialized"
                },
                "workflow_capabilities": {
                    "sql_generation": True,
                    "query_execution": bool(self.executor),
                    "summarization": bool(self.summarizer),
                    "full_workflow": bool(self.sql_generator and self.executor and self.summarizer)
                }
            }
            
            return self._create_result(
                success=True,
                data=status,
                metadata={"timestamp": time.time()}
            )
            
        except Exception as e:
            return self._create_result(
                success=False,
                error=f"Status check failed: {str(e)}"
            )
    
    async def execute_single_step(self, step: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a single workflow step for testing or debugging
        
        Args:
            step: Step name ('sql_generation', 'execution', 'summarization')
            input_data: Input data for the specific step
            
        Returns:
            Results from the specific agent
        """
        try:
            if step == "sql_generation":
                return await self.sql_generator.process(input_data)
            elif step == "execution":
                return await self.executor.process(input_data)
            elif step == "summarization":
                return await self.summarizer.process(input_data)
            else:
                return self._create_result(
                    success=False,
                    error=f"Unknown workflow step: {step}"
                )
                
        except Exception as e:
            return self._create_result(
                success=False,
                error=f"Single step execution failed: {str(e)}",
                metadata={"step": step}
            )
