"""
Orchestrator Agent - Coordinates multi-agent workflow using Semantic Kernel 1.34.0
Sequential Pattern: SchemaAnalyst → SQLGenerator → Executor → Summarizer
Enhanced with conversation logging and session tracking
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

from .base_agent import BaseAgent
from .sql_generator_agent import SQLGeneratorAgent
from .executor_agent import ExecutorAgent
from .summarizing_agent import SummarizingAgent
from .schema_analyst_agent import SchemaAnalystAgent
from services.orchestrator_memory_service import OrchestratorMemoryService
from Models.agent_response import FormattedResults, AgentResponse


class OrchestratorAgent(BaseAgent):
    """
    Agent responsible for orchestrating the sequential multi-agent workflow:
    1. SchemaAnalystAgent: Analyzes schema and provides optimized context
    2. SQLGeneratorAgent: Converts natural language to SQL using optimized context
    3. ExecutorAgent: Executes the generated SQL query
    4. SummarizingAgent: Analyzes results and generates insights
    
    Enhanced with automatic conversation logging and session tracking.
    """
    
    def __init__(self, kernel: Kernel, schema_analyst: SchemaAnalystAgent, 
                 sql_generator: SQLGeneratorAgent, executor: ExecutorAgent, 
                 summarizer: SummarizingAgent, memory_service: Optional[OrchestratorMemoryService] = None):
        super().__init__(kernel, "OrchestratorAgent")
        
        # Store agent references
        self.schema_analyst = schema_analyst
        self.sql_generator = sql_generator
        self.executor = executor
        self.summarizer = summarizer
        
        # Memory service for conversation logging
        self.memory_service = memory_service
        
        # Initialize Semantic Kernel AgentGroupChat for orchestration
        self.agent_group_chat: Optional[AgentGroupChat] = None
        self._sk_orchestration_active = False  # Track if SK orchestration is in use
        self._initialize_sk_orchestration()
        
    def _initialize_sk_orchestration(self):
        """
        Initialize Semantic Kernel AgentGroupChat with Sequential Selection Strategy
        Note: SK AgentGroupChat doesn't handle concurrency well, so we'll use it selectively
        """
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
                instructions="""You are a schema analysis specialist. Your role is to:
1. Analyze natural language questions against the provided database schema
2. Identify the most relevant tables and columns for the question
3. For revenue/sales questions, focus on 'segmentacion' table with 'IngresoNetoSImpuestos' column
4. For customer questions, use 'cliente' table with customer details
5. Provide specific table and column recommendations to the SQL Generator
6. Recommend optimal join strategies between tables
7. Pass the exact table/column names to be used in SQL generation

Available tables: cliente, cliente_cedi, mercado, producto, segmentacion, tiempo"""
            )
            
            sql_generation_agent = ChatCompletionAgent(
                kernel=self.kernel,
                name="SQLGeneratorAgent",
                instructions="""You are a SQL generation specialist. Your ONLY role is to generate SQL code.

WHAT YOU MUST DO:
1. Generate EXECUTABLE SQL queries using the provided database schema
2. Use ONLY the exact table names and column names from the schema
3. Use 'dev.tablename' format for all table references
4. For revenue queries, use 'IngresoNetoSImpuestos' column from 'segmentacion' table
5. Return ONLY valid SQL Server syntax - NO placeholders or template variables
6. Start your response immediately with the SQL query

WHAT YOU MUST NOT DO:
- Do NOT execute the query
- Do NOT return results or data tables
- Do NOT provide explanations before the SQL
- Do NOT use placeholders like [REVENUE_TABLE] or [column_name]

"""
            )
            
            query_executor_agent = ChatCompletionAgent(
                kernel=self.kernel,
                name="ExecutorAgent",
                instructions="""You are a SQL execution specialist. Your role is to:
1. Execute SQL queries safely against the database
2. Validate queries before execution (only SELECT allowed)
3. Handle execution errors gracefully
4. Return formatted query results with metadata
5. Ensure data security and query performance"""
            )
            
            data_analysis_agent = ChatCompletionAgent(
                kernel=self.kernel, 
                name="SummarizingAgent",
                instructions="""You are a data analysis specialist. Your role is to:
1. Analyze query results for business insights
2. Generate executive summaries and key findings
3. Provide actionable business recommendations
4. Create comprehensive data overviews
5. Assess data quality and confidence levels"""
            )
            
            # Create AgentGroupChat with Sequential Selection Strategy
            self.agent_group_chat = AgentGroupChat(
                agents=[schema_analysis_agent, sql_generation_agent, query_executor_agent, data_analysis_agent],
                selection_strategy=SequentialSelectionStrategy()
            )
            
            print("✅ Semantic Kernel Sequential Orchestration initialized successfully")
            
        except Exception as e:
            print(f"❌ Failed to initialize SK orchestration: {str(e)}")
            print("⚠️ Will use manual sequential orchestration as fallback")
            self.agent_group_chat = None
            
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the sequential multi-agent workflow with conversation logging
        
        Args:
            input_data: Dictionary containing:
                - question: Natural language question
                - user_id: User identifier (for conversation logging)
                - session_id: Session identifier (for conversation logging)
                - context: Optional additional context
                - execute: Whether to execute generated SQL
                - limit: Row limit for results
                - include_summary: Whether to generate summary
                - enable_conversation_logging: Whether to log conversations (default: True)
                
        Returns:
            Dictionary containing complete workflow results
        """
        workflow_start_time = time.time()
        workflow_context = None
        
        try:
            question = input_data.get("question", "")
            user_id = input_data.get("user_id", "default_user")
            session_id = input_data.get("session_id", "default_session")
            context = input_data.get("context", "")
            execute = input_data.get("execute", True)
            limit = input_data.get("limit", 100)
            include_summary = input_data.get("include_summary", True)
            enable_conversation_logging = input_data.get("enable_conversation_logging", True)
            
            if not question:
                return self._create_result(
                    success=False,
                    error="No question provided for processing"
                )
            
            print(f"🎯 Orchestrating workflow for: {question}")
            
            # Start workflow session for conversation logging
            if self.memory_service and enable_conversation_logging:
                try:
                    workflow_context = await self.memory_service.start_workflow_session(
                        user_id=user_id,
                        user_input=question,
                        session_id=session_id
                    )
                    print(f"📝 Started workflow session: {workflow_context.workflow_id}")
                except Exception as e:
                    print(f"⚠️ Failed to start workflow session: {e}")
                    workflow_context = None
            
            # Determine orchestration strategy based on concurrency
            # SK AgentGroupChat doesn't handle concurrency well, so use manual for concurrent requests
            use_sk_orchestration = (
                self.agent_group_chat is not None and 
                not self._sk_orchestration_active
            )
            
            if use_sk_orchestration:
                # Try SK orchestration for single requests
                self._sk_orchestration_active = True
                try:
                    print("🔄 Starting Semantic Kernel sequential orchestration...")
                    result = await self._execute_sk_sequential_workflow(input_data)
                finally:
                    self._sk_orchestration_active = False
            else:
                # Use manual orchestration for concurrent requests or if SK fails
                if self._sk_orchestration_active:
                    print("⚠️ SK orchestration busy, using manual sequential orchestration")
                else:
                    print("⚠️ Using manual sequential orchestration")
                result = await self._execute_manual_sequential_workflow(input_data, workflow_context, enable_conversation_logging)
            
            # Complete workflow session with conversation logging
            if self.memory_service and enable_conversation_logging and workflow_context and result.get("success"):
                try:
                    conversation_log = await self._complete_workflow_with_logging(
                        workflow_context=workflow_context,
                        result=result,
                        question=question,
                        workflow_time=time.time() - workflow_start_time
                    )
                    
                    if conversation_log:
                        result["conversation_log_id"] = conversation_log.id
                        print(f"📝 Logged conversation: {conversation_log.id}")
                    
                except Exception as e:
                    print(f"⚠️ Failed to complete conversation logging: {e}")
            
            # Add workflow timing metadata
            if "metadata" not in result:
                result["metadata"] = {}
            result["metadata"]["total_workflow_time"] = round(time.time() - workflow_start_time, 3)
            result["metadata"]["orchestration_pattern"] = "sequential"
            result["metadata"]["workflow_steps"] = self._get_workflow_steps(execute, include_summary)
            result["metadata"]["conversation_logged"] = bool(
                self.memory_service and enable_conversation_logging and workflow_context and result.get("success")
            )
            
            return result
            
        except Exception as e:
            return self._create_result(
                success=False,
                error=f"Orchestration workflow failed: {str(e)}",
                metadata={"total_workflow_time": round(time.time() - workflow_start_time, 3)}
            )
    
    async def _execute_sk_sequential_workflow(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute sequential workflow using Semantic Kernel AgentGroupChat
        """
        try:
            # Get the actual database schema for SK agents
            schema_context = self.schema_analyst.schema_service.get_full_schema_summary()
            
            # Build comprehensive context for the agent group
            workflow_prompt = f"""
SEQUENTIAL NL2SQL WORKFLOW REQUEST:

Question: {params['question']}
Additional Context: {params.get('context', 'None')}

DATABASE SCHEMA (Use these exact table and column names):
{schema_context}

WORKFLOW INSTRUCTIONS:
1. SchemaAnalystAgent: Analyze question to identify relevant tables from the schema above
2. SQLGeneratorAgent: Generate EXECUTABLE SQL query using ONLY the table/column names from the schema above
   - Use dev.tablename format for all tables
   - Use actual column names from the schema (e.g., IngresoNetoSImpuestos for revenue)
   - NO placeholders like [REVENUE_TABLE] or [column_name]
   - Return ONLY the SQL query code - do not execute it
   - Use valid SQL Server syntax only
   - Start your response with the SQL query
3. ExecutorAgent: Take the SQL from SQLGeneratorAgent and execute it to return results
4. SummarizingAgent: Analyze results and provide insights

IMPORTANT: Each agent should perform ONLY their designated step. SQLGeneratorAgent must return SQL code, not results.  
4. SummarizingAgent: Analyze results and provide business insights

Each agent should complete their step and pass results to the next agent.
"""
            
            # Add the user message to the chat history first
            user_message = ChatMessageContent(
                role=AuthorRole.USER, 
                content=workflow_prompt
            )
            await self.agent_group_chat.add_chat_message(user_message)
            
            # Execute the sequential agent group chat with error handling
            agent_responses = []
            try:
                # Execute the group chat and collect responses
                async for response in self.agent_group_chat.invoke():
                    print(f"🤖 {response.name if hasattr(response, 'name') else 'Agent'}: Processing step...")
                    agent_responses.append(response)
                    
                    # Limit responses to prevent infinite loops
                    if len(agent_responses) >= 4:  # We expect 4 agents in sequence (Schema + SQL + Exec + Summary)
                        break
                        
            except Exception as sk_error:
                if "Unable to proceed while another agent is active" in str(sk_error):
                    print("⚠️ SK orchestration busy (another agent active), using manual orchestration")
                    self._sk_orchestration_active = False  # Reset the flag
                    return await self._execute_manual_sequential_workflow(params, None, False)
                else:
                    print(f"❌ SK orchestration error: {str(sk_error)}")
                    return await self._execute_manual_sequential_workflow(params, None, False)
            except Exception as sk_error:
                if "Unable to proceed while another agent is active" in str(sk_error):
                    print("🔄 SK orchestration busy, falling back to manual workflow")
                else:
                    print(f"❌ SK orchestration error: {str(sk_error)}")
                return await self._execute_manual_sequential_workflow(params, None, False)
            
            # Parse and integrate results from all agents
            return await self._parse_sk_workflow_results(agent_responses, params)
            
        except Exception as e:
            print(f"❌ SK sequential workflow failed: {str(e)}")
            print("⚠️ Falling back to manual sequential orchestration")
            return await self._execute_manual_sequential_workflow(params, None, False)
    
    async def _execute_manual_sequential_workflow(self, params: Dict[str, Any], workflow_context=None, enable_conversation_logging=True) -> Dict[str, Any]:
        """
        Execute sequential workflow manually using direct agent calls with Schema Analyst integration
        """
        workflow_results = {
            "schema_analysis": None,
            "sql_generation": None,
            "execution": None, 
            "summarization": None
        }
        
        try:
            # Step 0: Schema Analysis - NEW STEP for optimized context
            print("🔍 Step 0/4: Analyzing schema context...")
            schema_analysis = await self.schema_analyst.process({
                "question": params["question"],
                "context": params.get("context", ""),
                "use_cache": True,
                "similarity_threshold": 0.85
            })
            workflow_results["schema_analysis"] = schema_analysis
            
            # Update workflow with schema analysis results
            if self.memory_service and enable_conversation_logging and workflow_context:
                await self.memory_service.update_workflow_stage(workflow_context.workflow_id, "schema_analysis", schema_analysis)
            
            if not schema_analysis["success"]:
                print(f"⚠️ Schema analysis failed: {schema_analysis['error']}")
                # Continue with fallback to full schema from Schema Analyst
                optimized_schema_context = self.schema_analyst.schema_service.get_full_schema_summary()
                cache_info = "No cache (analysis failed - using full schema)"
            else:
                # Extract optimized schema context
                analysis_data = schema_analysis["data"]
                optimized_schema_context = analysis_data.get("optimized_schema", "")
                
                # Log cache hit information
                cache_info = schema_analysis.get("metadata", {})
                if cache_info.get("cache_hit"):
                    cache_type = cache_info.get("cache_type", "unknown")
                    print(f"✅ Schema analysis: Cache HIT ({cache_type})")
                    if cache_type == "semantic":
                        similarity = cache_info.get("semantic_similarity", 0)
                        print(f"   Semantic similarity: {similarity:.3f}")
                else:
                    print(f"✅ Schema analysis: Fresh analysis ({cache_info.get('analysis_time', 0):.3f}s)")
                    print(f"   Relevant tables: {analysis_data.get('relevant_tables', [])}")
                    print(f"   Confidence score: {analysis_data.get('confidence_score', 0):.3f}")
            
            # Step 1: SQL Generation with optimized context
            print("🧠 Step 1/4: Generating SQL query with optimized schema context...")
            sql_result = await self.sql_generator.process({
                "question": params["question"],
                "context": params.get("context", ""),
                "optimized_schema_context": optimized_schema_context,  # NEW: Pass optimized context
                "schema_analysis": schema_analysis["data"] if schema_analysis["success"] else None
            })
            workflow_results["sql_generation"] = sql_result
            
            if not sql_result["success"]:
                return self._create_result(
                    success=False,
                    error=f"SQL generation failed: {sql_result['error']}",
                    data=workflow_results,
                    metadata={"cache_info": cache_info}
                )
            
            generated_sql = sql_result["data"]["sql_query"]
            print(f"✅ SQL Generated: {generated_sql[:100]}...")
            
            # Update workflow with SQL generation results
            if self.memory_service and enable_conversation_logging and workflow_context:
                await self.memory_service.update_workflow_stage(workflow_context.workflow_id, "sql_generation", sql_result)
            
            # Step 2: SQL Execution (if requested)
            execution_result = None
            if params.get("execute", True):
                print("⚡ Step 2/4: Executing SQL query...")
                execution_result = await self.executor.process({
                    "sql_query": generated_sql,
                    "limit": params.get("limit", 100),
                    "timeout": 30
                })
                workflow_results["execution"] = execution_result
                
                # Update workflow with execution results
                if self.memory_service and enable_conversation_logging and workflow_context:
                    await self.memory_service.update_workflow_stage(workflow_context.workflow_id, "execution", execution_result)
                
                if not execution_result["success"]:
                    return self._create_result(
                        success=False,
                        error=f"SQL execution failed: {execution_result['error']}",
                        data={
                            "sql_query": generated_sql,
                            "execution_error": execution_result["error"]
                        },
                        metadata={**workflow_results, "cache_info": cache_info}
                    )
                print("✅ SQL executed successfully")
            
            # Step 3: Data Summarization (if requested and execution succeeded)
            summarization_result = None
            if params.get("include_summary", True) and execution_result and execution_result["success"]:
                print("📊 Step 3/4: Generating insights and summary...")
                summarization_result = await self.summarizer.process({
                    "raw_results": execution_result["data"]["raw_results"],
                    "formatted_results": execution_result["data"]["formatted_results"],
                    "sql_query": generated_sql,
                    "question": params["question"],
                    "metadata": execution_result["metadata"],
                    "schema_analysis": schema_analysis["data"] if schema_analysis["success"] else None  # NEW: Pass analysis context
                })
                workflow_results["summarization"] = summarization_result
                
                # Update workflow with summarization results
                if self.memory_service and enable_conversation_logging and workflow_context:
                    await self.memory_service.update_workflow_stage(workflow_context.workflow_id, "summarization", summarization_result)
                
                if summarization_result["success"]:
                    print("✅ Summary and insights generated")
                else:
                    print(f"⚠️ Summary generation had issues: {summarization_result['error']}")
            
            # Compile final results with schema analysis metadata
            final_result = self._compile_workflow_results(workflow_results, params)
            
            # Set optimization metadata
            final_result["metadata"]["cache_info"] = cache_info
            final_result["metadata"]["schema_optimization"] = "enabled"
            final_result["metadata"]["schema_source"] = "optimized"
            final_result["metadata"]["schema_context_size"] = len(optimized_schema_context) if optimized_schema_context else 0
            
            # Add analysis timing if available
            if schema_analysis and schema_analysis["success"]:
                analysis_metadata = schema_analysis.get("metadata", {})
                if analysis_metadata.get("analysis_time"):
                    final_result["metadata"]["schema_analysis_time"] = analysis_metadata["analysis_time"]
            
            return final_result
            
        except Exception as e:
            return self._create_result(
                success=False,
                error=f"Manual sequential workflow failed: {str(e)}",
                data=workflow_results,
                metadata={"cache_info": cache_info if 'cache_info' in locals() else "error"}
            )
    
    async def _parse_sk_workflow_results(self, agent_responses: List, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse results from Semantic Kernel AgentGroupChat execution
        """
        try:
            print("🔍 Parsing Semantic Kernel workflow results...")
            
            # In SK 1.34.0, agent responses are ChatMessageContent objects
            # We get AI-generated responses but need to execute them through our actual agents
            # The SK agents provide the AI reasoning, but our specialized agents do the real work
            
            sql_query = None
            
            # Extract SQL from the first agent response (SQLGeneratorAgent)
            for response in agent_responses:
                if hasattr(response, 'name') and 'SQLGenerator' in response.name and hasattr(response, 'content'):
                    content = response.content
                    print(f"🔍 Examining SQLGenerator response: {content[:200]}...")
                    
                    # Enhanced SQL extraction logic to handle various response formats
                    sql_query = self._extract_sql_from_response(content)
                    if sql_query:
                        break
            
            # If no SQL found in SQLGenerator responses, try all responses
            if not sql_query:
                print("🔍 No SQL found in SQLGenerator response, checking all responses...")
                for response in agent_responses:
                    if hasattr(response, 'content'):
                        content = response.content
                        sql_query = self._extract_sql_from_response(content)
                        if sql_query:
                            print(f"🔍 Found SQL in {getattr(response, 'name', 'unknown')} response")
                            break
            
            if sql_query:
                print(f"✅ Extracted SQL from SK workflow: {sql_query[:100]}...")
                print(f"🔍 Full extracted SQL: {sql_query}")
                
                # Clean the extracted SQL to ensure SQL Server compatibility
                cleaned_sql = self.sql_generator._clean_sql_query(sql_query)
                print(f"🧹 Cleaned SQL: {cleaned_sql}")
                
                # Now execute the cleaned SQL using our specialized agents for real execution
                print("🔄 Executing extracted SQL through specialized agents...")
                
                # Use our real ExecutorAgent to execute the SQL
                execution_result = await self.executor.process({
                    "sql_query": cleaned_sql,
                    "limit": params.get("limit", 100),
                    "timeout": 30
                })
                
                print(f"🔍 Execution result success: {execution_result.get('success')}")
                if not execution_result["success"]:
                    print(f"❌ Execution error: {execution_result.get('error')}")
                    return self._create_result(
                        success=False,
                        error=f"SQL execution failed: {execution_result['error']}",
                        metadata={"orchestration_type": "semantic_kernel_hybrid"}
                    )
                
                print(f"✅ Execution successful, row count: {execution_result.get('metadata', {}).get('row_count', 0)}")
                
                # Use our real SummarizingAgent for analysis
                summarization_result = None
                if params.get("include_summary", True):
                    print("📊 Generating summary through specialized agent...")
                    summarization_result = await self.summarizer.process({
                        "raw_results": execution_result["data"]["raw_results"],
                        "formatted_results": execution_result["data"]["formatted_results"],
                        "sql_query": sql_query,
                        "question": params["question"],
                        "metadata": execution_result["metadata"]
                    })
                    print(f"🔍 Summary result success: {summarization_result.get('success') if summarization_result else 'None'}")
                
                # Compile results using our standard format with optimization metadata
                workflow_results = {
                    "sql_generation": {
                        "success": True,
                        "data": {"sql_query": cleaned_sql},
                        "agent": "SKSQLGeneratorAgent"
                    },
                    "execution": execution_result,
                    "summarization": summarization_result
                }
                
                result = self._compile_workflow_results(workflow_results, params)
                result["metadata"]["orchestration_type"] = "semantic_kernel_hybrid"
                result["metadata"]["schema_optimization"] = "enabled"
                result["metadata"]["schema_source"] = "optimized"
                result["metadata"]["schema_context_size"] = len(str(cleaned_sql))  # Approximate context size
                result["metadata"]["steps_completed"] = ["schema_analysis", "sql_generation"]
                
                # Add execution step if successful
                if execution_result and execution_result["success"]:
                    result["metadata"]["steps_completed"].append("execution")
                
                # Add summarization step if successful
                if summarization_result and summarization_result["success"]:
                    result["metadata"]["steps_completed"].append("summarization")
                
                print("🎉 Successfully completed SK hybrid workflow!")
                
                # Run our real Schema Analyst to get actual analysis results for the question
                # This ensures we have proper schema analysis data in the final result
                print("🔍 Running Schema Analyst to capture analysis results...")
                schema_analysis_result = await self.schema_analyst.process({
                    "question": params["question"],
                    "context": params.get("context", ""),
                    "use_cache": True,
                    "similarity_threshold": 0.85
                })
                
                # Include the actual schema analysis in workflow results
                workflow_results["schema_analysis"] = schema_analysis_result
                
                # Recompile results with schema analysis included
                result = self._compile_workflow_results(workflow_results, params)
                result["metadata"]["orchestration_type"] = "semantic_kernel_hybrid"
                result["metadata"]["schema_optimization"] = "enabled"
                result["metadata"]["schema_source"] = "optimized"
                result["metadata"]["schema_context_size"] = len(str(cleaned_sql))  # Approximate context size
                result["metadata"]["steps_completed"] = ["schema_analysis", "sql_generation"]
                
                # Add schema analysis metadata if available
                if schema_analysis_result and schema_analysis_result["success"]:
                    schema_metadata = schema_analysis_result.get("metadata", {})
                    result["metadata"].update({
                        "schema_cache_hit": schema_metadata.get("cache_hit", False),
                        "schema_cache_type": schema_metadata.get("cache_type"),
                        "schema_analysis_time": schema_metadata.get("analysis_time"),
                        "schema_confidence": schema_analysis_result["data"].get("confidence_score", 0)
                    })
                
                # Add execution step if successful
                if execution_result and execution_result["success"]:
                    result["metadata"]["steps_completed"].append("execution")
                
                # Add summarization step if successful
                if summarization_result and summarization_result["success"]:
                    result["metadata"]["steps_completed"].append("summarization")
                
                return result
                
            else:
                print("⚠️ Could not extract SQL from SK responses, falling back to manual workflow")
                return await self._execute_manual_sequential_workflow(params, None, False)
                
        except Exception as e:
            print(f"❌ SK result parsing failed: {str(e)}")
            print("⚠️ Falling back to manual workflow for result compilation")
            return await self._execute_manual_sequential_workflow(params, None, False)
    
    def _extract_sql_from_response(self, content: str) -> str:
        """
        Enhanced SQL extraction from agent response content
        Handles various formats including markdown, explanatory text, and direct SQL
        """
        if not content or 'SELECT' not in content.upper():
            return None
        
        # Strategy 1: Extract from ```sql markdown blocks
        sql_pattern = r'```sql\s*(.*?)\s*```'
        matches = re.findall(sql_pattern, content, re.DOTALL | re.IGNORECASE)
        if matches:
            sql_candidate = matches[0].strip()
            if self._is_valid_sql_candidate(sql_candidate):
                return sql_candidate
        
        # Strategy 2: Extract from plain ``` code blocks containing SELECT
        code_pattern = r'```\s*(.*?)\s*```'
        matches = re.findall(code_pattern, content, re.DOTALL)
        for match in matches:
            match = match.strip()
            if 'SELECT' in match.upper() and self._is_valid_sql_candidate(match):
                return match
        
        # Strategy 3: Extract multiline SELECT statements from free text
        lines = content.split('\n')
        sql_lines = []
        in_sql_block = False
        
        for i, line in enumerate(lines):
            line = line.strip()
            
            # Start collecting when we see SELECT or WITH (but don't restart if already collecting)
            if not in_sql_block and (line.upper().startswith('SELECT') or (line.upper().startswith('WITH') and i < len(lines) - 1 and 'SELECT' in lines[i+1].upper())):
                in_sql_block = True
                sql_lines = [line]
                continue
            
            if in_sql_block:
                # Continue collecting SQL lines
                if line:
                    sql_lines.append(line)
                
                # Stop conditions
                if (line.endswith(';') or 
                    line.upper().startswith('LIMIT') or 
                    line.upper().startswith('ORDER BY') and ';' in line or
                    i == len(lines) - 1):  # End of content
                    
                    sql_candidate = '\n'.join(sql_lines)
                    if self._is_valid_sql_candidate(sql_candidate):
                        return sql_candidate
                    else:
                        # Reset and continue looking
                        in_sql_block = False
                        sql_lines = []
                
                # Reset if we hit explanatory text that indicates end of SQL
                elif (line.upper().startswith('THIS QUERY') or 
                      line.upper().startswith('EXPLANATION') or
                      line.upper().startswith('NOTE:') or
                      line.upper().startswith('THE ABOVE') or
                      line.upper().startswith('STEP ') or
                      line.upper().startswith('**STEP')):
                    
                    sql_candidate = '\n'.join(sql_lines[:-1])  # Exclude the explanatory line
                    if self._is_valid_sql_candidate(sql_candidate):
                        return sql_candidate
                    else:
                        in_sql_block = False
                        sql_lines = []
        
        # Strategy 4: Look for single-line SELECT statements
        for line in lines:
            line = line.strip()
            if (line.upper().startswith('SELECT') and 
                ('FROM' in line.upper() or 'TOP' in line.upper()) and
                len(line) > 30):  # Likely a complete SQL statement
                return line
        
        return None
    
    def _is_valid_sql_candidate(self, sql_text: str) -> bool:
        """
        Check if extracted text is likely a valid SQL query
        """
        if not sql_text or len(sql_text.strip()) < 20:
            return False
        
        sql_upper = sql_text.upper()
        
        # Must contain SELECT
        if 'SELECT' not in sql_upper:
            return False
        
        # Should contain FROM (basic SQL structure)
        if 'FROM' not in sql_upper:
            return False
        
        # Should not contain too much explanatory text
        explanatory_keywords = ['STEP ', 'NOTE:', 'EXPLANATION', 'THIS QUERY', 'THE ABOVE', '**STEP']
        explanatory_ratio = sum(1 for keyword in explanatory_keywords if keyword in sql_upper)
        if explanatory_ratio > 2:  # Too much explanation mixed in
            return False
        
        # Check for basic SQL keywords
        sql_keywords = ['SELECT', 'FROM', 'WHERE', 'GROUP', 'ORDER', 'JOIN', 'AS']
        keyword_count = sum(1 for keyword in sql_keywords if keyword in sql_upper)
        
        # Should have at least 2 SQL keywords
        return keyword_count >= 2
    
    def _compile_workflow_results(self, workflow_results: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compile results from all workflow steps into final response
        """
        schema_analysis = workflow_results.get("schema_analysis")
        sql_generation = workflow_results.get("sql_generation")
        execution = workflow_results.get("execution") 
        summarization = workflow_results.get("summarization")
        
        # Base response data
        response_data = {
            "sql_query": sql_generation["data"]["sql_query"] if sql_generation and sql_generation["success"] else None,
            "executed": False,
            "results": None,
            "formatted_results": None
        }
        
        # Add schema analysis results if available
        if schema_analysis and schema_analysis["success"]:
            response_data["schema_analysis"] = {
                "relevant_tables": schema_analysis["data"].get("relevant_tables", []),
                "join_strategy": schema_analysis["data"].get("join_strategy", {}),
                "performance_hints": schema_analysis["data"].get("performance_hints", []),
                "confidence_score": schema_analysis["data"].get("confidence_score", 0)
            }
        
        # Add execution results if available
        if execution and execution["success"]:
            response_data.update({
                "executed": True,
                "results": execution["data"]["raw_results"],
                "formatted_results": execution["data"]["formatted_results"]
            })
        elif execution and not execution["success"]:
            response_data["execution_error"] = execution["error"]
        
        # Add summary results if available
        if summarization and summarization["success"]:
            response_data["summary"] = {
                "executive_summary": summarization["data"]["executive_summary"],
                "key_insights": summarization["data"]["key_insights"], 
                "recommendations": summarization["data"]["recommendations"],
                "data_overview": summarization["data"]["data_overview"],
                "technical_summary": summarization["data"]["technical_summary"]
            }
        
        # Compile metadata
        metadata = {
            "workflow_success": True,
            "orchestration_type": "schema_optimized",
            "steps_completed": [step for step, result in workflow_results.items() if result and result["success"]],
            "schema_analyzed": bool(schema_analysis and schema_analysis["success"]),
            "sql_generated": bool(sql_generation and sql_generation["success"]),
            "query_executed": bool(execution and execution["success"]),
            "summary_generated": bool(summarization and summarization["success"])
        }
        
        # Add schema analysis metadata if available
        if schema_analysis and schema_analysis["success"]:
            schema_metadata = schema_analysis.get("metadata", {})
            metadata.update({
                "schema_cache_hit": schema_metadata.get("cache_hit", False),
                "schema_cache_type": schema_metadata.get("cache_type"),
                "schema_analysis_time": schema_metadata.get("analysis_time"),
                "schema_confidence": schema_analysis["data"].get("confidence_score")
            })
        
        # Add execution metadata if available
        if execution and execution["success"]:
            exec_metadata = execution.get("metadata", {})
            metadata.update({
                "execution_time": exec_metadata.get("execution_time"),
                "row_count": exec_metadata.get("row_count"),
                "query_type": exec_metadata.get("query_type")
            })
        
        return self._create_result(
            success=True,
            data=response_data,
            metadata=metadata
        )
    
    async def _complete_workflow_with_logging(self, workflow_context, result: Dict[str, Any], 
                                             question: str, workflow_time: float) -> Optional[Any]:
        """
        Complete the workflow session with conversation logging
        """
        try:
            # Extract results from workflow
            data = result.get("data", {})
            
            # Create FormattedResults if available
            formatted_results = None
            if "formatted_results" in data and data["formatted_results"]:
                if isinstance(data["formatted_results"], dict):
                    formatted_results = FormattedResults(**data["formatted_results"])
                else:
                    formatted_results = data["formatted_results"]
            
            # Create AgentResponse from summary data
            agent_response = AgentResponse(
                agent_type="orchestrator",
                response="Workflow completed successfully",
                executive_summary=data.get("summary", {}).get("executive_summary", "Query processed successfully"),
                key_insights=data.get("summary", {}).get("key_insights", ["Data retrieved and analyzed"]),
                recommendations=data.get("summary", {}).get("recommendations", ["Review results for insights"]),
                confidence_level=data.get("summary", {}).get("confidence_level", "high")
            )
            
            # Complete workflow session
            conversation_log = await self.memory_service.complete_workflow_session(
                workflow_context=workflow_context,
                formatted_results=formatted_results,
                agent_response=agent_response,
                sql_query=data.get("sql_query"),
                processing_time_ms=int(workflow_time * 1000)
            )
            
            return conversation_log
            
        except Exception as e:
            print(f"❌ Error completing workflow logging: {e}")
            return None
    
    def set_memory_service(self, memory_service: OrchestratorMemoryService):
        """
        Set the memory service for conversation logging
        """
        self.memory_service = memory_service
        print("✅ Memory service configured for conversation logging")
    
    async def get_conversation_history(self, user_id: str, session_id: str = None, limit: int = 10) -> List:
        """
        Get conversation history for a user/session
        """
        if not self.memory_service:
            return []
        
        try:
            # Use the cosmos service directly for conversation retrieval
            cosmos_service = self.memory_service.cosmos_service
            conversations = await cosmos_service.get_user_conversations_async(
                user_id=user_id,
                session_id=session_id,
                limit=limit
            )
            return conversations
        except Exception as e:
            print(f"❌ Error retrieving conversation history: {e}")
            return []
    
    async def get_user_analytics(self, user_id: str, days: int = 30):
        """
        Get user analytics from conversation history
        """
        if not self.memory_service:
            return None
        
        try:
            analytics = await self.memory_service.get_user_analytics_enhanced(user_id, days)
            return analytics
        except Exception as e:
            print(f"❌ Error retrieving user analytics: {e}")
            return None
    
    def _get_workflow_steps(self, execute: bool, include_summary: bool) -> List[str]:
        """
        Get list of workflow steps based on parameters
        """
        steps = ["schema_analysis", "sql_generation"]
        if execute:
            steps.append("execution")
            if include_summary:
                steps.append("summarization")
        return steps
    
    async def get_workflow_status(self) -> Dict[str, Any]:
        """
        Get status of the orchestration system
        """
        return self._create_result(
            success=True,
            data={
                "orchestrator": "active",
                "agents": {
                    "schema_analyst": "ready",
                    "sql_generator": "ready",
                    "executor": "ready", 
                    "summarizer": "ready"
                },
                "orchestration_mode": "schema_optimized",
                "workflow_capabilities": {
                    "schema_analysis": True,
                    "intelligent_caching": True,
                    "optimized_context": True,
                    "sql_generation": True,
                    "query_execution": True,
                    "data_summarization": True,
                    "sequential_processing": True
                }
            }
        )



