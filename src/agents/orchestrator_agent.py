"""
Orchestrator Agent - Coordinates multi-agent workflow using Semantic Kernel 1.34.0
Sequential Pattern: SQLGenerator → Executor → Summarizer
"""

import time
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


class OrchestratorAgent(BaseAgent):
    """
    Agent responsible for orchestrating the sequential multi-agent workflow:
    1. SQLGeneratorAgent: Converts natural language to SQL
    2. ExecutorAgent: Executes the generated SQL query
    3. SummarizingAgent: Analyzes results and generates insights
    """
    
    def __init__(self, kernel: Kernel, sql_generator: SQLGeneratorAgent, 
                 executor: ExecutorAgent, summarizer: SummarizingAgent):
        super().__init__(kernel, "OrchestratorAgent")
        
        # Store agent references
        self.sql_generator = sql_generator
        self.executor = executor
        self.summarizer = summarizer
        
        # Initialize Semantic Kernel AgentGroupChat for orchestration
        self.agent_group_chat: Optional[AgentGroupChat] = None
        self._initialize_sk_orchestration()
        
    def _initialize_sk_orchestration(self):
        """
        Initialize Semantic Kernel AgentGroupChat with Sequential Selection Strategy
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
            sql_generation_agent = ChatCompletionAgent(
                kernel=self.kernel,
                name="SQLGeneratorAgent",
                instructions="""You are a SQL generation specialist. Your role is to:
1. Analyze natural language questions about data
2. Generate accurate SQL queries using the provided database schema
3. Ensure all table names use the 'dev.' schema prefix
4. Focus only on SELECT queries for data retrieval
5. Return well-formatted, executable SQL queries"""
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
                agents=[sql_generation_agent, query_executor_agent, data_analysis_agent],
                selection_strategy=SequentialSelectionStrategy()
            )
            
            print("✅ Semantic Kernel Sequential Orchestration initialized successfully")
            
        except Exception as e:
            print(f"❌ Failed to initialize SK orchestration: {str(e)}")
            print("⚠️ Will use manual sequential orchestration as fallback")
            self.agent_group_chat = None
            
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the sequential multi-agent workflow
        
        Args:
            input_data: Dictionary containing:
                - question: Natural language question
                - context: Optional additional context
                - execute: Whether to execute generated SQL
                - limit: Row limit for results
                - include_summary: Whether to generate summary
                
        Returns:
            Dictionary containing complete workflow results
        """
        workflow_start_time = time.time()
        
        try:
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
            
            print(f"🎯 Orchestrating workflow for: {question}")
            
            # Execute sequential workflow
            if self.agent_group_chat:
                # Use Semantic Kernel AgentGroupChat orchestration
                result = await self._execute_sk_sequential_workflow(input_data)
            else:
                # Fallback to manual sequential orchestration
                print("⚠️ Using manual sequential orchestration")
                result = await self._execute_manual_sequential_workflow(input_data)
            
            # Add workflow timing metadata
            if "metadata" not in result:
                result["metadata"] = {}
            result["metadata"]["total_workflow_time"] = round(time.time() - workflow_start_time, 3)
            result["metadata"]["orchestration_pattern"] = "sequential"
            result["metadata"]["workflow_steps"] = self._get_workflow_steps(execute, include_summary)
            
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
            # Build comprehensive context for the agent group
            schema_context = self.sql_generator.schema_service.get_full_schema_summary()
            
            workflow_prompt = f"""
SEQUENTIAL NL2SQL WORKFLOW REQUEST:

Question: {params['question']}
Additional Context: {params.get('context', 'None')}
Execution Parameters:
- Execute Query: {params['execute']}
- Row Limit: {params['limit']} 
- Include Summary: {params['include_summary']}

Database Schema Context:
{schema_context}

WORKFLOW STEPS (Execute in this exact sequence):
1. SQLGeneratorAgent: Generate SQL query from the question
2. ExecutorAgent: Execute the SQL query and return results  
3. SummarizingAgent: Analyze results and provide business insights

Each agent should complete their step and pass results to the next agent.
"""
            
            print("🔄 Starting Semantic Kernel sequential orchestration...")
            
            # Add the user message to the chat history first
            user_message = ChatMessageContent(
                role=AuthorRole.USER, 
                content=workflow_prompt
            )
            await self.agent_group_chat.add_chat_message(user_message)
            
            # Execute the sequential agent group chat (no parameters needed)
            agent_responses = []
            async for response in self.agent_group_chat.invoke():
                print(f"🤖 {response.name if hasattr(response, 'name') else 'Agent'}: Processing step...")
                agent_responses.append(response)
                
                # Optional: limit responses to prevent infinite loops
                if len(agent_responses) >= 3:  # We expect 3 agents in sequence
                    break
            
            # Parse and integrate results from all agents
            return await self._parse_sk_workflow_results(agent_responses, params)
            
        except Exception as e:
            print(f"❌ SK sequential workflow failed: {str(e)}")
            print("⚠️ Falling back to manual sequential orchestration")
            return await self._execute_manual_sequential_workflow(params)
    
    async def _execute_manual_sequential_workflow(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute sequential workflow manually using direct agent calls
        """
        workflow_results = {
            "sql_generation": None,
            "execution": None, 
            "summarization": None
        }
        
        try:
            # Step 1: SQL Generation
            print("🧠 Step 1/3: Generating SQL query...")
            sql_result = await self.sql_generator.process({
                "question": params["question"],
                "context": params.get("context", "")
            })
            workflow_results["sql_generation"] = sql_result
            
            if not sql_result["success"]:
                return self._create_result(
                    success=False,
                    error=f"SQL generation failed: {sql_result['error']}",
                    data=workflow_results
                )
            
            generated_sql = sql_result["data"]["sql_query"]
            print(f"✅ SQL Generated: {generated_sql[:100]}...")
            
            # Step 2: SQL Execution (if requested)
            execution_result = None
            if params.get("execute", True):
                print("⚡ Step 2/3: Executing SQL query...")
                execution_result = await self.executor.process({
                    "sql_query": generated_sql,
                    "limit": params.get("limit", 100),
                    "timeout": 30
                })
                workflow_results["execution"] = execution_result
                
                if not execution_result["success"]:
                    return self._create_result(
                        success=False,
                        error=f"SQL execution failed: {execution_result['error']}",
                        data={
                            "sql_query": generated_sql,
                            "execution_error": execution_result["error"]
                        },
                        metadata=workflow_results
                    )
                print("✅ SQL executed successfully")
            
            # Step 3: Data Summarization (if requested and execution succeeded)
            summarization_result = None
            if params.get("include_summary", True) and execution_result and execution_result["success"]:
                print("📊 Step 3/3: Generating insights and summary...")
                summarization_result = await self.summarizer.process({
                    "raw_results": execution_result["data"]["raw_results"],
                    "formatted_results": execution_result["data"]["formatted_results"],
                    "sql_query": generated_sql,
                    "question": params["question"],
                    "metadata": execution_result["metadata"]
                })
                workflow_results["summarization"] = summarization_result
                
                if summarization_result["success"]:
                    print("✅ Summary and insights generated")
                else:
                    print(f"⚠️ Summary generation had issues: {summarization_result['error']}")
            
            # Compile final results
            return self._compile_workflow_results(workflow_results, params)
            
        except Exception as e:
            return self._create_result(
                success=False,
                error=f"Manual sequential workflow failed: {str(e)}",
                data=workflow_results
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
                    
                    # Simple extraction of SQL from markdown code blocks
                    if '```sql' in content:
                        sql_start = content.find('```sql') + 6
                        sql_end = content.find('```', sql_start)
                        if sql_end > sql_start:
                            sql_query = content[sql_start:sql_end].strip()
                    elif '```' in content and 'SELECT' in content.upper():
                        # Handle plain code blocks with SELECT
                        sql_start = content.find('```') + 3
                        sql_end = content.find('```', sql_start)
                        if sql_end > sql_start:
                            potential_sql = content[sql_start:sql_end].strip()
                            if 'SELECT' in potential_sql.upper():
                                sql_query = potential_sql
                    elif 'SELECT' in content.upper():
                        # Extract SELECT statement directly - look for complete statement
                        lines = content.split('\n')
                        sql_lines = []
                        in_sql = False
                        
                        for line in lines:
                            line = line.strip()
                            if line.upper().startswith('SELECT'):
                                in_sql = True
                            if in_sql and line:
                                sql_lines.append(line)
                                # Stop at semicolon or if we see typical SQL ending patterns
                                if line.endswith(';') or line.upper().startswith('LIMIT'):
                                    break
                        
                        if sql_lines and len(' '.join(sql_lines)) > 20:  # Ensure meaningful SQL
                            sql_query = '\n'.join(sql_lines)
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
                
                # Compile results using our standard format
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
                print("🎉 Successfully completed SK hybrid workflow!")
                return result
                
            else:
                print("⚠️ Could not extract SQL from SK responses, falling back to manual workflow")
                return await self._execute_manual_sequential_workflow(params)
                
        except Exception as e:
            print(f"❌ SK result parsing failed: {str(e)}")
            print("⚠️ Falling back to manual workflow for result compilation")
            return await self._execute_manual_sequential_workflow(params)
    
    def _compile_workflow_results(self, workflow_results: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compile results from all workflow steps into final response
        """
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
            "orchestration_type": "semantic_kernel" if self.agent_group_chat else "manual_sequential",
            "steps_completed": [step for step, result in workflow_results.items() if result and result["success"]],
            "sql_generated": bool(sql_generation and sql_generation["success"]),
            "query_executed": bool(execution and execution["success"]),
            "summary_generated": bool(summarization and summarization["success"])
        }
        
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
        Get status of the orchestration system
        """
        return self._create_result(
            success=True,
            data={
                "orchestrator": "active",
                "agents": {
                    "sql_generator": "ready",
                    "executor": "ready", 
                    "summarizer": "ready"
                },
                "orchestration_mode": "semantic_kernel" if self.agent_group_chat else "manual_sequential",
                "workflow_capabilities": {
                    "sql_generation": True,
                    "query_execution": True,
                    "data_summarization": True,
                    "sequential_processing": True
                }
            }
        )



