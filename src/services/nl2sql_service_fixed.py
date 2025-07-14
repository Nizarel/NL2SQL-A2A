"""
NL2SQL Service - Core Natural Language to SQL conversion service
Uses Semantic Kernel with database schema context for accurate SQL generation
"""

import re
import time
from typing import Dict, List, Any, Optional
from semantic_kernel import Kernel
from semantic_kernel.functions import KernelArguments
from semantic_kernel.contents import ChatHistory

from services.schema_service import SchemaService
from plugins.mcp_database_plugin import MCPDatabasePlugin


class NL2SQLService:
    """
    Service that converts natural language questions to SQL queries
    using Semantic Kernel and database schema context
    """
    
    def __init__(self, kernel: Kernel, schema_service: SchemaService, mcp_plugin: MCPDatabasePlugin):
        self.kernel = kernel
        self.schema_service = schema_service
        self.mcp_plugin = mcp_plugin
    
    def _create_sql_generation_prompt(self, question: str, schema_context: str) -> str:
        """
        Create the prompt for SQL generation
        """
        return f"""
You are an expert SQL query generator for a business analytics database.

DATABASE CONTEXT:
{schema_context}

BUSINESS DOMAIN KNOWLEDGE:
- This is a beverage/retail analytics database with customer and sales data
- Common business questions involve customer analysis, product performance, market territories
- Date-based analysis often uses the tiempo table
- Customer analysis uses cliente table with cliente_cedi for location data
- Product analysis uses producto table
- Market/territory analysis uses mercado with cliente_cedi

USER QUESTION: {question}

Generate a SQL query that:
1. Accurately answers the user's question
2. Uses proper SQL Server syntax
3. Includes appropriate JOINs based on table relationships
4. Uses meaningful aliases and column names
5. Is optimized for performance

Return ONLY the SQL query without any explanations or markdown formatting.
"""
    
    async def convert_question_to_sql(self, question: str, execute: bool = True, limit: int = 100) -> Dict[str, Any]:
        """
        Convert natural language question to SQL query and optionally execute it
        
        Args:
            question: Natural language question
            execute: Whether to execute the generated SQL
            limit: Maximum number of rows to return
            
        Returns:
            Dictionary containing SQL query, execution results, and metadata
        """
        try:
            # Get relevant schema context
            schema_context = await self._get_relevant_schema_context(question)
            
            # Generate SQL using the AI service
            sql_query = await self._generate_sql_with_kernel(question, schema_context)
            
            # Clean and validate the SQL
            cleaned_sql = self._clean_sql_query(sql_query)
            
            result = {
                "question": question,
                "sql_query": cleaned_sql,
                "executed": False,
                "results": None,
                "error": None,
                "row_count": 0,
                "execution_time": 0
            }
            
            # Execute the SQL if requested
            if execute:
                execution_result = await self._execute_sql_query(cleaned_sql, limit)
                result.update(execution_result)
            
            return result
            
        except Exception as e:
            return {
                "question": question,
                "sql_query": "",
                "executed": False,
                "results": None,
                "error": f"Error converting question to SQL: {str(e)}",
                "row_count": 0,
                "execution_time": 0
            }
    
    async def _get_relevant_schema_context(self, question: str) -> str:
        """
        Get relevant schema context based on the question content
        """
        # Use the schema service to get full context
        return self.schema_service.get_full_schema_summary()
    
    async def _generate_sql_with_kernel(self, question: str, schema_context: str) -> str:
        """
        Generate SQL using the AI service from Semantic Kernel
        """
        # Get the AI service from the kernel
        ai_service = self.kernel.get_service()
        
        # Create the prompt
        full_prompt = self._create_sql_generation_prompt(question, schema_context)
        
        # Generate SQL using chat completion
        chat_history = ChatHistory()
        chat_history.add_user_message(full_prompt)
        
        # Create settings for the AI service
        from semantic_kernel.connectors.ai.open_ai import OpenAIChatPromptExecutionSettings
        settings = OpenAIChatPromptExecutionSettings(
            max_tokens=1000,
            temperature=0.1
        )
        
        # Get response from AI service
        response = await ai_service.get_chat_message_content(
            chat_history=chat_history,
            settings=settings,
            kernel=self.kernel
        )
        
        return str(response).strip()
    
    def _clean_sql_query(self, sql_query: str) -> str:
        """
        Clean and validate the generated SQL query
        """
        # Remove markdown formatting if present
        sql_query = re.sub(r'```sql\s*', '', sql_query)
        sql_query = re.sub(r'```\s*', '', sql_query)
        
        # Remove extra whitespace and normalize
        sql_query = re.sub(r'\s+', ' ', sql_query.strip())
        
        # Ensure it ends with semicolon if it doesn't have one
        if not sql_query.endswith(';'):
            sql_query += ';'
        
        return sql_query
    
    async def _execute_sql_query(self, sql_query: str, limit: int) -> Dict[str, Any]:
        """
        Execute the SQL query using the MCP database plugin
        """
        start_time = time.time()
        
        try:
            # Execute query through MCP plugin
            result = await self.mcp_plugin.read_data(sql_query, limit)
            
            # Parse results to count rows (basic implementation)
            row_count = 0
            if result and isinstance(result, str):
                lines = result.strip().split('\n')
                row_count = max(0, len(lines) - 1)  # Subtract header row
            
            return {
                "executed": True,
                "results": result,
                "error": None,
                "row_count": row_count,
                "execution_time": time.time() - start_time
            }
            
        except Exception as e:
            return {
                "executed": False,
                "results": None,
                "error": f"Query execution failed: {str(e)}",
                "row_count": 0,
                "execution_time": time.time() - start_time
            }
