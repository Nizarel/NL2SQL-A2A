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

    async def convert_question_to_sql(self, question: str, execute: bool = False, limit: Optional[int] = None) -> Dict[str, Any]:
        """
        Convert natural language question to SQL and optionally execute it
        """
        start_time = time.time()
        
        try:
            # Get database schema context
            schema_context = await self.schema_service.get_schema_context()
            
            # Generate SQL using Semantic Kernel
            prompt = f"""
You are an expert SQL query generator. Convert this question to SQL:

Database Schema:
{schema_context}

Question: {question}

Return only the SQL query:
"""
            
            # Use chat completion to generate SQL
            from semantic_kernel.contents import ChatHistory
            chat_history = ChatHistory()
            chat_history.add_user_message(prompt)
            
            # Get AI service from kernel
            ai_service = self.kernel.get_service()
            response = await ai_service.get_chat_message_content(
                chat_history=chat_history,
                kernel=self.kernel
            )
            
            # Extract SQL from response
            sql_query = str(response).strip()
            
            if not sql_query:
                return {
                    "error": "Failed to generate SQL query",
                    "sql_query": None,
                    "executed": False,
                    "results": None,
                    "execution_time": time.time() - start_time,
                    "row_count": 0
                }
            
            result = {
                "error": None,
                "sql_query": sql_query,
                "executed": False,
                "results": None,
                "execution_time": time.time() - start_time,
                "row_count": 0
            }
            
            # Execute query if requested
            if execute:
                try:
                    execution_start = time.time()
                    
                    # Apply limit if specified
                    if limit and 'LIMIT' not in sql_query.upper():
                        sql_query += f" LIMIT {limit}"
                    
                    # Execute query through MCP plugin
                    query_result = await self.mcp_plugin.read_data(sql_query, limit or 100)
                    
                    result.update({
                        "executed": True,
                        "results": query_result,
                        "execution_time": time.time() - execution_start,
                        "row_count": len(query_result.split('\n')) - 1 if query_result else 0
                    })
                    
                except Exception as e:
                    result["error"] = f"Query execution failed: {str(e)}"
            
            return result
            
        except Exception as e:
            return {
                "error": f"SQL generation failed: {str(e)}",
                "sql_query": None,
                "executed": False,
                "results": None,
                "execution_time": time.time() - start_time,
                "row_count": 0
            }
