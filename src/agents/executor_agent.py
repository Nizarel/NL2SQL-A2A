"""
Executor Agent - Executes SQL queries and handles database operations
"""

import time
from typing import Dict, Any
from semantic_kernel import Kernel

from agents.base_agent import BaseAgent
from plugins.mcp_database_plugin import MCPDatabasePlugin


class ExecutorAgent(BaseAgent):
    """
    Agent responsible for executing SQL queries against the database
    """
    
    def __init__(self, kernel: Kernel, mcp_plugin: MCPDatabasePlugin):
        super().__init__(kernel, "ExecutorAgent")
        self.mcp_plugin = mcp_plugin
        
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute SQL query and return results
        
        Args:
            input_data: Dictionary containing:
                - sql_query: SQL query to execute
                - limit: Maximum number of rows to return (optional)
                - timeout: Query timeout in seconds (optional)
                
        Returns:
            Dictionary containing execution results and metadata
        """
        try:
            sql_query = input_data.get("sql_query", "")
            limit = input_data.get("limit", 100)
            timeout = input_data.get("timeout", 30)
            
            if not sql_query:
                return self._create_result(
                    success=False,
                    error="No SQL query provided"
                )
            
            # Validate SQL query
            validation_result = await self._validate_sql_query(sql_query)
            if not validation_result["is_valid"]:
                return self._create_result(
                    success=False,
                    error=f"SQL validation failed: {validation_result['error']}"
                )
            
            # Execute the query
            execution_result = await self._execute_query(sql_query, limit, timeout)
            
            if execution_result["success"]:
                return self._create_result(
                    success=True,
                    data={
                        "raw_results": execution_result["results"],
                        "formatted_results": execution_result["formatted_results"],
                        "sql_query": sql_query
                    },
                    metadata={
                        "execution_time": execution_result["execution_time"],
                        "row_count": execution_result["row_count"],
                        "query_type": validation_result["query_type"],
                        "columns": execution_result.get("columns", [])
                    }
                )
            else:
                return self._create_result(
                    success=False,
                    error=execution_result["error"],
                    metadata={
                        "sql_query": sql_query,
                        "execution_time": execution_result.get("execution_time", 0)
                    }
                )
                
        except Exception as e:
            return self._create_result(
                success=False,
                error=f"Query execution failed: {str(e)}"
            )
    
    async def _validate_sql_query(self, sql_query: str) -> Dict[str, Any]:
        """
        Validate SQL query before execution
        """
        try:
            # Basic SQL validation
            sql_upper = sql_query.upper().strip()
            
            # Check for dangerous operations (basic security)
            dangerous_keywords = ['DROP', 'DELETE', 'TRUNCATE', 'ALTER', 'CREATE', 'INSERT', 'UPDATE']
            for keyword in dangerous_keywords:
                if keyword in sql_upper:
                    return {
                        "is_valid": False,
                        "error": f"Dangerous operation detected: {keyword}. Only SELECT queries are allowed.",
                        "query_type": "DANGEROUS"
                    }
            
            # Ensure it's a SELECT query
            if not sql_upper.startswith('SELECT'):
                return {
                    "is_valid": False,
                    "error": "Only SELECT queries are allowed",
                    "query_type": "NON_SELECT"
                }
            
            # Check for required dev. prefix
            if 'FROM ' in sql_upper and 'dev.' not in sql_query.lower():
                return {
                    "is_valid": False,
                    "error": "Queries must use 'dev.' schema prefix for table names",
                    "query_type": "INVALID_SCHEMA"
                }
            
            return {
                "is_valid": True,
                "error": None,
                "query_type": self._determine_query_type(sql_query)
            }
            
        except Exception as e:
            return {
                "is_valid": False,
                "error": f"Validation error: {str(e)}",
                "query_type": "VALIDATION_ERROR"
            }
    
    async def _execute_query(self, sql_query: str, limit: int, timeout: int) -> Dict[str, Any]:
        """
        Execute SQL query with proper error handling and metrics
        """
        start_time = time.time()
        
        try:
            # Execute query via MCP plugin
            result = await self.mcp_plugin.read_data(sql_query, limit)
            
            execution_time = time.time() - start_time
            
            # Extract the actual text data from CallToolResult
            if hasattr(result, 'data'):
                result_text = result.data
            elif hasattr(result, 'content') and result.content:
                result_text = result.content[0].text if result.content else str(result)
            else:
                result_text = str(result)
            
            # Parse and format results
            formatted_results = self._format_query_results(result_text)
            row_count = self._count_rows_in_result(result_text)
            columns = self._extract_columns_from_result(result_text)
            
            return {
                "success": True,
                "results": result,
                "formatted_results": formatted_results,
                "execution_time": round(execution_time, 3),
                "row_count": row_count,
                "columns": columns,
                "error": None
            }
            
        except Exception as e:
            execution_time = time.time() - start_time
            return {
                "success": False,
                "results": None,
                "formatted_results": None,
                "execution_time": round(execution_time, 3),
                "row_count": 0,
                "columns": [],
                "error": f"Database execution error: {str(e)}"
            }
    
    def _format_query_results(self, raw_result: str) -> Dict[str, Any]:
        """
        Format raw query results into structured data
        """
        try:
            if not raw_result or "Query error" in raw_result:
                return {
                    "status": "error",
                    "message": raw_result or "No results returned"
                }
            
            # Handle CallToolResult structured data extraction
            if "CallToolResult" in raw_result:
                # Extract the actual data portion
                if "data='" in raw_result:
                    start_pos = raw_result.find("data='") + 6
                    end_pos = raw_result.find("', is_error=False")
                    if start_pos > 5 and end_pos > start_pos:
                        raw_result = raw_result[start_pos:end_pos]
            
            lines = raw_result.strip().split('\\n')
            
            # Find header and data lines
            header_line = None
            data_lines = []
            
            for i, line in enumerate(lines):
                if '|' in line and not line.startswith('=') and not line.startswith('-'):
                    if header_line is None:
                        header_line = line
                    else:
                        data_lines.append(line)
            
            if not header_line:
                return {
                    "status": "no_data",
                    "message": "No structured data found in results"
                }
            
            # Parse header
            headers = [col.strip() for col in header_line.split('|') if col.strip()]
            
            # Parse data rows
            rows = []
            for line in data_lines:
                if line.strip() and not line.startswith('-') and '|' in line:
                    row_data = [cell.strip() for cell in line.split('|') if cell.strip()]
                    if len(row_data) >= len(headers):
                        # Take only as many cells as we have headers
                        row_data = row_data[:len(headers)]
                        row_dict = dict(zip(headers, row_data))
                        rows.append(row_dict)
            
            return {
                "status": "success",
                "headers": headers,
                "rows": rows,
                "total_rows": len(rows)
            }
            
        except Exception as e:
            return {
                "status": "parsing_error",
                "message": f"Failed to parse results: {str(e)}",
                "raw_data": raw_result
            }
    
    def _count_rows_in_result(self, result_text: str) -> int:
        """
        Count the number of data rows in the query result
        """
        try:
            if "Query Results" in result_text and "rows)" in result_text:
                # Extract row count from: "Query Results (10 rows):"
                import re
                match = re.search(r'\((\d+)\s+rows?\)', result_text)
                if match:
                    return int(match.group(1))
            
            # Fallback: count lines that look like data rows
            lines = result_text.split('\n')
            data_lines = [line for line in lines if '|' in line and not line.startswith('=') and not line.startswith('-')]
            return max(0, len(data_lines) - 1)  # Subtract header row
            
        except Exception:
            return 0
    
    def _extract_columns_from_result(self, result_text: str) -> list:
        """
        Extract column names from query result
        """
        try:
            lines = result_text.split('\n')
            for line in lines:
                if '|' in line and not line.startswith('=') and not line.startswith('-'):
                    # This should be the header line
                    columns = [col.strip() for col in line.split('|')]
                    return [col for col in columns if col]  # Remove empty columns
            return []
        except Exception:
            return []
    
    def _determine_query_type(self, sql_query: str) -> str:
        """
        Determine the type of SQL query for metadata
        """
        sql_upper = sql_query.upper()
        
        if 'GROUP BY' in sql_upper:
            return "AGGREGATION"
        elif 'JOIN' in sql_upper:
            return "JOIN_QUERY"
        elif 'ORDER BY' in sql_upper:
            return "SORTED_SELECT"
        elif 'WHERE' in sql_upper:
            return "FILTERED_SELECT"
        else:
            return "SIMPLE_SELECT"
