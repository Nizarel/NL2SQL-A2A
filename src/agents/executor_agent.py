"""
Executor Agent - Executes SQL queries and handles database operations
"""

import time
import os
import re
from typing import Dict, Any
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion

from agents.base_agent import BaseAgent
from plugins.mcp_database_plugin import MCPDatabasePlugin


class ExecutorAgent(BaseAgent):
    """
    Agent responsible for executing SQL queries against the database
    Uses GPT-4o-mini for cost optimization since it primarily handles data formatting
    """
    
    def __init__(self, kernel: Kernel, mcp_plugin: MCPDatabasePlugin):
        # Create a separate kernel for cost optimization with mini model
        self.mini_kernel = self._create_mini_kernel()
        super().__init__(self.mini_kernel, "ExecutorAgent")
        self.mcp_plugin = mcp_plugin
        
    def _create_mini_kernel(self) -> Kernel:
        """
        Create a separate kernel with GPT-4o-mini for cost-efficient operations
        """
        mini_kernel = Kernel()
        
        # Setup mini model for cost optimization
        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")
        mini_deployment = os.getenv("AZURE_OPENAI_MINI_DEPLOYMENT_NAME", "gpt-4.1-mini")
        azure_api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
        
        if azure_endpoint and azure_api_key and mini_deployment:
            ai_service = AzureChatCompletion(
                deployment_name=mini_deployment,
                endpoint=azure_endpoint,
                api_key=azure_api_key,
                api_version=azure_api_version,
                service_id="azure_openai_mini"
            )
            mini_kernel.add_service(ai_service)
            print(f"💰 ExecutorAgent using cost-optimized model: {mini_deployment}")
        
        return mini_kernel
        
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
            
            # Ensure it's a SELECT query (including CTEs that start with WITH)
            if not (sql_upper.startswith('SELECT') or sql_upper.startswith('WITH')):
                return {
                    "is_valid": False,
                    "error": "Only SELECT queries are allowed",
                    "query_type": "NON_SELECT"
                }
            
            # If it starts with WITH, ensure it contains SELECT (CTE validation)
            if sql_upper.startswith('WITH') and 'SELECT' not in sql_upper:
                return {
                    "is_valid": False,
                    "error": "WITH queries must contain SELECT statements",
                    "query_type": "INVALID_CTE"
                }
            
            # Check for SQL Server syntax issues
            syntax_issues = self._check_sql_server_syntax(sql_query)
            if syntax_issues:
                return {
                    "is_valid": False,
                    "error": f"SQL Server syntax issues detected: {', '.join(syntax_issues)}",
                    "query_type": "SYNTAX_ERROR"
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
    
    def _check_sql_server_syntax(self, sql_query: str) -> list:
        """
        Check for common SQL Server syntax issues and return list of problems
        """
        issues = []
        sql_lower = sql_query.lower()
        
        # Check for PostgreSQL/MySQL INTERVAL syntax
        if 'interval ' in sql_lower and ('month' in sql_lower or 'day' in sql_lower or 'year' in sql_lower):
            issues.append("Use DATEADD() instead of INTERVAL for date arithmetic in SQL Server")
        
        # Check for malformed CTE structure - improved logic
        # Check if query appears to be a broken CTE (starts with SELECT but has CTE-like structure)
        if (sql_query.upper().startswith('SELECT') and 
            ('AS (' in sql_query or 'AS(' in sql_query) and 
            ',' in sql_query and 
            'FROM' in sql_query.upper()):
            
            # Count the CTEs (looking for patterns like "name AS (")  
            cte_pattern = r'\w+\s+AS\s*\('
            cte_matches = re.findall(cte_pattern, sql_query, re.IGNORECASE)
            
            if len(cte_matches) >= 1:
                issues.append("Query appears to be a malformed CTE - queries with CTEs must start with 'WITH' clause")
        
        # Check for common PostgreSQL functions not available in SQL Server
        postgresql_functions = ['extract(', 'date_trunc(', 'generate_series(']
        for func in postgresql_functions:
            if func in sql_lower:
                issues.append(f"Function '{func}' is not available in SQL Server")
        
        # Check for proper CTE syntax if WITH is present
        if sql_query.upper().startswith('WITH'):
            # Basic validation: ensure there's at least one SELECT in the query
            if 'SELECT' not in sql_query.upper():
                issues.append("CTE query missing SELECT statement")
            
            # Check for proper CTE structure: should have final SELECT after all CTEs
            # Look for the pattern where the last SELECT is not inside parentheses
            lines = [line.strip() for line in sql_query.split('\n') if line.strip()]
            
            # Find the last SELECT statement that's not inside a CTE definition
            final_select_found = False
            inside_cte = False
            paren_count = 0
            
            for line in lines:
                line_upper = line.upper()
                
                # Count parentheses to track if we're inside a CTE definition
                paren_count += line.count('(') - line.count(')')
                
                # If we find a SELECT and we're not inside parentheses, it's likely the final SELECT
                if 'SELECT' in line_upper and paren_count <= 0:
                    final_select_found = True
            
            if not final_select_found:
                issues.append("CTE query appears to be missing final SELECT statement")
        
        return issues
    
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
            error_message = str(e)
            
            # Provide more helpful error messages for common SQL Server issues
            if "Incorrect syntax near '12 months'" in error_message:
                error_message = "SQL Server syntax error: Use DATEADD(MONTH, -12, GETDATE()) instead of INTERVAL '12 months'"
            elif "Incorrect syntax near 'interval'" in error_message.lower():
                error_message = "SQL Server syntax error: INTERVAL is not supported. Use DATEADD() for date arithmetic"
            elif "Invalid object name" in error_message and "dev." not in error_message:
                error_message = f"Table not found: Ensure table names use 'dev.' schema prefix. Original error: {error_message}"
            elif "Incorrect syntax" in error_message:
                error_message = f"SQL syntax error: {error_message}. Check for SQL Server compatibility issues"
            
            return {
                "success": False,
                "results": None,
                "formatted_results": None,
                "execution_time": round(execution_time, 3),
                "row_count": 0,
                "columns": [],
                "error": f"Database execution error: {error_message}"
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
        
        if sql_upper.startswith('WITH'):
            if 'GROUP BY' in sql_upper:
                return "CTE_AGGREGATION"
            elif 'JOIN' in sql_upper:
                return "CTE_JOIN_QUERY"
            else:
                return "CTE_SELECT"
        elif 'GROUP BY' in sql_upper:
            return "AGGREGATION"
        elif 'JOIN' in sql_upper:
            return "JOIN_QUERY"
        elif 'ORDER BY' in sql_upper:
            return "SORTED_SELECT"
        elif 'WHERE' in sql_upper:
            return "FILTERED_SELECT"
        else:
            return "SIMPLE_SELECT"
