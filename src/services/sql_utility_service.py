"""
SQL Utility Service - Centralized SQL operations for consistency across agents
Consolidates SQL cleaning, validation, and extraction logic
"""

import re
from typing import Dict, Any, Optional, List


class SQLUtilityService:
    """
    Centralized service for SQL query operations including:
    - SQL extraction from agent responses
    - SQL cleaning and syntax conversion
    - SQL validation and formatting
    """
    
    @staticmethod
    def extract_sql_from_response(content: str) -> Optional[str]:
        """
        Enhanced SQL extraction from agent response content
        Handles various formats including markdown, explanatory text, and direct SQL
        
        Args:
            content: Raw response content from AI agent
            
        Returns:
            Clean SQL query string or None if no valid SQL found
        """
        if not content or 'SELECT' not in content.upper():
            return None
        
        # Strategy 1: Extract from ```sql markdown blocks
        sql_pattern = r'```sql\s*(.*?)\s*```'
        matches = re.findall(sql_pattern, content, re.DOTALL | re.IGNORECASE)
        if matches:
            sql_candidate = matches[0].strip()
            if SQLUtilityService._is_valid_sql_candidate(sql_candidate):
                return sql_candidate
        
        # Strategy 2: Extract from plain ``` code blocks containing SELECT
        code_pattern = r'```\s*(.*?)\s*```'
        matches = re.findall(code_pattern, content, re.DOTALL)
        for match in matches:
            match = match.strip()
            if 'SELECT' in match.upper() and SQLUtilityService._is_valid_sql_candidate(match):
                return match
        
        # Strategy 3: Extract multiline SELECT statements from free text
        lines = content.split('\n')
        sql_lines = []
        in_sql_block = False
        
        for i, line in enumerate(lines):
            line = line.strip()
            
            # Start collecting when we see SELECT or WITH
            if not in_sql_block and (line.upper().startswith('SELECT') or 
                                   (line.upper().startswith('WITH') and i < len(lines) - 1 and 
                                    'SELECT' in lines[i+1].upper())):
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
                    if SQLUtilityService._is_valid_sql_candidate(sql_candidate):
                        return sql_candidate
                    else:
                        # Reset and continue looking
                        in_sql_block = False
                        sql_lines = []
                
                # Reset if we hit explanatory text
                elif (line.upper().startswith('THIS QUERY') or 
                      line.upper().startswith('EXPLANATION') or
                      line.upper().startswith('NOTE:') or
                      line.upper().startswith('THE ABOVE') or
                      line.upper().startswith('STEP ') or
                      line.upper().startswith('**STEP')):
                    
                    sql_candidate = '\n'.join(sql_lines[:-1])  # Exclude explanatory line
                    if SQLUtilityService._is_valid_sql_candidate(sql_candidate):
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
    
    @staticmethod
    def clean_sql_query(sql_query: str) -> str:
        """
        Clean and validate the generated SQL query using modular cleaning functions
        
        Args:
            sql_query: Raw SQL query string
            
        Returns:
            Cleaned and validated SQL query
        """
        # Remove markdown formatting first
        sql_query = SQLUtilityService._clean_markdown_formatting(sql_query)
        
        # Apply SQL Server specific syntax conversions
        sql_query = SQLUtilityService._clean_sql_syntax(sql_query)
        sql_query = SQLUtilityService._clean_date_functions(sql_query)
        sql_query = SQLUtilityService._clean_limit_clauses(sql_query)
        sql_query = SQLUtilityService._validate_table_prefixes(sql_query)
        
        # Final cleanup and validation
        sql_query = SQLUtilityService._final_cleanup(sql_query)
        
        return sql_query
    
    @staticmethod
    def validate_sql_syntax(sql_query: str) -> Dict[str, Any]:
        """
        Validate SQL query syntax and structure
        
        Args:
            sql_query: SQL query to validate
            
        Returns:
            Validation result with success status and details
        """
        if not sql_query or not sql_query.strip():
            return {
                "valid": False,
                "error": "Empty SQL query",
                "query_type": None
            }
        
        sql_upper = sql_query.upper().strip()
        
        # Check for basic SQL structure
        if not sql_upper.startswith(('SELECT', 'WITH')):
            return {
                "valid": False,
                "error": "Query must start with SELECT or WITH",
                "query_type": None
            }
        
        # Determine query type
        query_type = SQLUtilityService._determine_query_type(sql_query)
        
        # Check for required components
        if 'FROM' not in sql_upper and query_type != "expression":
            return {
                "valid": False,
                "error": "Missing FROM clause",
                "query_type": query_type
            }
        
        # Check for SQL injection patterns (basic)
        suspicious_patterns = [
            r';\s*(DROP|DELETE|UPDATE|INSERT|ALTER|CREATE)\s+',
            r'EXEC\s*\(',
            r'EXECUTE\s*\(',
            r'--\s*\w'
        ]
        
        for pattern in suspicious_patterns:
            if re.search(pattern, sql_query, re.IGNORECASE):
                return {
                    "valid": False,
                    "error": "Potentially unsafe SQL detected",
                    "query_type": query_type
                }
        
        return {
            "valid": True,
            "error": None,
            "query_type": query_type
        }
    
    @staticmethod
    def extract_tables_from_sql(sql_query: str) -> List[str]:
        """
        Extract table names used in SQL query
        
        Args:
            sql_query: SQL query string
            
        Returns:
            List of table names found in the query
        """
        tables = []
        patterns = [
            r'FROM\s+(?:dev\.)?(\w+)',
            r'JOIN\s+(?:dev\.)?(\w+)'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, sql_query, re.IGNORECASE)
            tables.extend(matches)
        
        return list(set(tables))  # Remove duplicates
    
    # Private helper methods
    
    @staticmethod
    def _is_valid_sql_candidate(sql_candidate: str) -> bool:
        """Check if extracted text is a valid SQL candidate"""
        if not sql_candidate or len(sql_candidate.strip()) < 10:
            return False
        
        sql_upper = sql_candidate.upper().strip()
        
        # Must start with SELECT or WITH
        if not sql_upper.startswith(('SELECT', 'WITH')):
            return False
        
        # Must contain FROM (unless it's a simple expression)
        if 'FROM' not in sql_upper and 'GETDATE()' not in sql_upper:
            return False
        
        # Should not contain obvious non-SQL content
        non_sql_indicators = [
            'HTTP', 'HTML', 'JSON', 'XML',
            'EXPLANATION:', 'STEP 1:', 'ANALYSIS:',
            'THE QUERY', 'THIS WILL'
        ]
        
        for indicator in non_sql_indicators:
            if indicator in sql_upper:
                return False
        
        return True
    
    @staticmethod
    def _clean_markdown_formatting(sql_query: str) -> str:
        """Remove markdown SQL code block formatting"""
        sql_query = re.sub(r'^```sql\s*', '', sql_query, flags=re.MULTILINE)
        sql_query = re.sub(r'^```\s*', '', sql_query, flags=re.MULTILINE)
        sql_query = sql_query.strip()
        
        # Convert multi-line SQL to single-line format for SQL Server compatibility
        sql_query = re.sub(r'\s+', ' ', sql_query)
        return sql_query.strip()
    
    @staticmethod
    def _clean_sql_syntax(sql_query: str) -> str:
        """Convert PostgreSQL/MySQL syntax to SQL Server syntax"""
        # Convert CONCAT function to + operator
        sql_query = re.sub(r'\bCONCAT\s*\(([^)]+)\)', 
                          lambda m: m.group(1).replace(',', ' +'), 
                          sql_query, flags=re.IGNORECASE)
        
        # Fix column aliases spacing around AS keyword
        sql_query = re.sub(r'\s+AS\s+', ' AS ', sql_query, flags=re.IGNORECASE)
        
        return sql_query
    
    @staticmethod
    def _clean_date_functions(sql_query: str) -> str:
        """Convert date functions to SQL Server format"""
        # Define INTERVAL replacement patterns
        interval_patterns = [
            (r"(\w+)\s*-\s*INTERVAL\s+'(\d+)\s+months?'", r"DATEADD(MONTH, -\2, \1)"),
            (r"(\w+)\s*-\s*INTERVAL\s+'(\d+)\s+years?'", r"DATEADD(YEAR, -\2, \1)"),
            (r"(\w+)\s*-\s*INTERVAL\s+'(\d+)\s+days?'", r"DATEADD(DAY, -\2, \1)"),
            (r"(\w+)\s*\+\s*INTERVAL\s+'(\d+)\s+months?'", r"DATEADD(MONTH, \2, \1)"),
            (r"(\w+)\s*\+\s*INTERVAL\s+'(\d+)\s+years?'", r"DATEADD(YEAR, \2, \1)"),
            (r"(\w+)\s*\+\s*INTERVAL\s+'(\d+)\s+days?'", r"DATEADD(DAY, \2, \1)")
        ]
        
        for pattern, replacement in interval_patterns:
            sql_query = re.sub(pattern, replacement, sql_query, flags=re.IGNORECASE)
        
        # Handle complex INTERVAL expressions with arithmetic
        sql_query = re.sub(
            r"([A-Za-z_()]+)\s*-\s*INTERVAL\s+'(\d+)\s+months?'",
            lambda m: f"DATEADD(MONTH, -{m.group(2)}, {m.group(1)})",
            sql_query, flags=re.IGNORECASE
        )
        
        sql_query = re.sub(
            r"([A-Za-z_()]+)\s*-\s*INTERVAL\s+'(\d+)\s+days?'",
            lambda m: f"DATEADD(DAY, -{m.group(2)}, {m.group(1)})",
            sql_query, flags=re.IGNORECASE
        )
        
        # Convert date extraction functions
        sql_query = re.sub(r'\bEXTRACT\s*\(\s*YEAR\s+FROM\s+([^)]+)\)', r'YEAR(\1)', sql_query, flags=re.IGNORECASE)
        sql_query = re.sub(r'\bEXTRACT\s*\(\s*MONTH\s+FROM\s+([^)]+)\)', r'MONTH(\1)', sql_query, flags=re.IGNORECASE)
        sql_query = re.sub(r'\bEXTRACT\s*\(\s*DAY\s+FROM\s+([^)]+)\)', r'DAY(\1)', sql_query, flags=re.IGNORECASE)
        
        # Convert DATE_TRUNC to DATEPART for SQL Server
        sql_query = re.sub(r'\bDATE_TRUNC\s*\(\s*\'year\'\s*,\s*([^)]+)\)', r'YEAR(\1)', sql_query, flags=re.IGNORECASE)
        sql_query = re.sub(r'\bDATE_TRUNC\s*\(\s*\'month\'\s*,\s*([^)]+)\)', r'MONTH(\1)', sql_query, flags=re.IGNORECASE)
        
        # Convert current date/time functions
        sql_query = re.sub(r'\bNOW\s*\(\s*\)', 'GETDATE()', sql_query, flags=re.IGNORECASE)
        sql_query = re.sub(r'\bCURRENT_DATE\b', 'CAST(GETDATE() AS DATE)', sql_query, flags=re.IGNORECASE)
        
        # Fix invalid date arithmetic patterns
        sql_query = re.sub(
            r'\(\s*CAST\s*\(\s*GETDATE\s*\(\s*\)\s*AS\s+DATE\s*\)\s*-\s*(DATEADD\([^)]+\))\s*\)',
            r'\1', sql_query, flags=re.IGNORECASE
        )
        
        return sql_query
    
    @staticmethod
    def _clean_limit_clauses(sql_query: str) -> str:
        """Convert LIMIT and OFFSET...FETCH NEXT to SQL Server TOP clause"""
        # Handle LIMIT clauses
        limit_pattern = r'\bLIMIT\s+(\d+)'
        if re.search(limit_pattern, sql_query, re.IGNORECASE):
            match = re.search(limit_pattern, sql_query, re.IGNORECASE)
            if match:
                limit_num = match.group(1)
                # Remove LIMIT clause
                sql_query = re.sub(limit_pattern, '', sql_query, flags=re.IGNORECASE)
                # Add TOP clause if not already present
                if not re.search(r'\bTOP\s+\d+', sql_query, re.IGNORECASE):
                    sql_query = re.sub(r'\bSELECT\b', f'SELECT TOP {limit_num}', sql_query, count=1, flags=re.IGNORECASE)
        
        # Handle OFFSET...FETCH NEXT clauses
        offset_pattern = r'\bOFFSET\s+\d+\s+ROWS?\s+FETCH\s+NEXT\s+(\d+)\s+ROWS?\s+ONLY'
        if re.search(offset_pattern, sql_query, re.IGNORECASE):
            match = re.search(offset_pattern, sql_query, re.IGNORECASE)
            if match:
                limit_num = match.group(1)
                # Remove OFFSET...FETCH clause
                sql_query = re.sub(offset_pattern, '', sql_query, flags=re.IGNORECASE)
                # Add TOP clause if not already present
                if not re.search(r'\bTOP\s+\d+', sql_query, re.IGNORECASE):
                    sql_query = re.sub(r'\bSELECT\b', f'SELECT TOP {limit_num}', sql_query, count=1, flags=re.IGNORECASE)
        
        # Default TOP 10 for queries without explicit limits
        if (not re.search(r'\bTOP\s+\d+', sql_query, re.IGNORECASE) and 
            not re.search(r'\bCOUNT\s*\(', sql_query, re.IGNORECASE) and
            sql_query.upper().strip().startswith('SELECT')):
            sql_query = re.sub(r'\bSELECT\b', 'SELECT TOP 10', sql_query, count=1, flags=re.IGNORECASE)
        
        return sql_query
    
    @staticmethod
    def _validate_table_prefixes(sql_query: str) -> str:
        """Ensure dev. prefix is used for known tables"""
        table_names = ["cliente", "cliente_cedi", "mercado", "producto", "segmentacion", "tiempo"]
        
        for table in table_names:
            # Replace FROM table with FROM dev.table
            sql_query = re.sub(rf'\bFROM\s+{table}\b', f'FROM dev.{table}', sql_query, flags=re.IGNORECASE)
            # Replace JOIN table with JOIN dev.table
            sql_query = re.sub(rf'\bJOIN\s+{table}\b', f'JOIN dev.{table}', sql_query, flags=re.IGNORECASE)
        
        return sql_query
    
    @staticmethod
    def _final_cleanup(sql_query: str) -> str:
        """Final cleanup and validation"""
        # Ensure proper statement termination
        if not sql_query.endswith(';'):
            sql_query += ';'
        
        return sql_query
    
    @staticmethod
    def _determine_query_type(sql_query: str) -> str:
        """Determine the type of SQL query"""
        sql_upper = sql_query.upper().strip()
        
        if sql_upper.startswith('SELECT'):
            if 'COUNT(' in sql_upper:
                return "count"
            elif any(agg in sql_upper for agg in ['SUM(', 'AVG(', 'MAX(', 'MIN(']):
                return "aggregation"
            elif 'JOIN' in sql_upper:
                return "join"
            elif 'WHERE' in sql_upper:
                return "filtered_select"
            else:
                return "simple_select"
        elif sql_upper.startswith('WITH'):
            return "cte"
        else:
            return "unknown"
