"""
SQL Generator Agent - Analyzes user intent and generates SQL queries
"""

import re
from typing import Dict, Any
from semantic_kernel import Kernel

from agents.base_agent import BaseAgent
from services.schema_service import SchemaService


class SQLGeneratorAgent(BaseAgent):
    """
    Agent responsible for analyzing user intent and generating SQL queries
    """
    
    def __init__(self, kernel: Kernel, schema_service: SchemaService):
        super().__init__(kernel, "SQLGeneratorAgent")
        self.schema_service = schema_service
        
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process user question and generate SQL query
        
        Args:
            input_data: Dictionary containing:
                - question: User's natural language question
                - context: Optional additional context
                
        Returns:
            Dictionary containing generated SQL query and metadata
        """
        try:
            question = input_data.get("question", "")
            context = input_data.get("context", "")
            
            if not question:
                return self._create_result(
                    success=False,
                    error="No question provided"
                )
            
            # Get schema context
            schema_context = self.schema_service.get_full_schema_summary()
            
            # Analyze user intent
            intent_analysis = await self._analyze_intent(question, context)
            
            # Generate SQL query
            sql_query = await self._generate_sql(question, schema_context, intent_analysis)
            
            # Clean and validate SQL
            cleaned_sql = self._clean_sql_query(sql_query)
            
            return self._create_result(
                success=True,
                data={
                    "sql_query": cleaned_sql,
                    "intent_analysis": intent_analysis,
                    "question": question
                },
                metadata={
                    "schema_tables_used": self._extract_tables_from_sql(cleaned_sql),
                    "query_type": self._determine_query_type(cleaned_sql)
                }
            )
            
        except Exception as e:
            return self._create_result(
                success=False,
                error=f"SQL generation failed: {str(e)}"
            )
    
    async def _analyze_intent(self, question: str, context: str = "") -> Dict[str, Any]:
        """
        Analyze user intent from the question
        """
        intent_prompt = f"""
Analyze the user's intent from this business question:

Question: {question}
{f"Additional Context: {context}" if context else ""}

Identify:
1. Primary objective (what they want to know)
2. Key entities (customers, products, sales, time periods, etc.)
3. Metrics needed (revenue, count, average, etc.)
4. Filters or conditions
5. Grouping requirements
6. Sorting preferences

Return your analysis in this JSON format:
{{
    "objective": "description of what user wants",
    "entities": ["entity1", "entity2"],
    "metrics": ["metric1", "metric2"],
    "filters": ["filter1", "filter2"],
    "grouping": ["group_by_field1"],
    "sorting": {{"field": "sort_field", "direction": "ASC/DESC"}},
    "time_period": "specific time period if mentioned"
}}
"""
        
        try:
            response = await self._get_ai_response(intent_prompt, max_tokens=500, temperature=0.1)
            # Parse JSON response (simplified - in production, add proper JSON parsing)
            return {"analysis": response}
        except Exception as e:
            return {"analysis": f"Intent analysis failed: {str(e)}"}
    
    async def _generate_sql(self, question: str, schema_context: str, intent_analysis: Dict[str, Any]) -> str:
        """
        Generate SQL query based on question and schema context
        """
        sql_prompt = f"""
You are an expert SQL query generator for a Microsoft SQL Server business analytics database.

USER QUESTION: {question}

INTENT ANALYSIS:
{intent_analysis.get('analysis', 'No intent analysis available')}

DATABASE SCHEMA:
{schema_context}

CRITICAL SQL GENERATION RULES FOR SQL SERVER:
1. ALWAYS use 'dev.' prefix for table names (e.g., dev.cliente, dev.segmentacion)
2. **MANDATORY SQL SERVER SYNTAX ONLY**:
   - Use "SELECT TOP 10" instead of "SELECT ... LIMIT 10"
   - Use "SELECT TOP 10" instead of "OFFSET ... FETCH NEXT" syntax
   - Use YEAR(date_column) instead of EXTRACT(YEAR FROM date_column)
   - Use MONTH(date_column) instead of EXTRACT(MONTH FROM date_column)
   - Use DAY(date_column) instead of EXTRACT(DAY FROM date_column)
   - Use GETDATE() instead of NOW() or CURRENT_TIMESTAMP
   - Use CAST(GETDATE() AS DATE) instead of CURRENT_DATE
   - Use + for string concatenation instead of CONCAT()
   - Use proper spacing around AS keyword for column aliases
   
   EXAMPLES:
   ❌ WRONG: SELECT ... WHERE EXTRACT(YEAR FROM calday) = 2025
   ✅ CORRECT: SELECT ... WHERE YEAR(calday) = 2025
   
   ❌ WRONG: SELECT ... ORDER BY column LIMIT 10;
   ✅ CORRECT: SELECT TOP 10 ... ORDER BY column;
   
   ❌ WRONG: SELECT ... ORDER BY column OFFSET 0 ROWS FETCH NEXT 10 ROWS ONLY;
   ✅ CORRECT: SELECT TOP 10 ... ORDER BY column;
   
   ❌ WRONG: SUM(revenue)AS total_revenue
   ✅ CORRECT: SUM(revenue) AS total_revenue
   
3. Use correct column names as shown in schema:
   - customer_id (NOT cliente_id)
   - Nombre_cliente for customer names
   - IngresoNetoSImpuestos for revenue
4. Join tables properly using the relationships shown
5. Use meaningful column aliases with proper AS keyword spacing
6. Include appropriate WHERE clauses for business context

TABLE RELATIONSHIPS:
- dev.segmentacion (FACT) connects to:
  - dev.cliente via customer_id
  - dev.producto via material_id -> Material
  - dev.tiempo via calday -> Fecha
- dev.cliente_cedi bridges dev.cliente and dev.mercado
- dev.mercado connects via CEDIid -> cedi_id

BUSINESS CONTEXT:
- Revenue metric: IngresoNetoSImpuestos (primary revenue field)
- Customer identification: customer_id + Nombre_cliente
- Product identification: Material + Producto
- Time analysis: calday (transaction date)

Generate a SQL Server query that:
1. Accurately answers the user's question
2. Uses proper SQL Server syntax (NO PostgreSQL/MySQL functions)
3. Uses SELECT TOP instead of LIMIT or OFFSET...FETCH NEXT
4. Uses proper column names from the schema
5. Includes appropriate JOINs based on relationships
6. Has proper spacing around AS keyword in column aliases
7. Is optimized for performance
8. Returns meaningful business results

Return ONLY the SQL query without explanations or markdown formatting.
"""
        
        response = await self._get_ai_response(sql_prompt, max_tokens=800, temperature=0.1)
        return response
    
    def _clean_sql_query(self, sql_query: str) -> str:
        """
        Clean and validate the generated SQL query
        """
        # Remove markdown formatting
        sql_query = re.sub(r'^```sql\s*', '', sql_query, flags=re.MULTILINE)
        sql_query = re.sub(r'^```\s*', '', sql_query, flags=re.MULTILINE)
        sql_query = sql_query.strip()
        
        # Convert PostgreSQL/MySQL date functions to SQL Server format
        # EXTRACT(YEAR FROM date_col) -> YEAR(date_col)
        sql_query = re.sub(r'\bEXTRACT\s*\(\s*YEAR\s+FROM\s+([^)]+)\)', r'YEAR(\1)', sql_query, flags=re.IGNORECASE)
        sql_query = re.sub(r'\bEXTRACT\s*\(\s*MONTH\s+FROM\s+([^)]+)\)', r'MONTH(\1)', sql_query, flags=re.IGNORECASE)
        sql_query = re.sub(r'\bEXTRACT\s*\(\s*DAY\s+FROM\s+([^)]+)\)', r'DAY(\1)', sql_query, flags=re.IGNORECASE)
        
        # Convert DATE_TRUNC to DATEPART for SQL Server
        sql_query = re.sub(r'\bDATE_TRUNC\s*\(\s*\'year\'\s*,\s*([^)]+)\)', r'YEAR(\1)', sql_query, flags=re.IGNORECASE)
        sql_query = re.sub(r'\bDATE_TRUNC\s*\(\s*\'month\'\s*,\s*([^)]+)\)', r'MONTH(\1)', sql_query, flags=re.IGNORECASE)
        
        # Convert NOW() to GETDATE()
        sql_query = re.sub(r'\bNOW\s*\(\s*\)', 'GETDATE()', sql_query, flags=re.IGNORECASE)
        
        # Convert CURRENT_DATE to CAST(GETDATE() AS DATE)
        sql_query = re.sub(r'\bCURRENT_DATE\b', 'CAST(GETDATE() AS DATE)', sql_query, flags=re.IGNORECASE)
        
        # Convert CONCAT function (PostgreSQL) to + operator (SQL Server)
        sql_query = re.sub(r'\bCONCAT\s*\(([^)]+)\)', lambda m: m.group(1).replace(',', ' +'), sql_query, flags=re.IGNORECASE)
        
        # Convert OFFSET ... FETCH NEXT to TOP for better compatibility
        offset_fetch_pattern = r'\bOFFSET\s+(\d+)\s+ROWS?\s+FETCH\s+NEXT\s+(\d+)\s+ROWS?\s+ONLY\b'
        offset_match = re.search(offset_fetch_pattern, sql_query, re.IGNORECASE)
        
        if offset_match:
            offset_value = int(offset_match.group(1))
            fetch_value = int(offset_match.group(2))
            
            # Remove the OFFSET ... FETCH NEXT clause
            sql_query = re.sub(offset_fetch_pattern, '', sql_query, flags=re.IGNORECASE).strip()
            
            # If offset is 0, just use TOP
            if offset_value == 0:
                # Add TOP after SELECT
                sql_query = re.sub(r'\bSELECT\b', f'SELECT TOP {fetch_value}', sql_query, count=1, flags=re.IGNORECASE)
            else:
                # For non-zero offset, we need to use ROW_NUMBER() - more complex but compatible
                # This is a simplified approach - in practice might need more sophisticated handling
                print(f"⚠️ WARNING: OFFSET {offset_value} converted to TOP {fetch_value} (offset ignored for compatibility)")
                sql_query = re.sub(r'\bSELECT\b', f'SELECT TOP {fetch_value}', sql_query, count=1, flags=re.IGNORECASE)
        
        # Convert LIMIT to TOP for SQL Server compatibility
        # Handle various LIMIT patterns that might be generated
        
        # Pattern 1: "LIMIT n" at the end of query
        limit_pattern_end = r'\bLIMIT\s+(\d+)\s*;?\s*$'
        limit_match = re.search(limit_pattern_end, sql_query, re.IGNORECASE)
        
        # Pattern 2: "LIMIT n" anywhere in the query (before ORDER BY, etc.)
        limit_pattern_mid = r'\bLIMIT\s+(\d+)\b'
        
        if limit_match:
            limit_value = limit_match.group(1)
            # Remove the LIMIT clause completely
            sql_query = re.sub(limit_pattern_end, '', sql_query, flags=re.IGNORECASE).strip()
            # Add TOP after SELECT (handle multiple SELECT statements by targeting the first one)
            sql_query = re.sub(r'\bSELECT\b', f'SELECT TOP {limit_value}', sql_query, count=1, flags=re.IGNORECASE)
        elif re.search(limit_pattern_mid, sql_query, re.IGNORECASE):
            # Handle LIMIT in middle of query
            match = re.search(limit_pattern_mid, sql_query, re.IGNORECASE)
            if match:
                limit_value = match.group(1)
                # Remove the LIMIT clause
                sql_query = re.sub(limit_pattern_mid, '', sql_query, flags=re.IGNORECASE).strip()
                # Add TOP after SELECT
                sql_query = re.sub(r'\bSELECT\b', f'SELECT TOP {limit_value}', sql_query, count=1, flags=re.IGNORECASE)
        
        # Fix potential issues with column aliases and AS keyword
        # Ensure proper spacing around AS keyword
        sql_query = re.sub(r'\s+AS\s+', ' AS ', sql_query, flags=re.IGNORECASE)
        
        # Ensure dev. prefix is used for known tables
        table_names = ["cliente", "cliente_cedi", "mercado", "producto", "segmentacion", "tiempo"]
        for table in table_names:
            # Replace FROM table with FROM dev.table (case insensitive)
            sql_query = re.sub(
                rf'\bFROM\s+{table}\b',
                f'FROM dev.{table}',
                sql_query,
                flags=re.IGNORECASE
            )
            # Replace JOIN table with JOIN dev.table (case insensitive)
            sql_query = re.sub(
                rf'\bJOIN\s+{table}\b',
                f'JOIN dev.{table}',
                sql_query,
                flags=re.IGNORECASE
            )
        
        # Ensure proper statement termination
        if not sql_query.endswith(';'):
            sql_query += ';'
        
        # Final validation: Check if LIMIT still exists and warn
        if re.search(r'\bLIMIT\b', sql_query, re.IGNORECASE):
            print(f"⚠️ WARNING: LIMIT syntax still detected in SQL: {sql_query}")
            # Force conversion for any remaining LIMIT
            sql_query = re.sub(r'\bLIMIT\s+(\d+)\b', '', sql_query, flags=re.IGNORECASE)
            if not re.search(r'\bTOP\s+\d+\b', sql_query, re.IGNORECASE):
                sql_query = re.sub(r'\bSELECT\b', 'SELECT TOP 10', sql_query, count=1, flags=re.IGNORECASE)
        
        return sql_query
    
    def _extract_tables_from_sql(self, sql_query: str) -> list:
        """
        Extract table names used in SQL query
        """
        # Simple regex to find table names after FROM and JOIN
        tables = []
        patterns = [
            r'FROM\s+(?:dev\.)?(\w+)',
            r'JOIN\s+(?:dev\.)?(\w+)'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, sql_query, re.IGNORECASE)
            tables.extend(matches)
        
        return list(set(tables))  # Remove duplicates
    
    def _determine_query_type(self, sql_query: str) -> str:
        """
        Determine the type of SQL query
        """
        sql_upper = sql_query.upper()
        
        if 'SELECT' in sql_upper:
            if 'GROUP BY' in sql_upper:
                return "AGGREGATION"
            elif 'JOIN' in sql_upper:
                return "JOIN_QUERY"
            else:
                return "SIMPLE_SELECT"
        elif 'INSERT' in sql_upper:
            return "INSERT"
        elif 'UPDATE' in sql_upper:
            return "UPDATE"
        elif 'DELETE' in sql_upper:
            return "DELETE"
        else:
            return "UNKNOWN"
