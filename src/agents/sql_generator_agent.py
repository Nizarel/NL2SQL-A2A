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
You are an expert SQL query generator for a business analytics database.

USER QUESTION: {question}

INTENT ANALYSIS:
{intent_analysis.get('analysis', 'No intent analysis available')}

DATABASE SCHEMA:
{schema_context}

CRITICAL SQL GENERATION RULES:
1. ALWAYS use 'dev.' prefix for table names (e.g., dev.cliente, dev.segmentacion)
2. Use SQL Server syntax (TOP instead of LIMIT)
3. Use correct column names as shown in schema:
   - customer_id (NOT cliente_id)
   - Nombre_cliente for customer names
   - IngresoNetoSImpuestos for revenue
4. Join tables properly using the relationships shown
5. Use meaningful column aliases
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

Generate a SQL query that:
1. Accurately answers the user's question
2. Uses proper column names from the schema
3. Includes appropriate JOINs based on relationships
4. Is optimized for performance
5. Returns meaningful business results

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
