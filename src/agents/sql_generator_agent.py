"""
SQL Generator Agent - Analyzes user intent and generates SQL queries
"""

import re
import os
from typing import Dict, Any
from semantic_kernel import Kernel
from semantic_kernel.prompt_template import PromptTemplateConfig
from semantic_kernel.connectors.ai.prompt_execution_settings import PromptExecutionSettings
from semantic_kernel.functions import KernelFunctionFromPrompt

from agents.base_agent import BaseAgent
from services.schema_service import SchemaService


class SQLGeneratorAgent(BaseAgent):
    """
    Agent responsible for analyzing user intent and generating SQL queries
    """
    
    def __init__(self, kernel: Kernel, schema_service: SchemaService):
        super().__init__(kernel, "SQLGeneratorAgent")
        self.schema_service = schema_service
        self._setup_templates()
        
    def _setup_templates(self):
        """
        Setup Jinja2 templates for intent analysis and SQL generation
        """
        # Get the templates directory
        templates_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'templates')
        
        try:
            # Load intent analysis template
            intent_template_path = os.path.join(templates_dir, 'intent_analysis.jinja2')
            with open(intent_template_path, 'r', encoding='utf-8') as f:
                intent_template_content = f.read()
            
            # Load SQL generation template
            sql_template_path = os.path.join(templates_dir, 'sql_generation.jinja2')
            with open(sql_template_path, 'r', encoding='utf-8') as f:
                sql_template_content = f.read()
                
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Template file not found: {e}. Please ensure template files exist in {templates_dir}")
        except Exception as e:
            raise Exception(f"Error loading template files: {e}")
        
        # Create prompt template configs
        intent_config = PromptTemplateConfig(
            template=intent_template_content,
            name="intent_analysis",
            template_format="jinja2",
            execution_settings={
                "default": PromptExecutionSettings(
                    max_tokens=500,
                    temperature=0.1
                )
            }
        )
        
        sql_config = PromptTemplateConfig(
            template=sql_template_content,
            name="sql_generation", 
            template_format="jinja2",
            execution_settings={
                "default": PromptExecutionSettings(
                    max_tokens=800,
                    temperature=0.1
                )
            }
        )
        
        # Create kernel functions from templates
        self.intent_analysis_function = KernelFunctionFromPrompt(
            function_name="analyze_intent",
            prompt_template_config=intent_config
        )
        
        self.sql_generation_function = KernelFunctionFromPrompt(
            function_name="generate_sql",
            prompt_template_config=sql_config
        )
        
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
        Analyze user intent from the question using Jinja2 template
        """
        try:
            # Create kernel arguments for the template
            from semantic_kernel.functions import KernelArguments
            
            arguments = KernelArguments(
                question=question,
                context=context if context else None
            )
            
            # Invoke the intent analysis function
            result = await self.kernel.invoke(
                self.intent_analysis_function,
                arguments
            )
            
            # Parse the result
            response = str(result).strip()
            return {"analysis": response}
            
        except Exception as e:
            return {"analysis": f"Intent analysis failed: {str(e)}"}
    
    async def _generate_sql(self, question: str, schema_context: str, intent_analysis: Dict[str, Any]) -> str:
        """
        Generate SQL query based on question and schema context using Jinja2 template
        """
        try:
            # Create kernel arguments for the template
            from semantic_kernel.functions import KernelArguments
            
            arguments = KernelArguments(
                question=question,
                schema_context=schema_context,
                intent_analysis=intent_analysis
            )
            
            # Invoke the SQL generation function
            result = await self.kernel.invoke(
                self.sql_generation_function,
                arguments
            )
            
            # Return the generated SQL
            return str(result).strip()
            
        except Exception as e:
            raise Exception(f"SQL generation failed: {str(e)}")
    
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