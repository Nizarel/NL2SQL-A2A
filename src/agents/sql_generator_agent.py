"""
Enhanced SQL Generator Agent - Analyzes user intent and generates optimized SQL queries
with complexity analysis and adaptive templating
"""

import re
import os
import hashlib
from typing import Dict, Any, List
from semantic_kernel import Kernel
from semantic_kernel.prompt_template import PromptTemplateConfig
from semantic_kernel.connectors.ai.prompt_execution_settings import PromptExecutionSettings
from semantic_kernel.functions import KernelFunctionFromPrompt

from agents.base_agent import BaseAgent


class SQLGeneratorAgent(BaseAgent):
    """
    Enhanced SQL Generator Agent with complexity analysis and adaptive templating
    
    Features:
    - Query complexity analysis (0.0 - 1.0 scale)
    - Adaptive template selection based on complexity
    - Performance optimization hints
    - Advanced SQL generation patterns
    """
    
    # Template selection thresholds for better performance
    TEMPLATE_THRESHOLDS = {
        0.7: "advanced",
        0.3: "intermediate", 
        0.2: "enhanced",
        0.0: "basic"
    }
    
    def __init__(self, kernel: Kernel):
        super().__init__(kernel, "SQLGeneratorAgent")
        self.template_functions = {}  # Store multiple template functions
        self._template_cache = {}  # Cache for loaded templates
        self._setup_templates()
        
    def _setup_templates(self):
        """
        Setup multiple Jinja2 templates for different complexity levels
        """
        # Get the templates directory
        templates_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'templates')
        
        # Template files for different complexity levels
        template_files = {
            'basic': 'sql_generation.jinja2',
            'intermediate': 'intermediate_sql_generation.jinja2', 
            'advanced': 'advanced_sql_generation.jinja2',
            'enhanced': 'enhanced_sql_generation.jinja2'
        }
        
        try:
            # Load intent analysis template
            intent_template_path = os.path.join(templates_dir, 'intent_analysis.jinja2')
            with open(intent_template_path, 'r', encoding='utf-8') as f:
                intent_template_content = f.read()
                
            # Load all SQL generation templates
            sql_templates = {}
            for level, filename in template_files.items():
                template_path = os.path.join(templates_dir, filename)
                try:
                    with open(template_path, 'r', encoding='utf-8') as f:
                        sql_templates[level] = f.read()
                except FileNotFoundError:
                    print(f"⚠️ Template {filename} not found, using basic template as fallback")
                    if 'basic' in sql_templates:
                        sql_templates[level] = sql_templates['basic']
                
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Core template file not found: {e}. Please ensure template files exist in {templates_dir}")
        except Exception as e:
            raise Exception(f"Error loading template files: {e}")
        
        # Create intent analysis function
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
        
        self.intent_analysis_function = KernelFunctionFromPrompt(
            function_name="analyze_intent",
            prompt_template_config=intent_config
        )
        
        # Create SQL generation functions for each complexity level
        for level, template_content in sql_templates.items():
            sql_config = PromptTemplateConfig(
                template=template_content,
                name=f"sql_generation_{level}",
                template_format="jinja2",
                execution_settings={
                    "default": PromptExecutionSettings(
                        max_tokens=1200 if level in ['advanced', 'enhanced'] else 800,
                        temperature=0.1
                    )
                }
            )
            
            self.template_functions[level] = KernelFunctionFromPrompt(
                function_name=f"generate_sql_{level}",
                prompt_template_config=sql_config
            )
        
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enhanced process method with complexity analysis and adaptive templating
        
        Args:
            input_data: Dictionary containing:
                - question: User's natural language question
                - context: Optional additional context
                - optimized_schema_context: Optimized schema from Schema Analyst
                - schema_analysis: Full schema analysis results
                
        Returns:
            Dictionary containing generated SQL query, complexity analysis, and metadata
        """
        try:
            question = input_data.get("question", "")
            context = input_data.get("context", "")
            optimized_schema_context = input_data.get("optimized_schema_context")
            schema_analysis = input_data.get("schema_analysis")
            
            if not question:
                return self._create_result(
                    success=False,
                    error="No question provided"
                )
            
            # Schema context must be provided by orchestrator via optimized_schema_context
            if optimized_schema_context:
                schema_context = optimized_schema_context
                schema_source = "optimized"
                print("🎯 Using optimized schema context from Schema Analyst")
            else:
                # No fallback - orchestrator should always provide schema context
                return self._create_result(
                    success=False,
                    error="No schema context provided. Schema Analyst should provide optimized context."
                )
            
            # STEP 1: Analyze query complexity
            complexity_analysis = self._analyze_query_complexity(question)
            complexity_score = complexity_analysis["complexity_score"]
            
            print(f"🔍 Query complexity score: {complexity_score:.2f} ({complexity_analysis['complexity_level']})")
            
            # STEP 2: Select appropriate template based on complexity
            template_choice = self._select_template_by_complexity(complexity_score)
            
            print(f"📋 Selected template: {template_choice}")
            
            # STEP 3: Generate performance hints and optimization context
            optimization_context = self._generate_optimization_context(
                complexity_analysis, schema_analysis, question
            )
            
            # STEP 4: Analyze user intent (enhanced with complexity context)
            intent_analysis = await self._analyze_intent(question, context, schema_analysis)
            
            # STEP 5: Generate SQL query with adaptive template and optimization context
            sql_query = await self._generate_sql_with_complexity(
                question, schema_context, intent_analysis, 
                schema_analysis, complexity_analysis, optimization_context, template_choice
            )
            
            # STEP 6: Clean and validate SQL
            cleaned_sql = self._clean_sql_query(sql_query)
            
            # STEP 7: Prepare enhanced metadata
            metadata = {
                "schema_tables_used": self._extract_tables_from_sql(cleaned_sql),
                "query_type": self._determine_query_type(cleaned_sql),
                "schema_source": schema_source,
                "schema_context_size": len(schema_context) if schema_context else 0,
                "template_used": template_choice,
                "complexity_score": complexity_score,
                "complexity_level": complexity_analysis['complexity_level'],
                "optimization_hints_applied": len(optimization_context.get('performance_hints', []))
            }
            
            # Add schema analysis metadata if available
            if schema_analysis:
                metadata.update({
                    "relevant_tables": schema_analysis.get("relevant_tables", []),
                    "confidence_score": schema_analysis.get("confidence_score", 0),
                    "join_strategy": schema_analysis.get("join_strategy", {}).get("strategy", "unknown"),
                    "performance_hints_count": len(schema_analysis.get("performance_hints", []))
                })
            
            return self._create_result(
                success=True,
                data={
                    "sql_query": cleaned_sql,
                    "intent_analysis": intent_analysis,
                    "complexity_analysis": complexity_analysis,
                    "optimization_context": optimization_context,
                    "question": question
                },
                metadata=metadata
            )
            
        except Exception as e:
            return self._create_result(
                success=False,
                error=f"Enhanced SQL generation failed: {str(e)}"
            )
    
    def _analyze_query_complexity(self, question: str) -> Dict[str, Any]:
        """
        Analyze query complexity based on various indicators using optimized pattern groups
        
        Returns complexity score (0.0 - 1.0) and detailed analysis
        """
        # Grouped complexity indicators for better performance
        complexity_pattern_groups = {
            'join_indicators': {
                'patterns': [
                    r'\b(join|joins|joining|combine|merge|connect)\b',
                    r'\b(multiple tables|several tables|across tables)\b',
                    r'\b(left join|right join|outer join|inner join)\b'
                ],
                'weight': 0.4
            },
            'aggregation_indicators': {
                'patterns': [
                    r'\b(group by|grouping|aggregate|sum|count|average|min|max|total)\b',
                    r'\b(having|group having)\b'
                ],
                'weight': 0.35
            },
            'advanced_patterns': {
                'patterns': [
                    r'\b(subquery|nested|sub-query|within|inside)\b',
                    r'\b(window function|partition|over|rank|row_number)\b',
                    r'\b(cte|common table|recursive)\b'
                ],
                'weight': 0.5
            },
            'analytical_functions': {
                'patterns': [
                    r'\b(top|bottom|rank|percentile|quartile|best|worst|highest|lowest)\b',
                    r'\b(trend|analysis|analytics|compare|comparison)\b',
                    r'\b(year over year|month over month|time series)\b'
                ],
                'weight': 0.35
            },
            'complexity_modifiers': {
                'patterns': [
                    r'\b(multiple|several|various|different|distinct)\b',
                    r'\b(complex|complicated|advanced)\b',
                    r'\b(all|everything|entire|complete|full)\b',
                    r'\b(detailed|detail|comprehensive)\b'
                ],
                'weight': 0.25
            },
            'temporal_patterns': {
                'patterns': [
                    r'\b(last|previous|past|recent|since|between|range)\b',
                    r'\b(monthly|yearly|quarterly|weekly|daily)\b',
                    r'\b(months?|years?|days?|weeks?|time period)\b'
                ],
                'weight': 0.2
            },
            'business_patterns': {
                'patterns': [
                    r'\b(revenue|sales|profit|income|earnings)\b',
                    r'\b(customer|client|product|category|region|market)\b',
                    r'\b(by\s+\w+|per\s+\w+|for\s+each)\b'
                ],
                'weight': 0.15
            }
        }
        
        # Calculate base complexity score
        score = 0.0
        matched_patterns = []
        question_lower = str(question).lower() if question else ""
        
        # Process pattern groups efficiently
        for group_name, group_data in complexity_pattern_groups.items():
            group_match = False
            weight = float(group_data.get('weight', 0.0))  # Ensure weight is float
            for pattern in group_data['patterns']:
                try:
                    if re.search(pattern, question_lower):
                        if not group_match:  # Only count once per group to avoid over-scoring
                            score += weight
                            group_match = True
                        matched_patterns.append(f"{group_name}:{pattern.split('|')[0].replace('\\b', '').replace('(', '')}")
                except (TypeError, AttributeError) as e:
                    print(f"⚠️ Pattern matching error for {pattern}: {e}")
                    continue
                    
        # Additional complexity factors
        word_count = len(question.split())
        try:
            if word_count > 15:
                score += 0.1
            if word_count > 25:
                score += 0.15
            if word_count > 35:
                score += 0.2
        except TypeError:
            # Handle case where word_count might not be an integer
            word_count = int(len(question.split())) if question else 0
            if word_count > 15:
                score += 0.1
            if word_count > 25:
                score += 0.15
            if word_count > 35:
                score += 0.2
        
        # Question structure complexity
        try:
            question_complexity = question.count('?') + question.count(',') * 0.1
            score += min(question_complexity, 0.3)
        except TypeError:
            # Handle case where question might not be a string
            question = str(question) if question else ""
            question_complexity = question.count('?') + question.count(',') * 0.1
            score += min(question_complexity, 0.3)
        
        # Combination bonuses
        if any('aggregation' in pattern for pattern in matched_patterns) and \
           any('temporal' in pattern for pattern in matched_patterns):
            score += 0.2  # Time-based aggregation bonus
        
        if any('analytical' in pattern for pattern in matched_patterns) and \
           any('business' in pattern for pattern in matched_patterns):
            score += 0.15  # Business analytics bonus
        
        # Cap the score at 1.0
        score = min(float(score), 1.0)
        
        # Determine complexity level with adjusted thresholds
        try:
            if score >= 0.7:
                complexity_level = "HIGH"
                estimated_tables = "3+"
                estimated_joins = "2+"
            elif score >= 0.3:
                complexity_level = "MEDIUM"
                estimated_tables = "2-3"
                estimated_joins = "1-2"
            else:
                complexity_level = "LOW"
                estimated_tables = "1-2"
                estimated_joins = "0-1"
        except TypeError:
            print(f"⚠️ Score comparison error, score: {score} (type: {type(score)})")
            complexity_level = "LOW"
            estimated_tables = "1-2"
            estimated_joins = "0-1"
        
        # Detailed analysis with optimized pattern checking
        analysis = {
            "complexity_score": score,
            "complexity_level": complexity_level,
            "matched_patterns": matched_patterns[:5],  # Top 5 patterns
            "word_count": word_count,
            "estimated_tables_needed": estimated_tables,
            "estimated_joins_needed": estimated_joins,
            "requires_aggregation": any('aggregation' in pattern for pattern in matched_patterns),
            "requires_time_analysis": any('temporal' in pattern for pattern in matched_patterns),
            "requires_ranking": any('analytical' in pattern for pattern in matched_patterns)
        }
        
        return analysis
    
    def _select_template_by_complexity(self, complexity_score: float) -> str:
        """
        Select appropriate template based on complexity score using optimized thresholds
        """
        try:
            # Ensure complexity_score is a float
            score = float(complexity_score) if complexity_score is not None else 0.0
            
            for threshold, template in self.TEMPLATE_THRESHOLDS.items():
                if score >= float(threshold):
                    return template if template in self.template_functions else "basic"
            return "basic"
        except (ValueError, TypeError) as e:
            print(f"⚠️ Template selection error: {e}, using basic template")
            return "basic"
    
    def _generate_optimization_context(self, complexity_analysis: Dict[str, Any], 
                                     schema_analysis: Dict[str, Any] = None,
                                     question: str = "") -> Dict[str, Any]:
        """
        Generate optimization context based on complexity with conditional processing for performance
        """
        try:
            complexity_score = float(complexity_analysis.get("complexity_score", 0.0))
            complexity_level = complexity_analysis.get("complexity_level", "LOW")
            
            # For very simple queries, return basic context to avoid overhead
            if complexity_score < 0.2:
                return self._generate_basic_context(complexity_analysis)
            
            # For medium complexity, generate moderate context
            elif complexity_score < 0.7:
                return self._generate_medium_context(complexity_analysis, schema_analysis, question)
            
            # For high complexity, generate full advanced context
            else:
                return self._generate_advanced_context(complexity_analysis, schema_analysis, question)
                
        except (ValueError, TypeError, KeyError) as e:
            print(f"⚠️ Optimization context generation error: {e}, using basic context")
            return self._generate_basic_context(complexity_analysis)
    
    def _generate_basic_context(self, complexity_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate basic optimization context for simple queries"""
        return {
            "optimization_level": "basic",
            "performance_hints": ["Use appropriate indexes", "Keep query simple"],
            "suggested_columns": [],
            "table_priorities": [],
            "join_strategy": "standard",
            "aggregation_hints": [],
            "relationship_hints": [],
            "business_context_hints": []
        }
    
    def _generate_medium_context(self, complexity_analysis: Dict[str, Any], 
                               schema_analysis: Dict[str, Any] = None,
                               question: str = "") -> Dict[str, Any]:
        """Generate medium optimization context for moderately complex queries"""
        optimization_context = {
            "optimization_level": "medium",
            "performance_hints": [],
            "suggested_columns": [],
            "table_priorities": [],
            "join_strategy": "standard",
            "aggregation_hints": [],
            "relationship_hints": [],
            "business_context_hints": []
        }
        
        # Add performance hints based on specific needs
        hints = []
        
        if complexity_analysis.get("requires_aggregation"):
            hints.append("Use appropriate indexes on GROUP BY columns")
            optimization_context["aggregation_hints"].append("Apply GROUP BY on indexed columns")
        
        if complexity_analysis.get("requires_time_analysis"):
            hints.append("Optimize date range queries with proper indexing")
        
        if complexity_analysis.get("estimated_joins_needed"):
            try:
                joins_needed = int(complexity_analysis.get("estimated_joins_needed", 0))
                if joins_needed > 0:
                    hints.append("Use INNER JOIN when possible for better performance")
                    optimization_context["join_strategy"] = "optimized_order"
            except (ValueError, TypeError):
                # Handle non-numeric join estimates like "1-2" or "2+"
                joins_str = str(complexity_analysis.get("estimated_joins_needed", "0"))
                if any(char.isdigit() for char in joins_str):
                    hints.append("Use INNER JOIN when possible for better performance")
                    optimization_context["join_strategy"] = "optimized_order"
        
        optimization_context["performance_hints"] = hints
        
        # Add basic schema analysis context if available
        if schema_analysis:
            relevant_tables = schema_analysis.get("relevant_tables", [])[:3]  # Limit to 3 tables
            if relevant_tables:
                optimization_context["table_priorities"] = relevant_tables
        
        self._add_business_context_hints(optimization_context, question)
        
        return optimization_context
    
    def _generate_advanced_context(self, complexity_analysis: Dict[str, Any], 
                                 schema_analysis: Dict[str, Any] = None,
                                 question: str = "") -> Dict[str, Any]:
        """Generate comprehensive optimization context for complex queries"""
        optimization_context = {
            "optimization_level": "high",
            "performance_hints": [],
            "suggested_columns": [],
            "table_priorities": [],
            "join_strategy": "optimized_order",
            "aggregation_hints": [],
            "relationship_hints": [],
            "business_context_hints": []
        }
        
        # Generate comprehensive performance hints
        hints = []
        
        if complexity_analysis.get("requires_aggregation"):
            hints.extend([
                "Use appropriate indexes on GROUP BY columns",
                "Consider using aggregate functions efficiently"
            ])
            optimization_context["aggregation_hints"].extend([
                "Apply GROUP BY on indexed columns",
                "Use HAVING clause for post-aggregation filtering"
            ])
        
        if complexity_analysis.get("requires_time_analysis"):
            hints.extend([
                "Optimize date range queries with proper indexing",
                "Use YEAR(), MONTH(), DAY() functions for date filtering"
            ])
            
        if complexity_analysis.get("requires_ranking"):
            hints.extend([
                "Use ROW_NUMBER() with PARTITION BY for efficient ranking",
                "Consider TOP clause instead of complex ranking when possible"
            ])
        
        if complexity_analysis.get("estimated_joins_needed"):
            try:
                joins_needed = int(complexity_analysis.get("estimated_joins_needed", 0))
                if joins_needed > 1:
                    hints.extend([
                        "Optimize join order: most selective table first",
                        "Use INNER JOIN when possible for better performance"
                    ])
            except (ValueError, TypeError):
                # Handle non-numeric join estimates like "1-2" or "2+"
                joins_str = str(complexity_analysis.get("estimated_joins_needed", "0"))
                if any(char.isdigit() for char in joins_str) and ("2" in joins_str or "+" in joins_str):
                    hints.extend([
                        "Optimize join order: most selective table first",
                        "Use INNER JOIN when possible for better performance"
                    ])
        
        # Add advanced optimization hints
        hints.extend([
            "Consider using CTE for better readability and performance",
            "Apply table hints (NOLOCK) for analytical queries",
            "Use query compilation hints for complex queries"
        ])
        
        optimization_context["performance_hints"] = hints
        
        # Extract comprehensive optimization hints from schema analysis
        if schema_analysis:
            relevant_tables = schema_analysis.get("relevant_tables", [])
            if relevant_tables:
                optimization_context["table_priorities"] = relevant_tables[:5]  # Top 5 tables
            
            if schema_analysis.get("performance_hints"):
                optimization_context["performance_hints"].extend(
                    schema_analysis["performance_hints"][:3]  # Add top 3 schema hints
                )
            
            # Extract suggested columns from schema analysis
            key_metrics = schema_analysis.get("key_metrics", [])
            if key_metrics:
                optimization_context["suggested_columns"] = key_metrics[:10]  # Top 10 columns
        
        self._add_business_context_hints(optimization_context, question)
        
        return optimization_context
    
    def _add_business_context_hints(self, optimization_context: Dict[str, Any], question: str):
        """Add business context hints based on question patterns"""
        question_lower = question.lower()
        business_hints = []
        
        if "revenue" in question_lower or "sales" in question_lower:
            business_hints.append("Focus on IngresoNetoSImpuestos as primary revenue metric")
        
        if "customer" in question_lower or "client" in question_lower:
            business_hints.append("Use customer_id + Nombre_cliente for customer identification")
        
        if "product" in question_lower or "item" in question_lower:
            business_hints.append("Use Material + Producto for complete product information")
        
        optimization_context["business_context_hints"] = business_hints
    
    async def _generate_sql_with_complexity(self, question: str, schema_context: str, 
                                          intent_analysis: Dict[str, Any],
                                          schema_analysis: Dict[str, Any] = None,
                                          complexity_analysis: Dict[str, Any] = None,
                                          optimization_context: Dict[str, Any] = None,
                                          template_choice: str = "basic") -> str:
        """
        Generate SQL query using complexity-aware template with optimization context
        """
        try:
            # Create kernel arguments for the enhanced template
            from semantic_kernel.functions import KernelArguments
            
            # Base arguments
            arguments = KernelArguments(
                question=question,
                schema_context=schema_context,
                intent_analysis=intent_analysis
            )
            
            # Add complexity analysis context
            if complexity_analysis:
                arguments["complexity_analysis"] = complexity_analysis
                arguments["complexity_score"] = complexity_analysis["complexity_score"]
            
            # Add optimization context
            if optimization_context:
                arguments["optimization_level"] = optimization_context.get("optimization_level", "medium")
                arguments["performance_hints"] = optimization_context.get("performance_hints", [])
                arguments["suggested_columns"] = optimization_context.get("suggested_columns", [])
                arguments["table_priorities"] = optimization_context.get("table_priorities", [])
                arguments["join_strategy"] = optimization_context.get("join_strategy", "standard")
                arguments["aggregation_hints"] = optimization_context.get("aggregation_hints", [])
                arguments["relationship_hints"] = optimization_context.get("relationship_hints", [])
                arguments["business_context_hints"] = optimization_context.get("business_context_hints", [])
            
            # Add schema analysis context for enhanced SQL generation
            if schema_analysis:
                arguments["relevant_tables"] = schema_analysis.get("relevant_tables", [])
                arguments["optimized_schema_context"] = schema_analysis.get("optimized_context", "")
                arguments["key_metrics"] = schema_analysis.get("key_metrics", [])
            
            # Add additional template variables for enhanced templates
            arguments["limit"] = 100  # Default limit
            arguments["debug_mode"] = False  # Can be made configurable
            arguments["result_size_warning"] = complexity_analysis.get("complexity_score", 0) > 0.7 if complexity_analysis else False
            
            # Select the appropriate template function
            template_function = self.template_functions.get(template_choice)
            if not template_function:
                print(f"⚠️ Template {template_choice} not found, falling back to basic")
                template_function = self.template_functions.get("basic", self.template_functions.get("enhanced"))
            
            if not template_function:
                raise Exception("No SQL generation template available")
            
            # Invoke the selected template function
            result = await self.kernel.invoke(
                template_function,
                arguments
            )
            
            # Return the generated SQL
            return str(result).strip()
            
        except Exception as e:
            raise Exception(f"Enhanced SQL generation failed: {str(e)}")

    async def _analyze_intent(self, question: str, context: str = "", schema_analysis: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Analyze user intent from the question using Jinja2 template
        Enhanced with schema analysis context if available
        """
        try:
            # Create kernel arguments for the template
            from semantic_kernel.functions import KernelArguments
            
            arguments = KernelArguments(
                question=question,
                context=context if context else None
            )
            
            # Add schema analysis context if available for enhanced intent understanding
            if schema_analysis:
                arguments["relevant_tables"] = str(schema_analysis.get("relevant_tables", []))
                arguments["business_context"] = str(schema_analysis.get("business_context", {}))
                arguments["key_metrics"] = str(schema_analysis.get("key_metrics", []))
            
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
    
    def _clean_sql_query(self, sql_query: str) -> str:
        """
        Clean and validate the generated SQL query using modular cleaning functions
        """
        # Remove markdown formatting first
        sql_query = self._clean_markdown_formatting(sql_query)
        
        # Apply SQL Server specific syntax conversions
        sql_query = self._clean_sql_syntax(sql_query)
        sql_query = self._clean_date_functions(sql_query)
        sql_query = self._clean_limit_clauses(sql_query)
        sql_query = self._validate_table_prefixes(sql_query)
        
        # Final cleanup and validation
        sql_query = self._final_cleanup(sql_query)
        
        return sql_query
    
    def _clean_markdown_formatting(self, sql_query: str) -> str:
        """Remove markdown SQL code block formatting"""
        sql_query = re.sub(r'^```sql\s*', '', sql_query, flags=re.MULTILINE)
        sql_query = re.sub(r'^```\s*', '', sql_query, flags=re.MULTILINE)
        sql_query = sql_query.strip()
        
        # Convert multi-line SQL to single-line format for SQL Server compatibility
        sql_query = re.sub(r'\s+', ' ', sql_query)
        return sql_query.strip()
    
    def _clean_sql_syntax(self, sql_query: str) -> str:
        """Convert PostgreSQL/MySQL syntax to SQL Server syntax"""
        # Convert CONCAT function to + operator
        sql_query = re.sub(r'\bCONCAT\s*\(([^)]+)\)', 
                          lambda m: m.group(1).replace(',', ' +'), 
                          sql_query, flags=re.IGNORECASE)
        
        # Fix column aliases spacing around AS keyword
        sql_query = re.sub(r'\s+AS\s+', ' AS ', sql_query, flags=re.IGNORECASE)
        
        return sql_query
    
    def _clean_date_functions(self, sql_query: str) -> str:
        """Convert date functions to SQL Server format"""
        # Convert PostgreSQL/MySQL INTERVAL syntax to SQL Server DATEADD
        interval_patterns = [
            (r"INTERVAL\s+'(\d+)\s+months?'", lambda m: f"DATEADD(MONTH, -{m.group(1)}, GETDATE())"),
            (r"INTERVAL\s+'(\d+)\s+MONTH'", lambda m: f"DATEADD(MONTH, -{m.group(1)}, GETDATE())"),
            (r"INTERVAL\s+'(\d+)\s+days?'", lambda m: f"DATEADD(DAY, -{m.group(1)}, GETDATE())"),
            (r"INTERVAL\s+'(\d+)\s+DAY'", lambda m: f"DATEADD(DAY, -{m.group(1)}, GETDATE())"),
            (r"INTERVAL\s+'(\d+)\s+years?'", lambda m: f"DATEADD(YEAR, -{m.group(1)}, GETDATE())"),
            (r"INTERVAL\s+'(\d+)\s+YEAR'", lambda m: f"DATEADD(YEAR, -{m.group(1)}, GETDATE())"),
            (r"INTERVAL\s+(\d+)\s+MONTH", lambda m: f"DATEADD(MONTH, -{m.group(1)}, GETDATE())"),
            (r"INTERVAL\s+(\d+)\s+DAY", lambda m: f"DATEADD(DAY, -{m.group(1)}, GETDATE())"),
            (r"INTERVAL\s+(\d+)\s+YEAR", lambda m: f"DATEADD(YEAR, -{m.group(1)}, GETDATE())"),
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
    
    def _clean_limit_clauses(self, sql_query: str) -> str:
        """Convert LIMIT and OFFSET...FETCH NEXT to SQL Server TOP clause"""
        # Convert OFFSET ... FETCH NEXT to TOP
        offset_fetch_pattern = r'\bOFFSET\s+(\d+)\s+ROWS?\s+FETCH\s+NEXT\s+(\d+)\s+ROWS?\s+ONLY\b'
        offset_match = re.search(offset_fetch_pattern, sql_query, re.IGNORECASE)
        
        if offset_match:
            offset_value = int(offset_match.group(1))
            fetch_value = int(offset_match.group(2))
            
            # Remove the OFFSET ... FETCH NEXT clause
            sql_query = re.sub(offset_fetch_pattern, '', sql_query, flags=re.IGNORECASE).strip()
            
            if offset_value == 0:
                # Add TOP after SELECT
                sql_query = re.sub(r'\bSELECT\b', f'SELECT TOP {fetch_value}', sql_query, count=1, flags=re.IGNORECASE)
            else:
                # For non-zero offset, use TOP with warning
                print(f"⚠️ WARNING: OFFSET {offset_value} converted to TOP {fetch_value} (offset ignored for compatibility)")
                sql_query = re.sub(r'\bSELECT\b', f'SELECT TOP {fetch_value}', sql_query, count=1, flags=re.IGNORECASE)
        
        # Convert LIMIT to TOP
        limit_pattern_end = r'\bLIMIT\s+(\d+)\s*;?\s*$'
        limit_pattern_mid = r'\bLIMIT\s+(\d+)\b'
        
        limit_match = re.search(limit_pattern_end, sql_query, re.IGNORECASE)
        
        if limit_match:
            limit_value = limit_match.group(1)
            sql_query = re.sub(limit_pattern_end, '', sql_query, flags=re.IGNORECASE).strip()
            sql_query = self._add_top_clause_to_query(sql_query, limit_value)
        elif re.search(limit_pattern_mid, sql_query, re.IGNORECASE):
            match = re.search(limit_pattern_mid, sql_query, re.IGNORECASE)
            if match:
                limit_value = match.group(1)
                sql_query = re.sub(limit_pattern_mid, '', sql_query, flags=re.IGNORECASE).strip()
                sql_query = self._add_top_clause_to_query(sql_query, limit_value)
        
        # Final validation: Check if LIMIT still exists
        if re.search(r'\bLIMIT\b', sql_query, re.IGNORECASE):
            print(f"⚠️ WARNING: LIMIT syntax still detected in SQL: {sql_query}")
            sql_query = re.sub(r'\bLIMIT\s+(\d+)\b', '', sql_query, flags=re.IGNORECASE)
            if not re.search(r'\bTOP\s+\d+\b', sql_query, re.IGNORECASE):
                sql_query = re.sub(r'\bSELECT\b', 'SELECT TOP 10', sql_query, count=1, flags=re.IGNORECASE)
        
        return sql_query
    
    def _validate_table_prefixes(self, sql_query: str) -> str:
        """Ensure dev. prefix is used for known tables"""
        table_names = ["cliente", "cliente_cedi", "mercado", "producto", "segmentacion", "tiempo"]
        
        for table in table_names:
            # Replace FROM table with FROM dev.table
            sql_query = re.sub(rf'\bFROM\s+{table}\b', f'FROM dev.{table}', sql_query, flags=re.IGNORECASE)
            # Replace JOIN table with JOIN dev.table
            sql_query = re.sub(rf'\bJOIN\s+{table}\b', f'JOIN dev.{table}', sql_query, flags=re.IGNORECASE)
        
        return sql_query
    
    def _final_cleanup(self, sql_query: str) -> str:
        """Final cleanup and validation"""
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
    
    def _add_top_clause_to_query(self, sql_query: str, limit_value: str) -> str:
        """
        Add TOP clause to the appropriate SELECT statement in a query.
        Handles both simple SELECT queries and CTE queries correctly.
        """
        if sql_query.upper().startswith('WITH'):
            # For CTE queries, find the final SELECT statement (not inside CTEs)
            lines = sql_query.split('\n')
            final_select_line_idx = -1
            
            # Find the last SELECT that's at the root level (not in a CTE definition)
            # Work backwards to find the final SELECT
            for i in reversed(range(len(lines))):
                line_stripped = lines[i].strip().upper()
                if line_stripped.startswith('SELECT'):
                    # Check if this is at the root level by counting parentheses
                    # from the start of the query to this line
                    paren_count = 0
                    for j in range(i + 1):
                        paren_count += lines[j].count('(') - lines[j].count(')')
                    
                    # If parentheses are balanced (0) or negative, it's the final SELECT
                    if paren_count <= 0:
                        final_select_line_idx = i
                        break
            
            if final_select_line_idx >= 0:
                # Replace the final SELECT with SELECT TOP n
                original_line = lines[final_select_line_idx]
                modified_line = re.sub(r'\bSELECT\b', f'SELECT TOP {limit_value}', original_line, count=1, flags=re.IGNORECASE)
                lines[final_select_line_idx] = modified_line
                return '\n'.join(lines)
            else:
                print("⚠️ Warning: Could not find final SELECT in CTE query to add TOP clause")
                return sql_query
        else:
            # For simple SELECT queries, add TOP after the first SELECT
            return re.sub(r'\bSELECT\b', f'SELECT TOP {limit_value}', sql_query, count=1, flags=re.IGNORECASE)
