import re
import os
import time
import hashlib
from typing import Dict, Any, List
from semantic_kernel import Kernel

from agents.base_agent import BaseAgent
from services.sql_utility_service import SQLUtilityService
from services.error_handling_service import ErrorHandlingService
from services.template_service import TemplateService
from services.monitoring_service import monitoring_service
from services.configuration_service import config_service


class SQLGeneratorAgent(BaseAgent):
    """
    Enhanced SQL Generator Agent with complexity analysis and adaptive templating
    
    Features:
    - Query complexity analysis (0.0 - 1.0 scale)
    - Adaptive template selection based on complexity
    - Performance optimization hints
    - Advanced SQL generation patterns
    """
    
    # Template selection thresholds for better performance and consistency
    TEMPLATE_THRESHOLDS = {
        # Removed multi_step template to ensure consistency - use advanced for all high complexity
        0.75: "advanced",       # High complexity - handles all complex queries including "for each" 
        0.3: "intermediate",    # Medium complexity
        0.2: "enhanced",        # Enhanced features
        0.0: "basic"            # Basic queries
    }
    
    def __init__(self, kernel: Kernel):
        super().__init__(kernel, "SQLGeneratorAgent")
        
        # Enhanced configuration management with fallback
        try:
            self.sql_generator_config = config_service.get_config("sql_generator") or {}
        except ValueError:
            # Fallback to default configuration if section doesn't exist
            self.sql_generator_config = {
                "default_limit": 100,
                "max_complexity_score": 1.0,
                "template_selection_mode": "adaptive",
                "performance_tracking": True,
                "optimization_level": "high"
            }
            print("⚠️ Using default SQL generator configuration (sql_generator section not found)")
        
        # Initialize services
        self.template_service = TemplateService()
        self.template_service.initialize_templates()
        
        # Initialize performance monitoring
        self.monitoring_service = monitoring_service
        self._initialize_performance_tracking()
        
    def _initialize_performance_tracking(self):
        """Initialize performance monitoring for SQL generation"""
        try:
            # Register SQL generator specific metrics
            self.monitoring_service.record_metric("sql_generator_initialized", 1)
            print("📊 SQL Generator monitoring initialized")
        except Exception as e:
            print(f"⚠️ Failed to initialize SQL Generator monitoring: {str(e)}")
        
    def _setup_templates(self):
        """
        Templates are now managed by TemplateService
        This method is kept for compatibility but delegates to the service
        """
        # Template setup is handled by TemplateService in __init__
        pass
        
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enhanced process method with complexity analysis, adaptive templating, and service integration
        
        Args:
            input_data: Dictionary containing:
                - question: User's natural language question
                - context: Optional additional context
                - optimized_schema_context: Optimized schema from Schema Analyst
                - schema_analysis: Full schema analysis results
                
        Returns:
            Dictionary containing generated SQL query, complexity analysis, and metadata
        """
        processing_start_time = time.time()
        correlation_id = input_data.get("correlation_id", f"sql_gen_{int(time.time())}")
        
        # Record processing start
        self.monitoring_service.record_metric("sql_generator_requests", 1)
        
        try:
            # Enhanced input validation
            question = input_data.get("question", "")
            context = input_data.get("context", "")
            optimized_schema_context = input_data.get("optimized_schema_context")
            schema_analysis = input_data.get("schema_analysis")
            
            # Enhanced validation with ConfigurationService
            max_question_length = self.sql_generator_config.get("max_question_length", 1000)
            if not question:
                return ErrorHandlingService.create_enhanced_error_response(
                    error=ValueError("No question provided for SQL generation"),
                    context={"operation": "sql_generation_validation", "correlation_id": correlation_id}
                )
            
            if len(question) > max_question_length:
                return ErrorHandlingService.create_enhanced_error_response(
                    error=ValueError(f"Question too long: {len(question)} > {max_question_length}"),
                    context={"operation": "sql_generation_validation", "correlation_id": correlation_id}
                )
            
            # Schema context validation with enhanced error handling
            if optimized_schema_context:
                schema_context = optimized_schema_context
                schema_source = "optimized"
                print("🎯 Using optimized schema context from Schema Analyst")
            else:
                return ErrorHandlingService.create_enhanced_error_response(
                    error=ValueError("No schema context provided. Schema Analyst should provide optimized context."),
                    context={
                        "operation": "schema_context_validation", 
                        "correlation_id": correlation_id,
                        "question": question[:100]
                    }
                )
            
            # STEP 1: Analyze query complexity with performance tracking
            complexity_start_time = time.time()
            complexity_analysis = self._analyze_query_complexity(question)
            complexity_score = complexity_analysis["complexity_score"]
            complexity_time = time.time() - complexity_start_time
            
            self.monitoring_service.record_metric("sql_generator_complexity_analysis_time", complexity_time * 1000)
            print(f"🔍 Query complexity score: {complexity_score:.2f} ({complexity_analysis['complexity_level']}) - {complexity_time:.3f}s")
            
            # STEP 2: Select appropriate template based on complexity
            template_start_time = time.time()
            template_choice = self._select_template_by_complexity(complexity_score, question)
            template_selection_time = time.time() - template_start_time
            
            self.monitoring_service.record_metric("sql_generator_template_selection_time", template_selection_time * 1000)
            print(f"📋 Selected template: {template_choice} - {template_selection_time:.3f}s")
            
            # STEP 3: Generate performance hints and optimization context
            optimization_start_time = time.time()
            optimization_context = self._generate_optimization_context(
                complexity_analysis, schema_analysis, question
            )
            optimization_time = time.time() - optimization_start_time
            
            self.monitoring_service.record_metric("sql_generator_optimization_context_time", optimization_time * 1000)
            
            # STEP 4: Analyze user intent (enhanced with complexity context)
            intent_start_time = time.time()
            intent_analysis = await self._analyze_intent(question, context, schema_analysis)
            intent_time = time.time() - intent_start_time
            
            self.monitoring_service.record_metric("sql_generator_intent_analysis_time", intent_time * 1000)
            
            # STEP 5: Generate SQL query with adaptive template and optimization context
            sql_generation_start_time = time.time()
            sql_query = await self._generate_sql_with_complexity(
                question, schema_context, intent_analysis, 
                schema_analysis, complexity_analysis, optimization_context, template_choice
            )
            sql_generation_time = time.time() - sql_generation_start_time
            
            self.monitoring_service.record_metric("sql_generator_sql_generation_time", sql_generation_time * 1000)
            
            # STEP 6: Clean and validate SQL with performance tracking
            cleaning_start_time = time.time()
            cleaned_sql = SQLUtilityService.clean_sql_query(sql_query)
            cleaning_time = time.time() - cleaning_start_time
            
            self.monitoring_service.record_metric("sql_generator_sql_cleaning_time", cleaning_time * 1000)
            
            # STEP 7: Prepare enhanced metadata with performance data
            total_processing_time = time.time() - processing_start_time
            
            metadata = {
                "schema_tables_used": SQLUtilityService.extract_tables_from_sql(cleaned_sql),
                "query_type": SQLUtilityService.validate_sql_syntax(cleaned_sql).get("query_type"),
                "schema_source": schema_source,
                "schema_context_size": len(schema_context) if schema_context else 0,
                "template_used": template_choice,
                "complexity_score": complexity_score,
                "complexity_level": complexity_analysis['complexity_level'],
                "optimization_hints_applied": len(optimization_context.get('performance_hints', [])),
                "correlation_id": correlation_id,
                "performance_metrics": {
                    "total_processing_time_ms": round(total_processing_time * 1000, 2),
                    "complexity_analysis_time_ms": round(complexity_time * 1000, 2),
                    "template_selection_time_ms": round(template_selection_time * 1000, 2),
                    "optimization_context_time_ms": round(optimization_time * 1000, 2),
                    "intent_analysis_time_ms": round(intent_time * 1000, 2),
                    "sql_generation_time_ms": round(sql_generation_time * 1000, 2),
                    "sql_cleaning_time_ms": round(cleaning_time * 1000, 2)
                }
            }
            
            # Add schema analysis metadata if available
            if schema_analysis:
                metadata.update({
                    "relevant_tables": schema_analysis.get("relevant_tables", []),
                    "confidence_score": schema_analysis.get("confidence_score", 0),
                    "join_strategy": schema_analysis.get("join_strategy", {}).get("strategy", "unknown"),
                    "performance_hints_count": len(schema_analysis.get("performance_hints", []))
                })
            
            # Record success metrics
            self.monitoring_service.record_metric("sql_generator_success_rate", 100.0)
            self.monitoring_service.record_metric("sql_generator_total_processing_time", total_processing_time * 1000)
            
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
            # Enhanced error handling with performance context
            processing_time = time.time() - processing_start_time
            self.monitoring_service.record_metric("sql_generator_success_rate", 0.0)
            self.monitoring_service.record_metric("sql_generator_error_count", 1)
            
            return ErrorHandlingService.handle_agent_processing_error(
                error=e,
                agent_name="SQLGeneratorAgent",
                input_data=input_data,
                step="enhanced_sql_generation",
                context={
                    "correlation_id": correlation_id,
                    "processing_time_ms": round(processing_time * 1000, 2),
                    "question": question[:100] if 'question' in locals() else "unknown"
                }
            )
    
    def _analyze_query_complexity(self, question: str) -> Dict[str, Any]:
        """
        Analyze query complexity based on various indicators using optimized pattern groups
        Enhanced with performance tracking and configurable thresholds
        
        Returns complexity score (0.0 - 1.0) and detailed analysis
        """
        analysis_start_time = time.time()
        
        try:
            # Get complexity analysis configuration
            enable_detailed_analysis = self.sql_generator_config.get("enable_detailed_complexity_analysis", True)
            complexity_threshold_high = self.sql_generator_config.get("complexity_threshold_high", 0.75)
            complexity_threshold_medium = self.sql_generator_config.get("complexity_threshold_medium", 0.3)
            
            # Grouped complexity indicators for better performance
            complexity_pattern_groups = {
                'multi_step_indicators': {
                    'patterns': [
                        r'\b(each|every|per|for\s+each|by\s+each)\b.*\b(cedi|region|territory|location)\b',
                        r'\b(highest|best|top|maximum|most|greatest)\b.*\b(profit|revenue|sales)\b.*\b(each|every|per)\b',
                        r'\b(which|what).*\b(product|category)\b.*\b(highest|best|top)\b.*\b(each|every|per)\b',
                        r'\b(compare|comparison|versus|vs|against)\b.*\b(across|between|within)\b',
                        r'\b(both|all|multiple)\b.*\b(product|category|dimension|level)\b'
                    ],
                    'weight': 0.6
                },
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
            
            # Calculate base complexity score with performance tracking
            score = 0.0
            matched_patterns = []
            question_lower = str(question).lower() if question else ""
            
            pattern_matching_start = time.time()
            
            # Process pattern groups efficiently
            for group_name, group_data in complexity_pattern_groups.items():
                group_match = False
                weight = float(group_data.get('weight', 0.0))
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
            
            pattern_matching_time = time.time() - pattern_matching_start
            self.monitoring_service.record_metric("sql_generator_pattern_matching_time", pattern_matching_time * 1000)
            
            # Additional complexity factors with error handling
            word_count = len(question.split()) if question else 0
            if word_count > 15:
                score += 0.1
            if word_count > 25:
                score += 0.15
            if word_count > 35:
                score += 0.2
            
            # Question structure complexity
            question_str = str(question) if question else ""
            question_complexity = question_str.count('?') + question_str.count(',') * 0.1
            score += min(question_complexity, 0.3)
            
            # Enhanced combination bonuses with configuration
            if enable_detailed_analysis:
                if any('aggregation' in pattern for pattern in matched_patterns) and \
                   any('temporal' in pattern for pattern in matched_patterns):
                    score += 0.2  # Time-based aggregation bonus
                
                if any('analytical' in pattern for pattern in matched_patterns) and \
                   any('business' in pattern for pattern in matched_patterns):
                    score += 0.15  # Business analytics bonus
                    
                # Multi-step analysis bonus with enhanced detection
                if any('multi_step' in pattern for pattern in matched_patterns):
                    multi_dimensional_phrases = [
                        "compare.*and.*across", "both.*and.*by", "multiple.*perspective", 
                        "various.*dimension", "different.*level.*analysis"
                    ]
                    truly_multi_dimensional = any(re.search(phrase, question_lower) for phrase in multi_dimensional_phrases)
                    
                    if truly_multi_dimensional:
                        score += 0.3
                        print("🔍 True multi-dimensional analysis patterns detected - boosting complexity score")
                    else:
                        score += 0.15
                        print("🎯 Complex single-dimensional analysis detected - moderate complexity boost")
                
                # Enhanced multi-step detection
                multi_step_phrases = [
                    "for each", "per cedi", "by cedi", "each cedi", 
                    "which product", "what product", "which category", "what category",
                    "highest profit", "best performing", "top performer"
                ]
                
                multi_step_matches = sum(1 for phrase in multi_step_phrases if phrase in question_lower)
                if multi_step_matches >= 3:
                    score += 0.2
                    print(f"🎯 Multiple complexity indicators detected: {multi_step_matches}")
                elif multi_step_matches >= 2:
                    score += 0.1
                    print(f"📊 Moderate complexity indicators detected: {multi_step_matches}")
            
            # Ensure score is within bounds
            score = min(max(float(score), 0.0), 1.0)
            
            # Determine complexity level with configurable thresholds
            if score >= complexity_threshold_high:
                if score >= 0.8:
                    complexity_level = "VERY_HIGH"
                    estimated_tables = "4+"
                    estimated_joins = "3+"
                else:
                    complexity_level = "HIGH"
                    estimated_tables = "3+"
                    estimated_joins = "2+"
            elif score >= complexity_threshold_medium:
                complexity_level = "MEDIUM"
                estimated_tables = "2-3"
                estimated_joins = "1-2"
            else:
                complexity_level = "LOW"
                estimated_tables = "1-2"
                estimated_joins = "0-1"
            
            # Enhanced analysis with performance metrics
            analysis_time = time.time() - analysis_start_time
            self.monitoring_service.record_metric("sql_generator_complexity_analysis_total_time", analysis_time * 1000)
            
            analysis = {
                "complexity_score": score,
                "complexity_level": complexity_level,
                "matched_patterns": matched_patterns[:5],  # Top 5 patterns
                "word_count": word_count,
                "estimated_tables_needed": estimated_tables,
                "estimated_joins_needed": estimated_joins,
                "requires_aggregation": any('aggregation' in pattern for pattern in matched_patterns),
                "requires_time_analysis": any('temporal' in pattern for pattern in matched_patterns),
                "requires_ranking": any('analytical' in pattern for pattern in matched_patterns),
                "requires_multi_step": any('multi_step' in pattern for pattern in matched_patterns),
                "analysis_performance": {
                    "total_analysis_time_ms": round(analysis_time * 1000, 2),
                    "pattern_matching_time_ms": round(pattern_matching_time * 1000, 2),
                    "patterns_matched": len(matched_patterns)
                }
            }
            
            return analysis
            
        except Exception as e:
            # Fallback analysis on error
            analysis_time = time.time() - analysis_start_time
            print(f"⚠️ Complexity analysis error: {e}, using basic analysis")
            
            return {
                "complexity_score": 0.5,  # Medium complexity as fallback
                "complexity_level": "MEDIUM",
                "matched_patterns": [],
                "word_count": len(question.split()) if question else 0,
                "estimated_tables_needed": "2-3",
                "estimated_joins_needed": "1-2",
                "requires_aggregation": False,
                "requires_time_analysis": False,
                "requires_ranking": False,
                "requires_multi_step": False,
                "analysis_performance": {
                    "total_analysis_time_ms": round(analysis_time * 1000, 2),
                    "error": str(e)
                }
            }
    
    def _select_template_by_complexity(self, complexity_score: float, question: str = "") -> str:
        """
        Select appropriate template based on complexity score and question pattern analysis
        """
        try:
            # Ensure complexity_score is a float
            score = float(complexity_score) if complexity_score is not None else 0.0
            question_lower = question.lower() if question else ""
            
            # Special logic for single-dimensional "for each" questions
            is_single_dimensional_each = (
                ("for each" in question_lower or "per " in question_lower or "by each" in question_lower) and 
                not ("compare" in question_lower and ("versus" in question_lower or "vs" in question_lower)) and
                not ("both" in question_lower and "and" in question_lower)
            )
            
            # Force single-dimensional "for each" questions to use advanced template (not multi_step)
            if is_single_dimensional_each and score >= 0.75:
                print(f"🎯 Single-dimensional 'for each' pattern detected - using advanced template (score: {score})")
                return "advanced"
            
            # Standard template selection for other patterns
            for threshold, template in self.TEMPLATE_THRESHOLDS.items():
                if score >= float(threshold):
                    return template
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
            template_function = self.template_service.get_template_function(template_choice)
            if not template_function:
                print(f"⚠️ Template {template_choice} not found, falling back to basic")
                template_function = self.template_service.get_template_function("basic")
            
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
            intent_function = self.template_service.get_intent_analysis_function()
            result = await self.kernel.invoke(
                intent_function,
                arguments
            )
            
            # Parse the result
            response = str(result).strip()
            return {"analysis": response}
            
        except Exception as e:
            return {"analysis": f"Intent analysis failed: {str(e)}"}
    
    # ===============================
    # LEGACY METHODS - MOVED TO SERVICES
    # ===============================
    # The following SQL cleaning and utility methods have been moved to:
    # - SQLUtilityService: SQL cleaning, extraction, and validation
    # - ErrorHandlingService: Standardized error handling
    # - TemplateService: Template management and selection
    # 
    # This eliminates code duplication and provides centralized services
    # that can be used across all agents consistently.
    # ===============================

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
