"""
Template Service - Advanced unified template management for SQL generation
Consolidates template loading and provides intelligent dynamic template selection
"""

import os
from typing import Dict, Any, Optional
from datetime import datetime
from semantic_kernel.prompt_template.prompt_template_config import PromptTemplateConfig
from semantic_kernel.functions.kernel_function_from_prompt import KernelFunctionFromPrompt
from semantic_kernel.connectors.ai.prompt_execution_settings import PromptExecutionSettings


class TemplateService:
    """
    Centralized service for template management including:
    - Unified template loading
    - Dynamic complexity-based template selection
    - Template caching and optimization
    - Shared component management
    """
    
    def __init__(self):
        self.templates_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'templates')
        self._template_cache = {}
        self._template_functions = {}
        self._shared_components = {}
        
    def initialize_templates(self) -> None:
        """
        Initialize all templates and shared components
        """
        self._load_shared_components()
        self._load_unified_template()
        self._create_template_functions()
    
    def get_template_function(self, complexity_level: str = "basic") -> Any:
        """
        Get template function based on complexity level
        
        Args:
            complexity_level: Complexity level (basic, intermediate, enhanced, advanced)
            
        Returns:
            Configured template function
        """
        # Normalize complexity level
        normalized_level = self._normalize_complexity_level(complexity_level)
        
        # Return cached function or create new one
        if normalized_level in self._template_functions:
            return self._template_functions[normalized_level]
        
        # Fallback to basic template
        return self._template_functions.get("basic")
    
    def get_intent_analysis_function(self) -> Any:
        """
        Get intent analysis template function
        
        Returns:
            Intent analysis template function
        """
        return self._template_functions.get("intent_analysis")
    
    def render_template_with_context(
        self,
        complexity_level: str,
        context: Dict[str, Any]
    ) -> str:
        """
        Render template with provided context
        
        Args:
            complexity_level: Template complexity level
            context: Template context variables
            
        Returns:
            Rendered template content
        """
        template_content = self._get_unified_template_content()
        
        # Add complexity level to context
        context["complexity_level"] = complexity_level
        context["shared_components"] = self._shared_components
        
        # Simple template rendering (in production, use Jinja2)
        rendered_content = template_content
        
        # Replace context variables
        for key, value in context.items():
            placeholder = f"{{{{{ key }}}}}"
            if isinstance(value, str):
                rendered_content = rendered_content.replace(placeholder, value)
        
        return rendered_content
    
    def _load_shared_components(self) -> None:
        """
        Load all shared template components
        """
        shared_dir = os.path.join(self.templates_dir, 'shared')
        
        if not os.path.exists(shared_dir):
            return
        
        shared_files = {
            'sql_server_rules': 'sql_server_rules.jinja2',
            'table_relationships': 'table_relationships.jinja2',
            'business_context': 'business_context.jinja2',
            'time_rules': 'time_rules.jinja2'
        }
        
        for component_name, filename in shared_files.items():
            file_path = os.path.join(shared_dir, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    self._shared_components[component_name] = f.read()
            except FileNotFoundError:
                print(f"⚠️ Shared component {filename} not found")
                self._shared_components[component_name] = f"# {component_name} component not available"
    
    def _load_unified_template(self) -> None:
        """
        Load or create unified template
        """
        unified_template_path = os.path.join(self.templates_dir, 'unified_sql_generation.jinja2')
        
        # If unified template exists, load it
        if os.path.exists(unified_template_path):
            with open(unified_template_path, 'r', encoding='utf-8') as f:
                self._template_cache['unified'] = f.read()
        else:
            # Create unified template from existing templates
            self._template_cache['unified'] = self._create_unified_template()
    
    def _create_unified_template(self) -> str:
        """
        Create unified template by consolidating existing templates
        """
        # First try to use the standalone intermediate template (no includes)
        standalone_template_path = os.path.join(self.templates_dir, 'intermediate_sql_generation_standalone.jinja2')
        if os.path.exists(standalone_template_path):
            try:
                with open(standalone_template_path, 'r', encoding='utf-8') as f:
                    return f.read()
            except Exception as e:
                print(f"⚠️ Could not load standalone template: {e}")
        
        # Fallback to basic template as foundation
        basic_template_path = os.path.join(self.templates_dir, 'sql_generation.jinja2')
        
        try:
            with open(basic_template_path, 'r', encoding='utf-8') as f:
                basic_content = f.read()
            
            # Create unified template with complexity conditions
            unified_template = f"""
{{# UNIFIED SQL GENERATION TEMPLATE #}}
{{# Dynamic template based on complexity level #}}

{{%- set complexity_level = complexity_level | default('basic') -%}}

{basic_content}

{{%- if complexity_level in ['enhanced', 'advanced'] -%}}
{{# Enhanced optimization for complex queries #}}
ADVANCED OPTIMIZATION TECHNIQUES:
- Use WITH (Common Table Expressions) for complex queries
- Apply appropriate indexing hints when necessary
- Implement query result pagination for large datasets
- Use CASE statements for conditional logic optimization
{{%- endif -%}}

{{%- if complexity_level == 'advanced' -%}}
{{# Maximum performance optimization #}}
MAXIMUM PERFORMANCE OPTIMIZATION:
- Implement query result caching strategies
- Use indexed views concepts where applicable
- Apply column store optimization patterns
- Implement parallel execution hints where beneficial
{{%- endif -%}}
"""
            return unified_template
            
        except FileNotFoundError:
            # Fallback minimal template
            return """
USER QUESTION: {{ question }}

DATABASE SCHEMA:
{{ schema_context }}

Generate a SQL Server query that answers the user's question.
Use proper SQL Server syntax and return only executable SQL code.
"""
    
    def _create_template_functions(self) -> None:
        """
        Create template functions for different complexity levels
        """
        # Intent analysis function
        intent_template = self._get_intent_template()
        intent_config = PromptTemplateConfig(
            template=intent_template,
            name="intent_analysis",
            template_format="jinja2",
            execution_settings={
                "default": PromptExecutionSettings(
                    max_tokens=500,
                    temperature=0.1
                )
            }
        )
        
        self._template_functions["intent_analysis"] = KernelFunctionFromPrompt(
            function_name="analyze_intent",
            prompt_template_config=intent_config
        )
        
        # SQL generation functions for each complexity level
        complexity_levels = ["basic", "intermediate", "enhanced", "advanced"]
        
        for level in complexity_levels:
            template_content = self._get_unified_template_content()
            
            # Adjust token limits based on complexity
            max_tokens = {
                "basic": 800,
                "intermediate": 1000,
                "enhanced": 1200,
                "advanced": 1500
            }.get(level, 800)
            
            sql_config = PromptTemplateConfig(
                template=template_content,
                name=f"sql_generation_{level}",
                template_format="jinja2",
                execution_settings={
                    "default": PromptExecutionSettings(
                        max_tokens=max_tokens,
                        temperature=0.1
                    )
                }
            )
            
            self._template_functions[level] = KernelFunctionFromPrompt(
                function_name=f"generate_sql_{level}",
                prompt_template_config=sql_config
            )
    
    def _get_unified_template_content(self) -> str:
        """
        Get unified template content
        """
        return self._template_cache.get('unified', self._create_unified_template())
    
    def _get_intent_template(self) -> str:
        """
        Get intent analysis template
        """
        intent_template_path = os.path.join(self.templates_dir, 'intent_analysis.jinja2')
        
        try:
            with open(intent_template_path, 'r', encoding='utf-8') as f:
                return f.read()
        except FileNotFoundError:
            # Fallback intent template
            return """
Analyze the user's question and provide structured intent analysis.

USER QUESTION: {{ question }}
CONTEXT: {{ context }}

Provide analysis in JSON format with:
- objectives: What the user wants to achieve
- entities: Tables/columns mentioned
- metrics: Calculations needed
- filters: Conditions to apply
- grouping: How to group results
- sorting: How to order results
"""
    
    def _normalize_complexity_level(self, complexity_level: str) -> str:
        """
        Enhanced complexity level normalization with more granular options
        """
        level_map = {
            # Basic level variations
            "basic": "basic",
            "simple": "basic",
            "easy": "basic",
            "beginner": "basic",
            
            # Intermediate level variations
            "intermediate": "intermediate", 
            "medium": "intermediate",
            "standard": "intermediate",
            "regular": "intermediate",
            
            # Enhanced level variations
            "enhanced": "enhanced",
            "complex": "enhanced",
            "sophisticated": "enhanced",
            "optimized": "enhanced",
            
            # Advanced level variations
            "advanced": "advanced",
            "maximum": "advanced",
            "expert": "advanced",
            "enterprise": "advanced",
            "production": "advanced",
            
            # New ultra level for maximum optimization
            "ultra": "ultra",
            "extreme": "ultra",
            "maximum_performance": "ultra"
        }
        
        normalized = level_map.get(complexity_level.lower(), "basic")
        
        # Log complexity level selection for analytics
        self._log_complexity_selection(complexity_level, normalized)
        
        return normalized
    
    def _log_complexity_selection(self, original: str, normalized: str) -> None:
        """
        Log template complexity selection for analytics
        """
        if not hasattr(self, '_complexity_stats'):
            self._complexity_stats = {}
        
        if normalized not in self._complexity_stats:
            self._complexity_stats[normalized] = 0
        self._complexity_stats[normalized] += 1
    
    def get_complexity_analytics(self) -> Dict[str, Any]:
        """
        Get analytics on template complexity usage
        """
        if not hasattr(self, '_complexity_stats'):
            return {"message": "No complexity analytics available"}
        
        total_uses = sum(self._complexity_stats.values())
        analytics = {
            "total_template_uses": total_uses,
            "complexity_distribution": {},
            "most_used_complexity": None,
            "least_used_complexity": None
        }
        
        if total_uses > 0:
            # Calculate percentages
            for level, count in self._complexity_stats.items():
                analytics["complexity_distribution"][level] = {
                    "count": count,
                    "percentage": round((count / total_uses) * 100, 2)
                }
            
            # Find most and least used
            analytics["most_used_complexity"] = max(self._complexity_stats, key=self._complexity_stats.get)
            analytics["least_used_complexity"] = min(self._complexity_stats, key=self._complexity_stats.get)
        
        return analytics
    
    def recommend_complexity_level(self, question: str, context: Optional[Dict[str, Any]] = None) -> str:
        """
        Intelligently recommend template complexity based on question analysis
        
        Args:
            question: User's natural language question
            context: Additional context for complexity determination
            
        Returns:
            Recommended complexity level
        """
        question_lower = question.lower()
        
        # Indicators for different complexity levels
        basic_indicators = [
            "show", "list", "get", "find", "what is", "how many", "count", "total"
        ]
        
        intermediate_indicators = [
            "join", "group by", "order by", "where", "filter", "average", "sum", 
            "max", "min", "between", "in", "like"
        ]
        
        enhanced_indicators = [
            "complex", "multiple", "relationship", "correlation", "trend", "analysis",
            "compare", "ranking", "top", "bottom", "performance", "insight"
        ]
        
        advanced_indicators = [
            "optimization", "efficiency", "advanced", "sophisticated", "enterprise",
            "production", "high-performance", "scalable", "with cte", "recursive"
        ]
        
        ultra_indicators = [
            "ultra", "extreme", "maximum", "enterprise-grade", "mission-critical",
            "high-volume", "real-time", "distributed", "parallel"
        ]
        
        # Count indicators
        basic_score = sum(1 for indicator in basic_indicators if indicator in question_lower)
        intermediate_score = sum(1 for indicator in intermediate_indicators if indicator in question_lower)
        enhanced_score = sum(1 for indicator in enhanced_indicators if indicator in question_lower)
        advanced_score = sum(1 for indicator in advanced_indicators if indicator in question_lower)
        ultra_score = sum(1 for indicator in ultra_indicators if indicator in question_lower)
        
        # Add context-based scoring
        if context:
            if context.get("table_count", 0) > 3:
                enhanced_score += 1
            if context.get("join_count", 0) > 2:
                advanced_score += 1
            if context.get("aggregation_count", 0) > 1:
                intermediate_score += 1
        
        # Determine complexity based on highest score
        scores = {
            "ultra": ultra_score,
            "advanced": advanced_score,
            "enhanced": enhanced_score,
            "intermediate": intermediate_score,
            "basic": basic_score
        }
        
        recommended = max(scores, key=scores.get)
        
        # Fallback to basic if all scores are 0
        if all(score == 0 for score in scores.values()):
            recommended = "basic"
        
        return recommended
    
    def create_custom_template(
        self,
        template_name: str,
        template_content: str,
        complexity_level: str = "custom",
        description: Optional[str] = None
    ) -> bool:
        """
        Create and register a custom template
        
        Args:
            template_name: Unique name for the template
            template_content: Jinja2 template content
            complexity_level: Complexity level assignment
            description: Optional template description
            
        Returns:
            True if template created successfully
        """
        try:
            # Validate template syntax (basic validation)
            if "{{" not in template_content or "}}" not in template_content:
                raise ValueError("Template content appears to be invalid (no template variables found)")
            
            # Store custom template
            if not hasattr(self, '_custom_templates'):
                self._custom_templates = {}
            
            self._custom_templates[template_name] = {
                "content": template_content,
                "complexity_level": complexity_level,
                "description": description,
                "created_at": datetime.now().isoformat(),
                "usage_count": 0
            }
            
            return True
            
        except Exception as e:
            print(f"⚠️ Failed to create custom template '{template_name}': {e}")
            return False
    
    def get_custom_templates(self) -> Dict[str, Any]:
        """
        Get all registered custom templates
        """
        if not hasattr(self, '_custom_templates'):
            return {}
        
        return self._custom_templates.copy()
    
    def optimize_template_cache(self) -> Dict[str, Any]:
        """
        Optimize template cache by removing unused templates and organizing
        """
        optimization_stats = {
            "templates_before": len(self._template_cache),
            "templates_after": 0,
            "memory_saved": 0,
            "optimizations_applied": []
        }
        
        # Remove unused templates (if usage tracking is available)
        if hasattr(self, '_template_usage'):
            unused_templates = [
                name for name, usage in self._template_usage.items() 
                if usage == 0
            ]
            for template_name in unused_templates:
                if template_name in self._template_cache:
                    del self._template_cache[template_name]
                    optimization_stats["optimizations_applied"].append(f"Removed unused template: {template_name}")
        
        optimization_stats["templates_after"] = len(self._template_cache)
        optimization_stats["memory_saved"] = optimization_stats["templates_before"] - optimization_stats["templates_after"]
        
        return optimization_stats
    
    def get_template_stats(self) -> Dict[str, Any]:
        """
        Get template loading and usage statistics
        """
        return {
            "templates_loaded": len(self._template_cache),
            "template_functions": len(self._template_functions),
            "shared_components": len(self._shared_components),
            "available_complexity_levels": list(self._template_functions.keys()),
            "templates_dir": self.templates_dir
        }
