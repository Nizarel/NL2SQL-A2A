"""
Template Service - Unified template management for SQL generation
Consolidates template loading and provides dynamic template selection
"""

import os
from typing import Dict, Any, Optional
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
        Normalize complexity level input
        """
        level_map = {
            "basic": "basic",
            "simple": "basic",
            "intermediate": "intermediate",
            "medium": "intermediate",
            "enhanced": "enhanced",
            "complex": "enhanced",
            "advanced": "advanced",
            "maximum": "advanced"
        }
        
        return level_map.get(complexity_level.lower(), "basic")
    
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
