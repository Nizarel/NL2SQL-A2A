# SQL Generator Agent Templates

This directory contains Jinja2 templates used by the SQL Generator Agent for cleaner, more maintainable prompt engineering.

## 📁 Template Files

### `intent_analysis.jinja2`
**Purpose**: Analyzes user intent from natural language questions  
**Variables**:
- `question` (string): The user's natural language question
- `context` (string, optional): Additional context provided by user

**Output**: JSON-formatted intent analysis including objectives, entities, metrics, filters, grouping, and sorting preferences.

### `sql_generation.jinja2`
**Purpose**: Generates SQL Server-compatible queries from user questions  
**Variables**:
- `question` (string): The user's natural language question
- `schema_context` (string): Database schema information
- `intent_analysis` (dict): Results from intent analysis template

**Output**: Clean SQL Server query without formatting or explanations.

## 🔧 Usage in Code

```python
from semantic_kernel.prompt_template import PromptTemplateConfig
from semantic_kernel.functions import KernelFunctionFromPrompt

# Load template content
with open('templates/sql_generation.jinja2', 'r') as f:
    template_content = f.read()

# Create template config
config = PromptTemplateConfig(
    template=template_content,
    template_format="jinja2",
    execution_settings={
        "default": PromptExecutionSettings(
            max_tokens=800,
            temperature=0.1
        )
    }
)

# Create kernel function
function = KernelFunctionFromPrompt(
    function_name="generate_sql",
    prompt_template_config=config
)
```

## 🎯 Benefits

1. **Separation of Concerns**: Prompts are separate from business logic
2. **Maintainability**: Easy to update prompts without changing code
3. **Reusability**: Templates can be reused across different agents
4. **Version Control**: Template changes are tracked separately
5. **Collaboration**: Non-developers can contribute to prompt engineering
6. **Testing**: Templates can be tested independently

## 📝 Template Guidelines

1. **Variables**: Use clear, descriptive variable names
2. **Conditionals**: Use `{% if %}` for optional content
3. **Comments**: Use `{# #}` for template comments
4. **Formatting**: Keep templates readable with proper indentation
5. **Validation**: Test templates with various input scenarios

## 🔄 Template Evolution

Templates should be versioned and tested when modified:

1. Update template file
2. Test with representative inputs
3. Validate output quality
4. Deploy with proper change tracking

This approach ensures robust, maintainable AI prompt engineering for the NL2SQL system.
