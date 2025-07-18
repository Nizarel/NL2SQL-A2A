"""
Summarizing Agent - Analyzes raw data and generates business insights
"""

from typing import Dict, Any, List
import os
from semantic_kernel import Kernel
from semantic_kernel.functions import KernelFunctionFromPrompt
from semantic_kernel.prompt_template import PromptTemplateConfig
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
from semantic_kernel.contents import ChatHistory
from semantic_kernel.functions import KernelArguments

from agents.base_agent import BaseAgent


class SummarizingAgent(BaseAgent):
    """
    Agent responsible for analyzing query results and generating business summaries
    """
    
    def __init__(self, kernel: Kernel):
        super().__init__(kernel, "SummarizingAgent")
        self.templates = {}
        self._setup_templates()
        
    def _setup_templates(self):
        """Setup Jinja2 templates for different analysis types"""
        template_dir = os.path.join(os.path.dirname(__file__), '..', 'templates')
        
        # Template configurations
        template_configs = {
            'comprehensive_summary': {
                'file': 'comprehensive_summary.jinja2',
                'name': 'comprehensive_summary_function',
                'description': 'Generate comprehensive business summary from query results'
            },
            'insights_extraction': {
                'file': 'insights_extraction.jinja2', 
                'name': 'insights_extraction_function',
                'description': 'Extract key business insights from data analysis'
            },
            'recommendations': {
                'file': 'recommendations.jinja2',
                'name': 'recommendations_function', 
                'description': 'Generate actionable business recommendations'
            }
        }
        
        # Load and create kernel functions for each template
        for template_key, config in template_configs.items():
            template_path = os.path.join(template_dir, config['file'])
            
            if os.path.exists(template_path):
                with open(template_path, 'r') as f:
                    template_content = f.read()
                
                # Create prompt template config with proper execution settings
                prompt_config = PromptTemplateConfig(
                    template=template_content,
                    name=config['name'],
                    description=config['description'],
                    template_format="jinja2",
                    execution_settings={
                        "default": {
                            "max_tokens": 800,
                            "temperature": 0.2
                        }
                    }
                )
                
                # Create kernel function from template
                self.templates[template_key] = KernelFunctionFromPrompt(
                    function_name=config['name'],
                    prompt_template_config=prompt_config
                )
            else:
                raise FileNotFoundError(f"Template file not found: {template_path}")
        
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze raw data and generate summary insights
        
        Args:
            input_data: Dictionary containing:
                - raw_results: Raw query results from executor
                - formatted_results: Formatted/structured results
                - sql_query: Original SQL query
                - question: Original user question
                - metadata: Query execution metadata
                
        Returns:
            Dictionary containing summary and insights
        """
        raw_results = input_data.get("raw_results", "")
        formatted_results = input_data.get("formatted_results", {})
        sql_query = input_data.get("sql_query", "")
        question = input_data.get("question", "")
        metadata = input_data.get("metadata", {})
        
        if not raw_results and not formatted_results:
            return self._create_result(
                success=False,
                error="No data provided for summarization"
            )
        
        # Generate different types of summaries
        summary_result = await self._generate_comprehensive_summary(
            question, sql_query, formatted_results, metadata
        )
        
        # Extract key insights
        insights = await self._extract_key_insights(formatted_results, question)
        
        # Generate business recommendations
        recommendations = await self._generate_recommendations(
            formatted_results, question, insights
        )
        
        return self._create_result(
            success=True,
            data={
                "executive_summary": summary_result["executive_summary"],
                "key_insights": insights,
                "recommendations": recommendations,
                "data_overview": summary_result["data_overview"],
                "technical_summary": summary_result["technical_summary"]
            },
            metadata={
                "data_quality_score": self._assess_data_quality(formatted_results),
                "insight_confidence": summary_result.get("confidence", 0.8),
                "summary_type": self._determine_summary_type(question)
            }
        )
    
    async def _generate_comprehensive_summary(self, question: str, sql_query: str, 
                                           formatted_results: Dict[str, Any], 
                                           metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate comprehensive summary of the results using template
        """
        # Prepare template arguments
        args = KernelArguments(
            question=question,
            sql_query=sql_query,
            formatted_results_analysis=self._format_results_for_analysis(formatted_results),
            metadata=metadata
        )
        
        # Execute template function
        response = await self.kernel.invoke(self.templates['comprehensive_summary'], args)
        response_text = str(response)
        
        # Parse the structured response
        summary_parts = self._parse_structured_response(response_text)
        
        return {
            "executive_summary": summary_parts.get("EXECUTIVE_SUMMARY", "Summary generation failed"),
            "data_overview": summary_parts.get("DATA_OVERVIEW", "Data overview not available"),
            "technical_summary": summary_parts.get("TECHNICAL_SUMMARY", "Technical details not available"),
            "confidence": 0.85
        }
    
    async def _extract_key_insights(self, formatted_results: Dict[str, Any], question: str) -> List[Dict[str, Any]]:
        """
        Extract key business insights from the data using template
        """
        # Prepare template arguments
        args = KernelArguments(
            question=question,
            formatted_results_analysis=self._format_results_for_analysis(formatted_results)
        )
        
        # Execute template function
        response = await self.kernel.invoke(self.templates['insights_extraction'], args)
        response_text = str(response)
        
        # Parse insights from response
        insights = []
        lines = response_text.split('\n')
        
        for line in lines:
            if line.startswith('INSIGHT_'):
                parts = line.split('|', 2)
                if len(parts) >= 3:
                    insight_text = parts[0].split(':', 1)[1].strip()
                    significance = parts[1].strip()
                    supporting_data = parts[2].strip()
                    
                    insights.append({
                        "finding": insight_text,
                        "business_significance": significance,
                        "supporting_data": supporting_data,
                        "confidence": 0.8
                    })
        
        return insights if insights else [{"finding": "No specific insights extracted", "business_significance": "Analysis inconclusive", "supporting_data": "Insufficient data", "confidence": 0.2}]
    
    async def _generate_recommendations(self, formatted_results: Dict[str, Any], 
                                      question: str, insights: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Generate actionable business recommendations using template
        """
        # Prepare template arguments
        args = KernelArguments(
            question=question,
            formatted_insights=self._format_insights_for_prompt(insights),
            formatted_results_analysis=self._format_results_for_analysis(formatted_results)
        )
        
        # Execute template function
        response = await self.kernel.invoke(self.templates['recommendations'], args)
        response_text = str(response)
        
        # Parse recommendations from response
        recommendations = []
        lines = response_text.split('\n')
        
        for line in lines:
            if line.startswith('RECOMMENDATION_'):
                parts = line.split('|', 2)
                if len(parts) >= 3:
                    action = parts[0].split(':', 1)[1].strip()
                    outcome = parts[1].strip()
                    priority = parts[2].replace('Priority:', '').strip()
                    
                    recommendations.append({
                        "action": action,
                        "expected_outcome": outcome,
                        "priority": priority,
                        "confidence": 0.75
                    })
        
        return recommendations if recommendations else [{"action": "Review data quality and methodology", "expected_outcome": "Better insights in future analysis", "priority": "Medium", "confidence": 0.5}]
    
    def _format_results_for_analysis(self, formatted_results: Dict[str, Any]) -> str:
        """
        Format results for AI analysis
        """
        if not formatted_results:
            return "No valid data available for analysis"
        
        # Handle different status types from ExecutorAgent
        status = formatted_results.get("status")
        if status == "error":
            return f"Query error: {formatted_results.get('message', 'Unknown error')}"
        elif status == "no_data":
            return "No structured data found in results"
        elif status == "parsing_error":
            # If parsing failed, try to extract info from raw data
            raw_data = formatted_results.get("raw_data", "")
            if raw_data and "rows)" in raw_data:
                return f"Data parsing failed, but raw results available: {raw_data[:200]}..."
            return "Data parsing failed and no raw data available"
        
        # Handle successful results
        if status != "success":
            # If no explicit status, check for data presence
            rows = formatted_results.get("rows", [])
            headers = formatted_results.get("headers", [])
            
            # If no structured data, try to parse from other fields
            if not rows and not headers:
                # Check if we have raw data we can work with
                raw_data = str(formatted_results)
                if "Query Results" in raw_data and "rows)" in raw_data:
                    # Extract key information from raw data
                    import re
                    row_match = re.search(r'\((\d+)\s+rows?\)', raw_data)
                    row_count = row_match.group(1) if row_match else "unknown"
                    
                    # Try to extract sample data
                    lines = raw_data.split('\\n')
                    data_lines = [line for line in lines if '|' in line and 'CallToolResult' not in line]
                    
                    if data_lines:
                        return f"Raw query results with {row_count} rows. Sample data structure:\n{data_lines[0] if data_lines else 'No sample available'}"
                
                return "No valid data available for analysis"
        
        rows = formatted_results.get("rows", [])
        headers = formatted_results.get("headers", [])
        
        if not rows:
            return "No data rows available"
        
        # Create a comprehensive summary of the data
        summary = f"Data contains {len(rows)} rows with columns: {', '.join(headers)}\n"
        
        # Show first few rows as sample
        summary += "\nSample data:\n"
        for i, row in enumerate(rows[:5]):  # Show first 5 rows
            row_str = " | ".join([f"{k}: {v}" for k, v in row.items()])
            summary += f"Row {i+1}: {row_str}\n"
        
        if len(rows) > 5:
            summary += f"... and {len(rows) - 5} more rows\n"
        
        return summary
    
    def _format_insights_for_prompt(self, insights: List[Dict[str, Any]]) -> str:
        """
        Format insights for AI prompt
        """
        if not insights:
            return "No insights available"
        
        formatted = ""
        for i, insight in enumerate(insights, 1):
            formatted += f"{i}. {insight.get('finding', 'N/A')} - {insight.get('business_significance', 'N/A')}\n"
        
        return formatted
    
    def _parse_structured_response(self, response: str) -> Dict[str, str]:
        """
        Parse structured AI response into dictionary
        """
        result = {}
        current_section = None
        current_content = []
        
        for line in response.split('\n'):
            line = line.strip()
            if ':' in line and line.split(':')[0].isupper():
                # Save previous section
                if current_section:
                    result[current_section] = '\n'.join(current_content).strip()
                
                # Start new section
                parts = line.split(':', 1)
                current_section = parts[0].strip()
                current_content = [parts[1].strip()] if len(parts) > 1 and parts[1].strip() else []
            elif current_section and line:
                current_content.append(line)
        
        # Save last section
        if current_section:
            result[current_section] = '\n'.join(current_content).strip()
        
        return result
    
    def _assess_data_quality(self, formatted_results: Dict[str, Any]) -> float:
        """
        Assess the quality of the data for scoring
        """
        if not formatted_results or formatted_results.get("status") != "success":
            return 0.0
        
        rows = formatted_results.get("rows", [])
        if not rows:
            return 0.1
        
        # Simple quality assessment based on completeness
        total_cells = 0
        empty_cells = 0
        
        for row in rows:
            for value in row.values():
                total_cells += 1
                if not value or str(value).strip() == "":
                    empty_cells += 1
        
        if total_cells == 0:
            return 0.0
        
        completeness = 1.0 - (empty_cells / total_cells)
        return min(1.0, max(0.0, completeness))
    
    def _determine_summary_type(self, question: str) -> str:
        """
        Determine the type of summary based on the question
        """
        question_lower = question.lower()
        
        if any(word in question_lower for word in ['top', 'best', 'highest', 'ranking']):
            return "RANKING_ANALYSIS"
        elif any(word in question_lower for word in ['total', 'sum', 'revenue', 'sales']):
            return "PERFORMANCE_ANALYSIS"
        elif any(word in question_lower for word in ['trend', 'time', 'period', 'month', 'year']):
            return "TREND_ANALYSIS"
        elif any(word in question_lower for word in ['compare', 'vs', 'versus', 'difference']):
            return "COMPARATIVE_ANALYSIS"
        else:
            return "GENERAL_ANALYSIS"
