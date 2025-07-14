"""
Summarizing Agent - Analyzes raw data and generates business insights
"""

from typing import Dict, Any, List
from semantic_kernel import Kernel

from agents.base_agent import BaseAgent


class SummarizingAgent(BaseAgent):
    """
    Agent responsible for analyzing query results and generating business summaries
    """
    
    def __init__(self, kernel: Kernel):
        super().__init__(kernel, "SummarizingAgent")
        
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
        try:
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
            
        except Exception as e:
            return self._create_result(
                success=False,
                error=f"Summarization failed: {str(e)}"
            )
    
    async def _generate_comprehensive_summary(self, question: str, sql_query: str, 
                                           formatted_results: Dict[str, Any], 
                                           metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate comprehensive summary of the results
        """
        summary_prompt = f"""
Analyze the following business query results and provide a comprehensive summary:

ORIGINAL QUESTION: {question}

SQL QUERY EXECUTED:
{sql_query}

QUERY RESULTS:
{self._format_results_for_analysis(formatted_results)}

EXECUTION METADATA:
- Execution Time: {metadata.get('execution_time', 'N/A')} seconds
- Rows Returned: {metadata.get('row_count', 'N/A')}
- Query Type: {metadata.get('query_type', 'N/A')}

Please provide:

1. EXECUTIVE SUMMARY (2-3 sentences):
   A high-level business summary that answers the user's question directly.

2. DATA OVERVIEW:
   Key statistics and patterns found in the data.

3. TECHNICAL SUMMARY:
   Brief technical details about the query execution and data quality.

Format your response as:
EXECUTIVE_SUMMARY: [your executive summary]
DATA_OVERVIEW: [your data overview]
TECHNICAL_SUMMARY: [your technical summary]
"""
        
        try:
            response = await self._get_ai_response(summary_prompt, max_tokens=800, temperature=0.2)
            
            # Parse the structured response
            summary_parts = self._parse_structured_response(response)
            
            return {
                "executive_summary": summary_parts.get("EXECUTIVE_SUMMARY", "Summary generation failed"),
                "data_overview": summary_parts.get("DATA_OVERVIEW", "Data overview not available"),
                "technical_summary": summary_parts.get("TECHNICAL_SUMMARY", "Technical details not available"),
                "confidence": 0.85
            }
            
        except Exception as e:
            return {
                "executive_summary": f"Summary generation error: {str(e)}",
                "data_overview": "Unable to analyze data",
                "technical_summary": "Technical analysis unavailable",
                "confidence": 0.1
            }
    
    async def _extract_key_insights(self, formatted_results: Dict[str, Any], question: str) -> List[Dict[str, Any]]:
        """
        Extract key business insights from the data
        """
        insights_prompt = f"""
Analyze this business data and extract 3-5 key insights:

BUSINESS QUESTION: {question}

DATA:
{self._format_results_for_analysis(formatted_results)}

Extract key insights that would be valuable to business stakeholders. For each insight:
1. Identify the specific finding
2. Explain its business significance
3. Provide supporting data points

Format each insight as:
INSIGHT_[NUMBER]: [Finding] | [Business Significance] | [Supporting Data]

Example:
INSIGHT_1: Top customer generates 25% of total revenue | High customer concentration risk | Customer XYZ: $43M revenue
"""
        
        try:
            response = await self._get_ai_response(insights_prompt, max_tokens=600, temperature=0.3)
            
            # Parse insights from response
            insights = []
            lines = response.split('\n')
            
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
            
        except Exception as e:
            return [{"finding": f"Insight extraction failed: {str(e)}", "business_significance": "Unable to determine", "supporting_data": "Error in analysis", "confidence": 0.1}]
    
    async def _generate_recommendations(self, formatted_results: Dict[str, Any], 
                                      question: str, insights: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Generate actionable business recommendations
        """
        recommendations_prompt = f"""
Based on the following business question, data analysis, and insights, provide 2-4 actionable recommendations:

BUSINESS QUESTION: {question}

KEY INSIGHTS:
{self._format_insights_for_prompt(insights)}

DATA CONTEXT:
{self._format_results_for_analysis(formatted_results)}

Provide specific, actionable recommendations that business stakeholders can implement. Each recommendation should:
1. Be specific and actionable
2. Address a business opportunity or risk
3. Be based on the data findings

Format each recommendation as:
RECOMMENDATION_[NUMBER]: [Action] | [Expected Outcome] | [Priority: High/Medium/Low]

Example:
RECOMMENDATION_1: Focus marketing efforts on top-performing customer segments | Increase revenue by 15-20% | Priority: High
"""
        
        try:
            response = await self._get_ai_response(recommendations_prompt, max_tokens=500, temperature=0.4)
            
            # Parse recommendations from response
            recommendations = []
            lines = response.split('\n')
            
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
            
        except Exception as e:
            return [{"action": f"Recommendation generation failed: {str(e)}", "expected_outcome": "Unable to determine", "priority": "Low", "confidence": 0.1}]
    
    def _format_results_for_analysis(self, formatted_results: Dict[str, Any]) -> str:
        """
        Format results for AI analysis
        """
        if not formatted_results or formatted_results.get("status") != "success":
            return "No valid data available for analysis"
        
        rows = formatted_results.get("rows", [])
        headers = formatted_results.get("headers", [])
        
        if not rows:
            return "No data rows available"
        
        # Create a summary of the data
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
