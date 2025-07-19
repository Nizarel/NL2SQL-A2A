"""
Simple test for Enhanced SQL Generator Agent complexity analysis
Tests only the complexity analysis functionality without requiring external dependencies
"""

import os
import sys
import re
from typing import Dict, Any

# Add the src directory to the path  
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

class ComplexityAnalyzer:
    """
    Isolated complexity analyzer for testing the enhanced SQL generation logic
    """
    
    def _analyze_query_complexity(self, question: str) -> Dict[str, Any]:
        """
        Analyze query complexity based on various indicators
        
        Returns complexity score (0.0 - 1.0) and detailed analysis
        """
        complexity_indicators = {
            # Join complexity indicators
            r'\b(join|joins|joining|combine|merge|connect)\b': 0.3,
            r'\b(multiple tables|several tables|across tables)\b': 0.25,
            r'\b(left join|right join|outer join|inner join)\b': 0.35,
            
            # Aggregation complexity
            # Aggregation complexity - increased weights
        r'\b(group by|grouping|aggregate|sum|count|average|min|max|total)\b': 0.35,
            r'\b(having|group having)\b': 0.3,
            
            # Advanced query patterns
            r'\b(subquery|nested|sub-query|within|inside)\b': 0.4,
            r'\b(window function|partition|over|rank|row_number)\b': 0.5,
            r'\b(cte|common table|recursive)\b': 0.45,
            
            # Multiple conditions/complexity
            r'\b(multiple|several|various|different|distinct)\b': 0.2,
            r'\b(complex|complicated|advanced)\b': 0.3,
            
            # Analytical functions
            r'\b(top|bottom|rank|percentile|quartile)\b': 0.3,
            r'\b(trend|analysis|analytics|compare|comparison)\b': 0.25,
            r'\b(year over year|month over month|time series)\b': 0.4,
            
            # Performance-intensive operations
            r'\b(all|everything|entire|complete|full)\b': 0.15,
            r'\b(detailed|detail|comprehensive)\b': 0.2,
            
            # Date/time complexity
            r'\b(last|previous|past|recent|since|between|range)\b': 0.15,
            r'\b(monthly|yearly|quarterly|weekly|daily)\b': 0.2,
        }
        
        # Calculate base complexity score
        score = 0.0
        matched_patterns = []
        
        question_lower = question.lower()
        
        for pattern, weight in complexity_indicators.items():
            if re.search(pattern, question_lower):
                score += weight
                matched_patterns.append(pattern.replace('\\b', '').replace('(', '').replace(')', '').split('|')[0])
        
        # Additional complexity factors
        word_count = len(question.split())
        if word_count > 20:
            score += 0.1  # Longer questions tend to be more complex
        if word_count > 30:
            score += 0.1
        
        # Count question marks and sub-clauses
        question_complexity = question.count('?') + question.count(',') * 0.05
        score += min(question_complexity, 0.2)
        
        # Cap the score at 1.0
        score = min(score, 1.0)
        
        # Adjust complexity thresholds for better classification
        if score >= 0.7:
            complexity_level = "HIGH"
            estimated_tables = "3+"
            estimated_joins = "2+"
        elif score >= 0.3:  # Lowered threshold for medium complexity
            complexity_level = "MEDIUM"
            estimated_tables = "2-3"
            estimated_joins = "1-2"
        else:
            complexity_level = "LOW"
            estimated_tables = "1-2"
            estimated_joins = "0-1"
        
        # Detailed analysis
        analysis = {
            "complexity_score": score,
            "complexity_level": complexity_level,
            "matched_patterns": matched_patterns[:5],  # Top 5 patterns
            "word_count": word_count,
            "estimated_tables_needed": estimated_tables,
            "estimated_joins_needed": estimated_joins,
            "requires_aggregation": any(re.search(p, question_lower) for p in [
                r'\b(sum|count|average|total|min|max|group)\b'
            ]),
            "requires_time_analysis": any(re.search(p, question_lower) for p in [
                r'\b(year|month|day|date|time|recent|last|since)\b'
            ]),
            "requires_ranking": any(re.search(p, question_lower) for p in [
                r'\b(top|bottom|best|worst|rank|highest|lowest)\b'
            ])
        }
        
        return analysis
    
    def _select_template_by_complexity(self, complexity_score: float) -> str:
        """
        Select appropriate template based on complexity score - adjusted thresholds
        """
        if complexity_score >= 0.7:
            return "advanced"
        elif complexity_score >= 0.3:  # Lowered threshold for intermediate
            return "intermediate"
        elif complexity_score >= 0.2:  # Enhanced for slightly complex queries
            return "enhanced"
        else:
            return "basic"


def test_complexity_analysis():
    """
    Test the complexity analysis functionality
    """
    print("🧪 Testing Enhanced SQL Generator - Complexity Analysis")
    print("=" * 60)
    
    analyzer = ComplexityAnalyzer()
    
    # Comprehensive test cases
    test_cases = [
        # Simple queries (LOW complexity)
        {
            "question": "Show me all customers",
            "expected_level": "LOW",
            "expected_template": "basic"
        },
        {
            "question": "List products",
            "expected_level": "LOW", 
            "expected_template": "basic"
        },
        {
            "question": "Get customer names",
            "expected_level": "LOW",
            "expected_template": "basic"
        },
        
        # Medium complexity queries
        {
            "question": "Show me total revenue by customer for the last 6 months",
            "expected_level": "MEDIUM",
            "expected_template": "intermediate"
        },
        {
            "question": "Join customer and product tables to show sales data",
            "expected_level": "MEDIUM", 
            "expected_template": "intermediate"
        },
        {
            "question": "Get average revenue by region with monthly grouping",
            "expected_level": "MEDIUM",
            "expected_template": "intermediate"
        },
        {
            "question": "Show count of products sold by category since last year",
            "expected_level": "MEDIUM",
            "expected_template": "intermediate"
        },
        
        # High complexity queries
        {
            "question": "Show me the top 10 customers by revenue with year over year growth analysis, including monthly trends and ranking within their market segment",
            "expected_level": "HIGH",
            "expected_template": "advanced"
        },
        {
            "question": "Create a comprehensive analysis of customer behavior with ranking, percentiles, and nested subqueries for detailed comparison across multiple time periods",
            "expected_level": "HIGH",
            "expected_template": "advanced"
        },
        {
            "question": "Generate advanced analytics with window functions, partition by regions, and complex aggregations for trend analysis",
            "expected_level": "HIGH",
            "expected_template": "advanced"
        },
        {
            "question": "Show detailed revenue analytics with CTE, multiple joins across various tables, and advanced ranking functions",
            "expected_level": "HIGH", 
            "expected_template": "advanced"
        }
    ]
    
    print(f"🔍 Testing {len(test_cases)} query complexity scenarios...\n")
    
    correct_complexity = 0
    correct_template = 0
    total_tests = len(test_cases)
    
    for i, test_case in enumerate(test_cases, 1):
        question = test_case["question"]
        expected_level = test_case["expected_level"]
        expected_template = test_case["expected_template"]
        
        try:
            # Analyze complexity
            complexity_analysis = analyzer._analyze_query_complexity(question)
            actual_level = complexity_analysis["complexity_level"]
            complexity_score = complexity_analysis["complexity_score"]
            
            # Select template
            actual_template = analyzer._select_template_by_complexity(complexity_score)
            
            # Check results
            complexity_correct = actual_level == expected_level
            template_correct = actual_template == expected_template
            
            if complexity_correct:
                correct_complexity += 1
            if template_correct:
                correct_template += 1
            
            # Status indicators
            complexity_status = "✅" if complexity_correct else "❌"
            template_status = "✅" if template_correct else "❌"
            
            print(f"Test {i:2d}: {question[:50]}...")
            print(f"         Score: {complexity_score:.2f}")
            print(f"    {complexity_status} Complexity: {actual_level} (expected {expected_level})")
            print(f"    {template_status} Template: {actual_template} (expected {expected_template})")
            
            # Show analysis details for interesting cases
            if complexity_analysis.get("matched_patterns"):
                patterns = complexity_analysis["matched_patterns"][:3]  # Show first 3
                print(f"         Patterns: {', '.join(patterns)}")
            
            # Show additional insights
            insights = []
            if complexity_analysis.get("requires_aggregation"):
                insights.append("aggregation")
            if complexity_analysis.get("requires_time_analysis"):
                insights.append("time-analysis")
            if complexity_analysis.get("requires_ranking"):
                insights.append("ranking")
            
            if insights:
                print(f"         Features: {', '.join(insights)}")
            
            print()
            
        except Exception as e:
            print(f"❌ Test {i} failed: {str(e)}")
    
    # Summary
    print("📊 COMPLEXITY ANALYSIS RESULTS")
    print("=" * 40)
    print(f"Total Tests: {total_tests}")
    print(f"Complexity Detection Accuracy: {correct_complexity}/{total_tests} ({correct_complexity/total_tests*100:.1f}%)")
    print(f"Template Selection Accuracy: {correct_template}/{total_tests} ({correct_template/total_tests*100:.1f}%)")
    
    if correct_complexity == total_tests and correct_template == total_tests:
        print("\n🎉 ALL TESTS PASSED! Enhanced SQL Generator complexity analysis is working perfectly!")
    elif correct_complexity/total_tests >= 0.8 and correct_template/total_tests >= 0.8:
        print("\n✅ Most tests passed! Complexity analysis is working well with minor adjustments needed.")
    else:
        print("\n⚠️ Some tests failed. Review complexity indicators and template selection logic.")


def test_optimization_context():
    """
    Test optimization context generation
    """
    print("\n🔧 Testing Optimization Context Generation")
    print("=" * 45)
    
    analyzer = ComplexityAnalyzer()
    
    test_questions = [
        "Show me revenue by customer",
        "Get top 10 products with sales analysis and ranking",
        "Complex analytics with multiple joins, aggregations, and year over year comparisons"
    ]
    
    for question in test_questions:
        complexity_analysis = analyzer._analyze_query_complexity(question)
        template_choice = analyzer._select_template_by_complexity(complexity_analysis["complexity_score"])
        
        print(f"Question: {question}")
        print(f"Complexity: {complexity_analysis['complexity_level']} ({complexity_analysis['complexity_score']:.2f})")
        print(f"Template: {template_choice}")
        print(f"Features: ", end="")
        
        features = []
        if complexity_analysis.get("requires_aggregation"):
            features.append("aggregation")
        if complexity_analysis.get("requires_time_analysis"):
            features.append("time-analysis")
        if complexity_analysis.get("requires_ranking"):
            features.append("ranking")
        
        print(", ".join(features) if features else "basic query")
        print()


if __name__ == "__main__":
    test_complexity_analysis()
    test_optimization_context()
