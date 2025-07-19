"""
Test script for Enhanced SQL Generator Agent with complexity analysis and adaptive templating
"""

import asyncio
import os
import sys

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import AzureOpenAIChatCompletion
from agents.sql_generator_agent import SQLGeneratorAgent


async def test_enhanced_sql_generator():
    """
    Test the enhanced SQL Generator Agent with different complexity levels
    """
    
    # Initialize kernel and service (using placeholder values - replace with actual)
    kernel = Kernel()
    
    try:
        # Add your Azure OpenAI service (replace with actual values)
        service_id = "default"
        chat_service = AzureOpenAIChatCompletion(
            service_id=service_id,
            deployment_name="gpt-4",  # Replace with your deployment
            endpoint="your-endpoint",  # Replace with your endpoint
            api_key="your-api-key"     # Replace with your API key
        )
        kernel.add_service(chat_service)
        
        # Initialize enhanced SQL Generator Agent
        sql_agent = SQLGeneratorAgent(kernel)
        
        # Test cases with different complexity levels
        test_cases = [
            {
                "name": "Simple Query (Low Complexity)",
                "question": "Show me all customers",
                "expected_complexity": "LOW"
            },
            {
                "name": "Medium Complexity Query",
                "question": "Show me total revenue by customer for the last 6 months",
                "expected_complexity": "MEDIUM"
            },
            {
                "name": "High Complexity Query", 
                "question": "Show me the top 10 customers by revenue with year over year growth analysis, including monthly trends and ranking within their market segment",
                "expected_complexity": "HIGH"
            }
        ]
        
        # Mock schema context for testing
        mock_schema_context = """
        Tables available:
        - dev.cliente (customer_id, Nombre_cliente)
        - dev.segmentacion (customer_id, IngresoNetoSImpuestos, calday, material_id)
        - dev.producto (Material, Producto)
        - dev.tiempo (Fecha, YEAR, MONTH)
        """
        
        print("🧪 Testing Enhanced SQL Generator Agent")
        print("=" * 50)
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n📋 Test Case {i}: {test_case['name']}")
            print(f"Question: {test_case['question']}")
            print("-" * 40)
            
            # Prepare input data
            input_data = {
                "question": test_case["question"],
                "context": "Business analytics database",
                "optimized_schema_context": mock_schema_context
            }
            
            try:
                # Process with enhanced SQL Generator Agent
                result = await sql_agent.process(input_data)
                
                if result.get("success"):
                    data = result["data"]
                    metadata = result["metadata"]
                    
                    # Display complexity analysis
                    complexity_analysis = data.get("complexity_analysis", {})
                    print(f"🔍 Complexity Score: {complexity_analysis.get('complexity_score', 0):.2f}")
                    print(f"📊 Complexity Level: {complexity_analysis.get('complexity_level', 'N/A')}")
                    print(f"📋 Template Used: {metadata.get('template_used', 'N/A')}")
                    
                    # Display optimization context
                    optimization_context = data.get("optimization_context", {})
                    optimization_level = optimization_context.get("optimization_level", "N/A")
                    performance_hints = optimization_context.get("performance_hints", [])
                    
                    print(f"⚡ Optimization Level: {optimization_level}")
                    print(f"💡 Performance Hints: {len(performance_hints)} hints applied")
                    
                    if performance_hints:
                        for hint in performance_hints[:3]:  # Show first 3 hints
                            print(f"   - {hint}")
                    
                    # Display generated SQL (truncated for readability)
                    sql_query = data.get("sql_query", "")
                    sql_preview = sql_query[:200] + "..." if len(sql_query) > 200 else sql_query
                    print(f"🔧 Generated SQL: {sql_preview}")
                    
                    # Verify expected complexity
                    actual_complexity = complexity_analysis.get('complexity_level', '')
                    expected_complexity = test_case['expected_complexity']
                    
                    if actual_complexity == expected_complexity:
                        print(f"✅ Complexity detection: CORRECT ({expected_complexity})")
                    else:
                        print(f"⚠️ Complexity detection: Expected {expected_complexity}, got {actual_complexity}")
                
                else:
                    print(f"❌ Test failed: {result.get('error', 'Unknown error')}")
                
            except Exception as e:
                print(f"❌ Test error: {str(e)}")
        
        print("\n🎉 Enhanced SQL Generator Agent testing completed!")
        
    except Exception as e:
        print(f"❌ Setup error: {str(e)}")
        print("⚠️ Please configure your Azure OpenAI credentials in the test script")


async def test_complexity_analysis_only():
    """
    Test only the complexity analysis functionality without requiring OpenAI
    """
    print("\n🔍 Testing Complexity Analysis Only")
    print("=" * 40)
    
    # Create a kernel (won't be used for this test)
    kernel = Kernel()
    
    # Create SQL Generator Agent (templates won't load but complexity analysis will work)
    try:
        sql_agent = SQLGeneratorAgent(kernel)
    except:
        # If template loading fails, create a minimal instance for testing
        sql_agent = SQLGeneratorAgent.__new__(SQLGeneratorAgent)
        sql_agent.name = "SQLGeneratorAgent"
    
    # Test cases for complexity analysis
    test_questions = [
        ("Show me customers", "LOW"),
        ("Get total sales by region", "MEDIUM"), 
        ("Show me top 10 customers by revenue with year over year growth analysis including monthly trends", "HIGH"),
        ("List products", "LOW"),
        ("Average revenue by customer group with ranking and percentile analysis", "HIGH"),
        ("Join customer and product tables to show sales", "MEDIUM")
    ]
    
    for question, expected_level in test_questions:
        try:
            complexity_analysis = sql_agent._analyze_query_complexity(question)
            
            score = complexity_analysis["complexity_score"]
            level = complexity_analysis["complexity_level"]
            
            status = "✅" if level == expected_level else "⚠️"
            
            print(f"{status} '{question[:50]}...'")
            print(f"    Score: {score:.2f}, Level: {level} (Expected: {expected_level})")
            
            if complexity_analysis.get("matched_patterns"):
                print(f"    Patterns: {len(complexity_analysis['matched_patterns'])} matched")
                
        except Exception as e:
            print(f"❌ Error analyzing: {question} - {str(e)}")


if __name__ == "__main__":
    print("🚀 Enhanced SQL Generator Agent - Test Suite")
    print("=" * 60)
    
    # Run complexity analysis test (doesn't require OpenAI)
    asyncio.run(test_complexity_analysis_only())
    
    # Uncomment and configure to test full functionality
    # asyncio.run(test_enhanced_sql_generator())
