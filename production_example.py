#!/usr/bin/env python3
"""
Production Example - How to use the NL2SQL Multi-Agent System
"""

import asyncio
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

from src.main import NL2SQLMultiAgentSystem


async def example_usage():
    """
    Example of how to use the NL2SQL system in production
    """
    # Method 1: Initialize manually
    print("🔧 Method 1: Manual initialization")
    system = NL2SQLMultiAgentSystem()
    await system.initialize()
    
    try:
        # Process a query
        result = await system.ask_question(
            question="Analyze Norte region profit performance by showing the top products for CEDIs in that region",
            execute=True,
            limit=10,
            include_summary=True
        )
        
        print(f"✅ Success: {result['success']}")
        if result['success']:
            data = result['data']
            print(f"📊 SQL: {data['sql_query'][:100]}...")
            print(f"📈 Rows: {result['metadata'].get('row_count', 0)}")
            if data.get('summary'):
                print(f"📝 Summary: {data['summary']['executive_summary'][:150]}...")
        
    finally:
        await system.close()
    
    print("\n" + "="*50)
    
    # Method 2: Use factory method (recommended for production)
    print("🚀 Method 2: Factory method (recommended)")
    system = await NL2SQLMultiAgentSystem.create_and_initialize()
    
    try:
        # Process multiple queries
        queries = [
            "Show top 5 customers by revenue",
            "What are the best selling products?",
            "Revenue analysis by region"
        ]
        
        for query in queries:
            print(f"\n🤔 Processing: {query}")
            result = await system.process_query(query, limit=5, include_summary=False)
            
            if result['success']:
                print(f"  ✅ Generated SQL successfully")
                if result['data'].get('executed'):
                    print(f"  📊 Retrieved {result['metadata'].get('row_count', 0)} rows")
                else:
                    print(f"  ⚠️ Query not executed: {result['data'].get('execution_error', 'Unknown')}")
            else:
                print(f"  ❌ Failed: {result['error']}")
    
    finally:
        await system.close()


async def production_api_example():
    """
    Example of how this could be used in a production API
    """
    print("\n" + "="*50)
    print("🏭 Production API Example")
    
    # Initialize once at application startup
    nl2sql_system = await NL2SQLMultiAgentSystem.create_and_initialize()
    
    def simulate_api_request(user_question: str, user_params: dict = None):
        """Simulates an API endpoint handler"""
        return asyncio.create_task(
            nl2sql_system.process_query(
                question=user_question,
                **(user_params or {})
            )
        )
    
    try:
        # Simulate concurrent API requests
        tasks = [
            simulate_api_request("Top 10 products by revenue", {"limit": 10}),
            simulate_api_request("Sales by region analysis", {"include_summary": True}),
            simulate_api_request("Customer segmentation overview", {"limit": 20})
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for i, result in enumerate(results, 1):
            if isinstance(result, Exception):
                print(f"  Request {i}: ❌ Failed with {type(result).__name__}")
            else:
                print(f"  Request {i}: ✅ Success - SQL generated and {'executed' if result['data'].get('executed') else 'not executed'}")
    
    finally:
        # Close at application shutdown
        await nl2sql_system.close()


if __name__ == "__main__":
    print("🧪 NL2SQL Production Examples")
    print("="*50)
    
    asyncio.run(example_usage())
    asyncio.run(production_api_example())
    
    print("\n🎉 Examples completed!")
