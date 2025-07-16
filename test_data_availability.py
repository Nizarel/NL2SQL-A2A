#!/usr/bin/env python3

"""
Test script to find what actual data exists in the database
"""

import asyncio
import os
import sys

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from Host_Agent.nl2sql_client_agent import SemanticKernelNL2SQLAgent

async def test_data_availability():
    """Test what data is actually available in the database"""
    
    print("🔍 Testing data availability...")
    print("=" * 60)
    
    # Initialize the agent
    agent = SemanticKernelNL2SQLAgent()
    
    try:
        # Test queries to find data
        test_queries = [
            "How many rows are in each table in the database?",
            "What's the data range in the segmentacion table?",
            "What's the data range in the tiempo table?",
            "Show me one record from any table that has data"
        ]
        
        for i, query in enumerate(test_queries, 1):
            print(f"🔍 Test {i}: {query}")
            print("-" * 50)
            
            response = await agent.chat(query)
            print(f"✅ Response: {response}")
            print("=" * 60)
            
            # Small delay between queries
            await asyncio.sleep(2)
            
    except Exception as e:
        print(f"❌ Error during test: {e}")
        return False
    finally:
        # Clean up
        await agent.close()
    
    return True

async def main():
    """Main test function"""
    print("🎉 Data availability test starting...")
    
    success = await test_data_availability()
    
    if success:
        print("\n🎉 Data availability test completed!")
    else:
        print("\n❌ Data availability test FAILED!")
    
    return success

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
