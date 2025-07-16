"""
Test A2A integration with queries that should return data
"""
import asyncio
import sys
import os

# Add the src path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from Host_Agent.nl2sql_client_agent import SemanticKernelNL2SQLAgent

async def test_with_realistic_queries():
    """Test with queries that are more likely to return data"""
    
    agent = SemanticKernelNL2SQLAgent("http://localhost:8002")
    
    try:
        await agent.initialize()
        print("🎉 Agent initialized successfully!")
        
        # Test queries that are more likely to have data
        test_queries = [
            "What data do we have in the segmentacion table? Show me a sample of 5 rows.",
            "Show me the latest dates available in our database",
            "What are the top 5 customers by total revenue across all available time periods?",
            "Show me some sample customer data from the cliente table"
        ]
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n{'='*60}")
            print(f"🔍 Test Query {i}: {query}")
            print('='*60)
            
            try:
                response = await agent.chat(query)
                print(f"✅ Response: {response}")
            except Exception as e:
                print(f"❌ Query failed: {str(e)}")
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        await agent.close()

if __name__ == "__main__":
    asyncio.run(test_with_realistic_queries())
