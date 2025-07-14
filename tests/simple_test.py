#!/usr/bin/env python3
"""
Simple test script for the NL2SQL system question
"""
import asyncio
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

async def test_simple_question():
    """Test with a simple question"""
    from main import NL2SQLAgent
    
    print("🚀 Testing NL2SQL System...")
    
    agent = NL2SQLAgent()
    
    try:
        # Initialize the agent
        await agent.initialize()
        print("✅ Agent initialized successfully!")
        
        # Test with the question
        question = "Show me the top 5 customers with their names and total revenue"
        print(f"\n🤔 Question: {question}")
        print("🔄 Converting to SQL...")
        
        result = await agent.ask_question(question, execute=True, limit=10)
        
        if result["error"] is None:
            print(f"\n✅ SUCCESS!")
            print(f"📝 Generated SQL: {result['sql_query']}")
            if result["executed"] and result["results"]:
                print(f"\n📊 Results:\n{result['results']}")
                print(f"⏱️  Execution time: {result['execution_time']}s")
                print(f"📈 Rows returned: {result['row_count']}")
            else:
                print("📊 Query was generated but not executed or returned no results")
        else:
            print(f"\n❌ ERROR: {result['error']}")
            
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
    finally:
        try:
            await agent.close()
        except:
            pass

if __name__ == "__main__":
    asyncio.run(test_simple_question())
