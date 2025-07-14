#!/usr/bin/env python3
"""
Quick test for the NL2SQL system
"""
import asyncio
import sys
import os

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))

async def test_nl2sql_system():
    """Test the NL2SQL system with a question"""
    try:
        from src.main import NL2SQLMultiAgentSystem
        
        print("🚀 Testing NL2SQL System...")
        
        agent = NL2SQLMultiAgentSystem()
        await agent.initialize()
        
        # Initialize the agent
        await agent.initialize()
        print("✅ Agent initialized successfully!")
        
        # Test with the question
        question = "Show me the top 3 customers with their names and total revenue"
        print(f"\n🤔 Question: {question}")
        print("🔄 Converting to SQL...")
        
        result = await agent.ask_question(question, execute=True, limit=5)
        
        print(f"🔍 Debug - result type: {type(result)}")
        print(f"🔍 Debug - result content: {result}")
        
        if isinstance(result, dict) and result.get("error") is None:
            print(f"\n✅ SUCCESS!")
            if 'sql_query' in result:
                print(f"📝 Generated SQL: {result['sql_query']}")
            if result.get("executed") and result.get("results"):
                print(f"\n📊 Results:\n{result['results']}")
                print(f"⏱️  Execution time: {result.get('execution_time')}s")
                print(f"📈 Rows returned: {result.get('row_count')}")
        elif isinstance(result, dict):
            print(f"\n❌ ERROR: {result.get('error', 'Unknown error')}")
        else:
            print(f"\n📊 Direct Result: {result}")
            
        # Test with our Norte region question to verify the SummarizingAgent fix
        print("\n" + "="*60)
        print("🧪 Testing Norte region profit analysis (our fix)...")
        
        norte_question = "Analyze Norte region profit performance by showing the top products for CEDIs in that region"
        norte_result = await agent.ask_question(norte_question, execute=True, limit=10)
        
        print(f"🎯 Norte Question: {norte_question}")
        print(f"🔍 Norte Debug - result type: {type(norte_result)}")
        print(f"🔍 Norte Debug - result content: {norte_result}")
        
        if isinstance(norte_result, dict) and norte_result.get("error") is None:
            print(f"✅ Norte SUCCESS!")
            if 'sql_query' in norte_result:
                print(f"📝 Generated SQL: {norte_result['sql_query']}")
            if norte_result.get("executed") and norte_result.get("results"):
                print(f"\n📊 Norte Results:\n{norte_result['results']}")
                print(f"⏱️  Norte Execution time: {norte_result.get('execution_time')}s")
                print(f"📈 Norte Rows returned: {norte_result.get('row_count')}")
        elif isinstance(norte_result, dict):
            print(f"❌ Norte ERROR: {norte_result.get('error', 'Unknown error')}")
        else:
            print(f"📊 Norte Direct Result: {norte_result}")
            
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
    finally:
        try:
            await agent.close()
            print("🔐 Agent closed successfully")
        except:
            pass

if __name__ == "__main__":
    asyncio.run(test_nl2sql_system())
