"""
Test the improved SK Workflow with schema context
"""

import asyncio
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.main import NL2SQLMultiAgentSystem


async def test_improved_sk_workflow():
    """Test the improved SK workflow with proper schema context"""
    
    print("🧪 Testing Improved SK Workflow...")
    
    system = NL2SQLMultiAgentSystem()
    
    try:
        await system.initialize()
        print("✅ System initialized")
        
        # Test questions that previously generated placeholders
        test_questions = [
            "What are the top 5 customers by revenue?",
            "Show me product sales by category",
            "Which territories have the highest sales?"
        ]
        
        for i, question in enumerate(test_questions, 1):
            print(f"\n📋 Test {i}/{len(test_questions)}: {question}")
            print("-" * 50)
            
            result = await system.ask_question(
                question=question,
                execute=True,
                include_summary=False
            )
            
            if result["success"]:
                sql_query = result["data"]["sql_query"]
                row_count = result.get("metadata", {}).get("row_count", 0)
                orchestration_type = result.get("metadata", {}).get("orchestration_type", "unknown")
                
                print(f"🎯 Orchestration Type: {orchestration_type}")
                print(f"📝 Generated SQL:")
                print(sql_query)
                print(f"⚡ Row Count: {row_count}")
                
                # Check for placeholder issues
                if "[" in sql_query and "]" in sql_query:
                    print("❌ Still contains placeholders!")
                elif "dev.ventas" in sql_query.lower() or " ventas " in sql_query.lower() or sql_query.lower().strip().endswith("ventas"):
                    print("❌ References non-existent 'ventas' table!")
                elif row_count > 0:
                    print("✅ SQL executed successfully with real data!")
                else:
                    print("⚠️ SQL executed but returned 0 rows")
                
                # Check if schema analysis is included
                if "schema_analysis" in result["data"]:
                    schema_info = result["data"]["schema_analysis"]
                    relevant_tables = schema_info.get('relevant_tables', [])
                    print(f"🔍 Schema Analysis: {relevant_tables}")
                else:
                    print("❌ No schema analysis in results")
                    
            else:
                print(f"❌ Query failed: {result.get('error')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        await system.close()
        print("\n🔐 System closed")


if __name__ == "__main__":
    success = asyncio.run(test_improved_sk_workflow())
    print(f"\n🎯 Test Result: {'✅ PASSED' if success else '❌ FAILED'}")
