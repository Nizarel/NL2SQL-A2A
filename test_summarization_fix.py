#!/usr/bin/env python3

"""
Test script to verify the summarization fix works
"""

import asyncio
import os
import sys

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from Host_Agent.nl2sql_client_agent import SemanticKernelNL2SQLAgent

async def test_summarization_fix():
    """Test that summarization no longer fails with KeyError"""
    
    print("🧪 Testing summarization fix...")
    print("=" * 60)
    
    # Initialize the agent
    agent = SemanticKernelNL2SQLAgent()
    
    try:
        # Simple test query that should complete the full workflow
        query = "Show me 5 rows from the cliente table"
        print(f"🔍 Test Query: {query}")
        print("=" * 60)
        
        response = await agent.chat(query)
        print(f"✅ Response: {response}")
        
        # Check if response indicates workflow failure
        if "Workflow failed" in str(response):
            print("❌ Summarization fix failed - workflow still failing")
            return False
        else:
            print("✅ Summarization fix successful - no workflow failure detected")
            return True
            
    except Exception as e:
        print(f"❌ Error during test: {e}")
        return False
    finally:
        # Clean up
        await agent.close()

async def main():
    """Main test function"""
    print("🎉 Agent initialization test starting...")
    
    success = await test_summarization_fix()
    
    if success:
        print("\n🎉 Summarization fix test PASSED!")
    else:
        print("\n❌ Summarization fix test FAILED!")
    
    return success

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
