"""
Simple test client for NL2SQL A2A Agent
Demonstrates how to use the client agent to query the NL2SQL orchestrator
"""

import asyncio
import logging
import sys
import os
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Host_Agent.nl2sql_client_agent import NL2SQLClientAgent, SemanticKernelNL2SQLAgent
from semantic_kernel import Kernel

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


async def test_basic_client():
    """Test the basic NL2SQL client without Semantic Kernel wrapper"""
    print("🧪 Testing Basic NL2SQL Client Agent...")
    
    try:
        # Create kernel and client
        kernel = Kernel()
        client = NL2SQLClientAgent(kernel, "http://localhost:8002")
        
        # Initialize connection
        await client.initialize_connection()
        
        # Test capabilities
        print("\n📋 Getting agent capabilities...")
        capabilities = await client.get_capabilities()
        print(f"Agent: {capabilities['name']}")
        print(f"Description: {capabilities['description']}")
        print(f"Streaming: {capabilities['capabilities']['streaming']}")
        
        # Test queries
        test_queries = [
            "Show me sales data for 2025",
            "What are the top performing regions?",
            "Find products with declining sales"
        ]
        
        for query in test_queries:
            print(f"\n🤔 Query: {query}")
            result = await client.ask_question(query)
            print(f"📊 Result:\n{result}")
            print("-" * 50)
            
        await client.close()
        print("✅ Basic client test completed successfully!")
        
    except Exception as e:
        print(f"❌ Basic client test failed: {str(e)}")
        import traceback
        traceback.print_exc()


async def test_semantic_kernel_agent():
    """Test the full Semantic Kernel NL2SQL agent"""
    print("\n🧪 Testing Semantic Kernel NL2SQL Agent...")
    
    try:
        # Create and initialize agent
        agent = SemanticKernelNL2SQLAgent("http://localhost:8002")
        await agent.initialize()
        
        # Test conversation
        conversations = [
            "Hello! Can you help me analyze some data?",
            "Show me the revenue by region for 2025",
            "Which distribution centers are performing best?",
            "Can you find any sales trends in the data?"
        ]
        
        for message in conversations:
            print(f"\n👤 User: {message}")
            response = await agent.chat(message)
            print(f"🤖 Agent: {response}")
            print("-" * 50)
            
        await agent.close()
        print("✅ Semantic Kernel agent test completed successfully!")
        
    except Exception as e:
        print(f"❌ Semantic Kernel agent test failed: {str(e)}")
        import traceback
        traceback.print_exc()


async def test_streaming():
    """Test streaming responses"""
    print("\n🧪 Testing Streaming Responses...")
    
    try:
        kernel = Kernel()
        client = NL2SQLClientAgent(kernel, "http://localhost:8002")
        await client.initialize_connection()
        
        query = "Analyze customer purchase patterns and provide detailed insights"
        print(f"🤔 Streaming Query: {query}")
        print("📡 Streaming Response:")
        
        async for chunk in client.query_database_streaming(query):
            print(chunk, end='', flush=True)
        
        print("\n✅ Streaming test completed!")
        await client.close()
        
    except Exception as e:
        print(f"❌ Streaming test failed: {str(e)}")


async def interactive_mode():
    """Interactive mode for testing queries"""
    print("\n🎯 Interactive NL2SQL Client Mode")
    print("Type 'quit' or 'exit' to stop")
    print("Type 'help' for example queries")
    
    try:
        agent = SemanticKernelNL2SQLAgent("http://localhost:8002")
        await agent.initialize()
        
        print("✅ Connected to NL2SQL A2A Server!")
        
        while True:
            try:
                user_input = input("\n💬 Ask a data question: ").strip()
                
                if user_input.lower() in ['quit', 'exit']:
                    break
                elif user_input.lower() == 'help':
                    print_help()
                    continue
                elif not user_input:
                    continue
                
                print("🔄 Processing...")
                response = await agent.chat(user_input)
                print(f"\n📊 Response:\n{response}")
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"❌ Error: {str(e)}")
        
        await agent.close()
        print("\n👋 Goodbye!")
        
    except Exception as e:
        print(f"❌ Interactive mode failed: {str(e)}")


def print_help():
    """Print example queries"""
    examples = [
        "Show me sales data for 2025",
        "What are the top 5 products by revenue?",
        "Which regions have the highest growth?",
        "Find customers who haven't purchased in 6 months",
        "Analyze monthly sales trends this year",
        "Compare performance between different CEDIs",
        "Show me product categories with declining sales"
    ]
    
    print("\n📚 Example queries you can try:")
    for i, example in enumerate(examples, 1):
        print(f"  {i}. {example}")


async def main():
    """Main test function"""
    print("🚀 NL2SQL A2A Client Agent Test Suite")
    print("=" * 50)
    
    # Check if A2A server is running
    try:
        import httpx
        async with httpx.AsyncClient() as client:
            response = await client.get("http://localhost:8002/agent-card", timeout=5.0)
            response.raise_for_status()
        print("✅ A2A server is running")
    except Exception as e:
        print(f"❌ A2A server not accessible: {str(e)}")
        print("Please start the A2A server first:")
        print("  cd src/")
        print("  python pure_a2a_server.py --host localhost --port 8002")
        return
    
    # Run tests
    test_mode = input("\nSelect test mode:\n1. Basic Client Test\n2. Semantic Kernel Test\n3. Streaming Test\n4. Interactive Mode\n5. All Tests\nChoice (1-5): ").strip()
    
    if test_mode == "1":
        await test_basic_client()
    elif test_mode == "2":
        await test_semantic_kernel_agent()
    elif test_mode == "3":
        await test_streaming()
    elif test_mode == "4":
        await interactive_mode()
    elif test_mode == "5":
        await test_basic_client()
        await test_semantic_kernel_agent()
        await test_streaming()
    else:
        print("Invalid choice. Running all tests...")
        await test_basic_client()
        await test_semantic_kernel_agent()
        await test_streaming()
    
    print("\n🎉 Test suite completed!")


if __name__ == "__main__":
    asyncio.run(main())
