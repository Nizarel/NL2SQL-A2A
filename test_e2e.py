#!/usr/bin/env python3
"""
End-to-End Test Script for NL2SQL Multi-Agent System
Testing specific user questions with detailed workflow tracking
"""

import sys
import os
import asyncio
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from dotenv import load_dotenv
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion

from plugins.mcp_database_plugin import MCPDatabasePlugin
from services.schema_service import SchemaService
from agents.sql_generator_agent import SQLGeneratorAgent
from agents.executor_agent import ExecutorAgent
from agents.summarizing_agent import SummarizingAgent
from agents.orchestrator_agent import OrchestratorAgent


async def test_nl2sql_system():
    """Test the NL2SQL system with specific user questions"""
    
    # Load environment variables
    dotenv_path = os.path.join(os.path.dirname(__file__), 'src', '.env')
    load_dotenv(dotenv_path, override=True)
    
    # Test questions from user
    test_questions = [
        "Which CEDI has the highest dispatch volume in march 2025?",
        "What are the best-selling products in May 2025?",
        "Which product or category has generated the highest profit in the last quarter for each CEDI in January 2025 in Norte region?"
    ]
    
    try:
        # Initialize system
        print("🚀 INITIALIZING NL2SQL MULTI-AGENT SYSTEM")
        print("=" * 70)
        
        kernel = Kernel()
        
        # Setup Azure OpenAI
        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")
        azure_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
        azure_api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
        
        if not all([azure_endpoint, azure_api_key, azure_deployment]):
            print("❌ Missing Azure OpenAI configuration")
            return
        
        ai_service = AzureChatCompletion(
            deployment_name=azure_deployment,
            endpoint=azure_endpoint,
            api_key=azure_api_key,
            api_version=azure_api_version,
            service_id="azure_openai"
        )
        kernel.add_service(ai_service)
        print("✅ Azure OpenAI service configured")
        
        # Initialize MCP Plugin
        mcp_server_url = os.getenv("MCP_SERVER_URL", "https://azsql-fastmcpserv2.jollyfield-479bc951.eastus2.azurecontainerapps.io/mcp/")
        mcp_plugin = MCPDatabasePlugin(mcp_server_url=mcp_server_url)
        kernel.add_plugin(mcp_plugin, plugin_name="database")
        print("✅ MCP Database Plugin configured")
        
        # Initialize Schema Service
        schema_service = SchemaService(mcp_plugin)
        await schema_service.initialize_schema_context()
        print("✅ Schema Service initialized")
        
        # Initialize Agents
        sql_generator = SQLGeneratorAgent(kernel, schema_service)
        executor = ExecutorAgent(kernel, mcp_plugin)
        summarizer = SummarizingAgent(kernel)
        orchestrator = OrchestratorAgent(kernel, sql_generator, executor, summarizer)
        print("✅ All agents initialized")
        
        # Test each question
        print(f"\n🧪 TESTING {len(test_questions)} USER QUESTIONS")
        print("=" * 70)
        
        for i, question in enumerate(test_questions, 1):
            print(f"\n📋 TEST {i}/{len(test_questions)}")
            print(f"Question: {question}")
            print("-" * 50)
            
            # Process question through orchestrator
            result = await orchestrator.process({
                "question": question,
                "execute": True,
                "limit": 10,
                "include_summary": True,
                "context": ""
            })
            
            if result["success"]:
                print("✅ WORKFLOW COMPLETED SUCCESSFULLY!")
                
                data = result["data"]
                metadata = result["metadata"]
                
                # Step results
                print(f"\n🔍 WORKFLOW STEPS COMPLETED:")
                for step in metadata.get("steps_completed", []):
                    print(f"  ✅ {step}")
                
                print(f"\n📝 STEP 1 - SQL GENERATION:")
                print(f"Generated SQL: {data.get('sql_query', 'N/A')}")
                
                if data.get("executed"):
                    print(f"\n⚡ STEP 2 - SQL EXECUTION:")
                    print(f"✅ Query executed successfully")
                    print(f"⏱️  Execution time: {metadata.get('execution_time', 'N/A')}s")
                    print(f"📈 Rows returned: {metadata.get('row_count', 'N/A')}")
                    print(f"🔧 Query type: {metadata.get('query_type', 'N/A')}")
                    
                    # Show sample results
                    results = str(data.get('results', 'N/A'))
                    if len(results) > 300:
                        results = results[:300] + "..."
                    print(f"\n📊 RESULTS PREVIEW:\n{results}")
                    
                    if data.get("summary"):
                        print(f"\n🧠 STEP 3 - AI SUMMARIZATION:")
                        summary = data["summary"]
                        print(f"Executive Summary: {summary.get('executive_summary', 'N/A')}")
                        
                        insights = summary.get('key_insights', [])
                        if insights:
                            print(f"\nKey Insights ({len(insights)}):")
                            for j, insight in enumerate(insights[:3], 1):  # Show first 3
                                print(f"  {j}. {insight.get('finding', 'N/A')}")
                        
                        recommendations = summary.get('recommendations', [])
                        if recommendations:
                            print(f"\nRecommendations ({len(recommendations)}):")
                            for j, rec in enumerate(recommendations[:2], 1):  # Show first 2
                                print(f"  {j}. {rec.get('action', 'N/A')} (Priority: {rec.get('priority', 'N/A')})")
                else:
                    print(f"\n❌ STEP 2 - SQL EXECUTION FAILED:")
                    print(f"Error: {data.get('execution_error', 'Unknown error')}")
                    
                print(f"\n📊 WORKFLOW METADATA:")
                print(f"  Total time: {metadata.get('total_workflow_time', 'N/A')}s")
                print(f"  Orchestration type: {metadata.get('orchestration_type', 'N/A')}")
                print(f"  Pattern: {metadata.get('orchestration_pattern', 'N/A')}")
                
            else:
                print(f"❌ WORKFLOW FAILED: {result['error']}")
            
            print("\n" + "=" * 70)
        
        print(f"\n🎯 END-TO-END TESTING COMPLETED!")
        print(f"✅ Tested {len(test_questions)} questions")
        print(f"🤖 Sequential Orchestration Pattern: SQLGenerator → Executor → Summarizer")
        print(f"📦 Using Semantic Kernel 1.34.0 with manual orchestration fallback")
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("🧪 STARTING END-TO-END NL2SQL TESTING...")
    asyncio.run(test_nl2sql_system())
