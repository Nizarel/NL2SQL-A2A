#!/usr/bin/env python3
import asyncio
import sys
import os

sys.path.insert(0, '/workspaces/NL2SQL/src')

from src.main import NL2SQLMultiAgentSystem

async def debug_sk_responses():
    try:
        print('🔍 Debugging Semantic Kernel agent responses...')
        agent = NL2SQLMultiAgentSystem()
        await agent.initialize()
        
        # Access the orchestrator to debug its SK workflow
        orchestrator = agent.orchestrator_agent
        
        if not orchestrator.agent_group_chat:
            print("❌ No SK agent group chat available")
            return
            
        # Test with a simple question to see response structure
        params = {
            'question': 'Show me top 3 customers',
            'execute': False,  # Don't execute, just generate SQL
            'limit': 5,
            'include_summary': False
        }
        
        schema_context = orchestrator.sql_generator.schema_service.get_full_schema_summary()
        
        workflow_prompt = f"""
SEQUENTIAL NL2SQL WORKFLOW REQUEST:
Question: {params['question']}
Database Schema Context: {schema_context[:500]}...

WORKFLOW STEPS:
1. SQLGeneratorAgent: Generate SQL query only
2. ExecutorAgent: Skip execution
3. SummarizingAgent: Skip summary
"""
        
        # Add user message
        from semantic_kernel.contents import ChatMessageContent, AuthorRole
        user_message = ChatMessageContent(
            role=AuthorRole.USER, 
            content=workflow_prompt
        )
        await orchestrator.agent_group_chat.add_chat_message(user_message)
        
        # Get responses and examine structure
        print("🤖 Getting agent responses...")
        agent_responses = []
        async for response in orchestrator.agent_group_chat.invoke():
            print(f"\\n📋 Response #{len(agent_responses) + 1}:")
            print(f"  Type: {type(response)}")
            print(f"  Has 'name': {hasattr(response, 'name')}")
            print(f"  Has 'content': {hasattr(response, 'content')}")
            print(f"  Has 'role': {hasattr(response, 'role')}")
            
            if hasattr(response, 'name'):
                print(f"  Name: {response.name}")
            if hasattr(response, 'content'):
                print(f"  Content (first 200 chars): {str(response.content)[:200]}...")
            if hasattr(response, 'role'):
                print(f"  Role: {response.role}")
                
            # Show all available attributes
            print(f"  All attributes: {[attr for attr in dir(response) if not attr.startswith('_')]}")
            
            agent_responses.append(response)
            
            if len(agent_responses) >= 1:  # Just get first response for debugging
                break
        
        print(f"\\n✅ Collected {len(agent_responses)} agent responses")
        await agent.close()
        
    except Exception as e:
        print(f'❌ Error: {e}')
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(debug_sk_responses())
