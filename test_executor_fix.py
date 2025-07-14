#!/usr/bin/env python3
import asyncio
import sys
import os

sys.path.insert(0, '/workspaces/NL2SQL/src')

from src.main import NL2SQLMultiAgentSystem

async def test_fix():
    try:
        print('🧪 Testing ExecutorAgent fix...')
        agent = NL2SQLMultiAgentSystem()
        await agent.initialize()
        
        question = 'Analyze Norte region profit performance by showing the top products for CEDIs in that region'
        print(f'🎯 Question: {question}')
        
        result = await agent.ask_question(question, execute=True, limit=10)
        
        # Check if data is properly formatted
        formatted_results = result['data']['formatted_results']
        print(f'📊 Formatted Results Status: {formatted_results.get("status")}')
        print(f'📊 Rows Found: {len(formatted_results.get("rows", []))}')
        print(f'📊 Headers: {formatted_results.get("headers", [])}')
        
        if formatted_results.get('rows'):
            print('✅ SUCCESS - Data properly parsed!')
            print('📋 Sample rows:')
            for i, row in enumerate(formatted_results['rows'][:3]):
                print(f'   Row {i+1}: {row}')
                
            # Check if SummarizingAgent now gets correct insights
            summary = result['data']['summary']
            print(f'📝 Executive Summary: {summary["executive_summary"]}')
        else:
            print('❌ Still no data in formatted results')
        
        await agent.close()
        
    except Exception as e:
        print(f'❌ Error: {e}')
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_fix())
