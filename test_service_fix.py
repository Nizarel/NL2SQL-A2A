#!/usr/bin/env python3

import asyncio
import sys
import os

# Add src to path
sys.path.insert(0, '/workspaces/NL2SQL/src')

from services.nl2sql_service import NL2SQLService

async def test_summarizing_fix():
    try:
        print('🚀 Starting NL2SQL service test...')
        service = NL2SQLService()
        
        question = 'Analyze Norte region profit performance by showing the top products for CEDIs in that region'
        print(f'🎯 Question: {question}')
        
        result = await service.process_query(question)
        
        print('📊 Final Analysis:')
        print(result)
        
    except Exception as e:
        print(f'❌ Error: {e}')
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_summarizing_fix())
