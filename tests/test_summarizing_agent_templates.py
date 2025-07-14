#!/usr/bin/env python3
"""
Test script to verify SummarizingAgent template integration
"""

import os
import sys
import asyncio

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
from agents.summarizing_agent import SummarizingAgent

async def test_summarizing_agent_templates():
    """Test that the SummarizingAgent templates load correctly"""
    
    print("Testing SummarizingAgent template integration...")
    
    try:
        # Create kernel with mock service (for testing template loading only)
        kernel = Kernel()
        
        # Add a mock OpenAI service for template validation
        mock_service = OpenAIChatCompletion(
            service_id="test-service",
            api_key="test-key",  # This won't be used for template loading test
            ai_model_id="gpt-4"
        )
        kernel.add_service(mock_service)
        
        # Create SummarizingAgent
        agent = SummarizingAgent(kernel)
        
        # Check if templates were loaded
        expected_templates = ['comprehensive_summary', 'insights_extraction', 'recommendations']
        
        print(f"Expected templates: {expected_templates}")
        print(f"Loaded templates: {list(agent.templates.keys())}")
        
        # Verify all templates are loaded
        missing_templates = []
        for template in expected_templates:
            if template not in agent.templates:
                missing_templates.append(template)
            else:
                print(f"✅ Template '{template}' loaded successfully")
        
        if missing_templates:
            print(f"❌ Missing templates: {missing_templates}")
            return False
        else:
            print("🎉 All SummarizingAgent templates loaded successfully!")
            return True
            
    except Exception as e:
        print(f"❌ Error testing SummarizingAgent: {str(e)}")
        return False

if __name__ == "__main__":
    result = asyncio.run(test_summarizing_agent_templates())
    sys.exit(0 if result else 1)
